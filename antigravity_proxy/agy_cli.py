"""Backend adapter using persistent ``agy acp --stdio`` workers for low latency."""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Iterator

from .config import MODEL_MAP, ProxyConfig
from .translator import content_to_text

logger = logging.getLogger("antigravity_proxy.agy_cli")

_ACP_TIMEOUT_BUFFER = 15


class AgyCliError(Exception):
    """Error raised when invoking or parsing output from the ``agy`` CLI."""

    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


def parse_agy_usage(usage_raw: dict | None) -> dict:
    """Convert ``agy`` CLI/ACP usage dict (snake_case or camelCase) to OpenAI counters."""
    usage = usage_raw or {}
    pt = int(usage.get("inputTokens") or usage.get("input_tokens") or 0)
    ct = int(usage.get("outputTokens") or usage.get("output_tokens") or 0)
    tt = int(usage.get("totalTokens") or usage.get("total_tokens") or (pt + ct))
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
    }


def format_messages_for_agy(messages: list[dict]) -> str:
    """Flatten OpenAI ``messages`` into a single prompt string for ``agy``."""
    system_parts: list[str] = []
    turns: list[tuple[str, str]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "user")).lower()
        text = content_to_text(msg.get("content"))
        if not text:
            continue
        if role in ("system", "developer"):
            system_parts.append(text)
        else:
            turns.append((role, text))

    if not turns:
        return ""

    sections: list[str] = []
    if system_parts:
        sections.append("System instructions:\n" + "\n\n".join(system_parts))

    if len(turns) == 1 and turns[0][0] == "user":
        sections.append(turns[0][1])
    else:
        transcript = "\n\n".join(f"[{role.capitalize()}]\n{text}" for role, text in turns)
        sections.append(
            "Conversation history:\n"
            + transcript
            + "\n\nRespond directly to the latest user message."
        )

    return "\n\n".join(sections)


def _platform_subprocess_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


class AgyAcpWorker:
    """Persistent ``agy acp --stdio`` JSON-RPC worker to avoid per-request cold starts."""

    def __init__(self, exe: str, config: ProxyConfig) -> None:
        self._exe = exe
        self._config = config
        self._proc: subprocess.Popen | None = None
        self._tmp_dir: tempfile.TemporaryDirectory | None = None
        self._cwd: str | None = None
        self._req_id = 0
        self._initialized = False

    def _resolve_cwd(self) -> str:
        if self._config.agy_cwd and os.path.isdir(self._config.agy_cwd):
            return self._config.agy_cwd
        if self._tmp_dir is None:
            self._tmp_dir = tempfile.TemporaryDirectory(prefix="agy-acp-")
        return self._tmp_dir.name

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def ensure_started(self) -> None:
        """Start and initialize the ACP stdio server if not already running."""
        if self.is_alive() and self._initialized:
            return
        self.close_process()

        self._cwd = self._resolve_cwd()
        cmd = [
            self._exe,
            "acp",
            "--stdio",
            "--disable-telemetry",
            "--include-default-customizations=false",
            "--cli_log_file",
            os.devnull,
        ]
        self._proc = subprocess.Popen(
            cmd,
            cwd=self._cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **_platform_subprocess_kwargs(),
        )
        self._req_id = 0
        self._initialized = False
        self._rpc(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {},
                "clientInfo": {"name": "antigravity-proxy", "version": "2.0.0"},
            },
        )
        self._initialized = True

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _write_json(self, obj: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps(obj) + "\n")
            self._proc.stdin.flush()
        except OSError as exc:
            raise AgyCliError(f"Failed writing to agy ACP stdin: {exc}", status=502) from exc

    def _read_until_response(self, req_id: int) -> Iterator[dict]:
        """Yield notification updates until the JSON-RPC response for ``req_id`` arrives."""
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                code = self._proc.poll()
                raise AgyCliError(
                    f"agy ACP worker exited unexpectedly (code={code})",
                    status=502,
                )
            stripped = line.strip()
            if not stripped:
                continue
            try:
                msg = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            if msg.get("id") == req_id:
                if "error" in msg and msg["error"]:
                    err = msg["error"]
                    err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    raise AgyCliError(f"agy ACP error: {err_msg}", status=502)
                yield {"__rpc_result__": msg.get("result") or {}}
                return

            # Reject any server-initiated tool permission requests immediately.
            if "id" in msg and "method" in msg:
                self._write_json({
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "error": {
                        "code": -32000,
                        "message": "Tool execution is disabled in proxy mode",
                    },
                })
                continue

            if msg.get("method") == "session/update":
                yield msg

    def _rpc(self, method: str, params: dict) -> dict:
        req_id = self._next_id()
        self._write_json({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        })
        for item in self._read_until_response(req_id):
            if "__rpc_result__" in item:
                return item["__rpc_result__"]
        return {}

    def stream_prompt(self, prompt: str, model: str) -> Iterator[dict]:
        """Run one isolated ACP session and yield ``text`` and final ``usage`` dicts."""
        self.ensure_started()
        assert self._cwd is not None

        new_res = self._rpc("session/new", {"cwd": self._cwd, "mcpServers": []})
        session_id = new_res.get("sessionId")
        if not session_id:
            raise AgyCliError(f"agy ACP session/new missing sessionId: {new_res}", status=502)

        try:
            current_model = (new_res.get("models") or {}).get("currentModelId")
            if model and model != current_model:
                try:
                    self._rpc("session/set_model", {"sessionId": session_id, "modelId": model})
                except AgyCliError as exc:
                    raise AgyCliError(f"Invalid or unavailable model '{model}': {exc}", status=400) from exc

            req_id = self._next_id()
            self._write_json({
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": prompt}],
                },
            })

            for item in self._read_until_response(req_id):
                if "__rpc_result__" in item:
                    result = item["__rpc_result__"]
                    yield {
                        "finish_reason": "stop",
                        "usage": parse_agy_usage(result.get("usage")),
                    }
                    break

                upd = (item.get("params") or {}).get("update") or {}
                if upd.get("sessionUpdate") == "agent_message_chunk":
                    content = upd.get("content") or {}
                    if content.get("type") == "text" and content.get("text"):
                        yield {"text": content["text"]}
        finally:
            if self.is_alive():
                try:
                    self._rpc("session/close", {"sessionId": session_id})
                except Exception:
                    pass

    def close_process(self) -> None:
        """Terminate the ACP child process and close stdio pipes."""
        proc = self._proc
        self._proc = None
        self._initialized = False
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)

    def close(self) -> None:
        """Shut down the worker and remove its temporary working directory."""
        self.close_process()
        if self._tmp_dir is not None:
            try:
                self._tmp_dir.cleanup()
            except OSError:
                pass
            self._tmp_dir = None


class AgyCliClient:
    """Thread-safe pool of warm ``agy acp --stdio`` workers exposing OpenAI chat."""

    def __init__(self, config: ProxyConfig) -> None:
        self._config = config
        self._cached_models: list[str] | None = None
        self._max_workers = max(1, config.max_concurrency) if config.max_concurrency >= 0 else 1
        self._pool: queue.LifoQueue[AgyAcpWorker] = queue.LifoQueue()
        self._lock = threading.Lock()
        self._created = 0
        self._closed = False

    def resolve_bin(self) -> str | None:
        """Locate the ``agy`` executable across PATH and standard Windows/Linux dirs."""
        found = shutil.which(self._config.agy_bin)
        if found:
            return found

        if self._config.agy_bin != "agy":
            return self._config.agy_bin if os.path.isfile(self._config.agy_bin) else None

        candidates: list[str] = []
        if os.name == "nt":
            local_app_data = os.environ.get("LOCALAPPDATA", "")
            if local_app_data:
                candidates.append(os.path.join(local_app_data, "agy", "bin", "agy.exe"))
            candidates.append(
                os.path.expanduser(os.path.join("~", "AppData", "Local", "agy", "bin", "agy.exe"))
            )
        else:
            candidates.extend([
                os.path.expanduser("~/.local/bin/agy"),
                "/usr/local/bin/agy",
                "/usr/bin/agy",
            ])

        for path in candidates:
            if path and os.path.isfile(path):
                return path
        return None

    def preflight_check(self) -> bool:
        """Verify the ``agy`` executable is available on this machine."""
        resolved = self.resolve_bin()
        if not resolved:
            logger.error("`%s` executable not found in PATH or standard locations.", self._config.agy_bin)
            logger.error("Install Antigravity CLI (`agy`) or pass --agy-bin.")
            return False
        logger.info("Using official agy CLI binary: %s", resolved)
        return True

    def warm_up(self) -> None:
        """Pre-start one ACP worker so the first incoming request avoids boot latency."""
        worker = self._acquire_worker()
        try:
            worker.ensure_started()
        finally:
            self._release_worker(worker, broken=False)

    def _acquire_worker(self) -> AgyAcpWorker:
        try:
            return self._pool.get_nowait()
        except queue.Empty:
            pass

        with self._lock:
            if self._created < self._max_workers:
                exe = self.resolve_bin()
                if not exe:
                    raise AgyCliError(f"`{self._config.agy_bin}` executable not found", status=500)
                self._created += 1
                return AgyAcpWorker(exe, self._config)

        return self._pool.get()

    def _release_worker(self, worker: AgyAcpWorker, *, broken: bool) -> None:
        if self._closed or broken or not worker.is_alive():
            worker.close()
            with self._lock:
                self._created = max(0, self._created - 1)
            return
        self._pool.put(worker)

    def list_models(self) -> list[str]:
        """Return model IDs reported by ``agy models``, falling back to MODEL_MAP."""
        if self._cached_models is not None:
            return self._cached_models

        exe = self.resolve_bin()
        if exe:
            try:
                proc = subprocess.run(
                    [exe, "models"],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=15,
                    check=False,
                    **_platform_subprocess_kwargs(),
                )
                if proc.returncode == 0:
                    models = [
                        line.split()[0]
                        for line in proc.stdout.splitlines()
                        if line.strip()
                    ]
                    if models:
                        self._cached_models = models
                        return models
            except Exception as exc:
                logger.warning("Failed to list models via `agy models`: %s", exc)

        return list(MODEL_MAP.keys())

    def call_chat(self, body: dict, model: str) -> dict:
        """Execute a non-streaming OpenAI chat completion via a warm ACP worker."""
        chunks: list[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        finish_reason = "stop"

        for item in self.stream_chat(body, model):
            if item.get("text"):
                chunks.append(item["text"])
            if "finish_reason" in item:
                finish_reason = item["finish_reason"]
            if "usage" in item:
                usage = item["usage"]

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model") or model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "".join(chunks)},
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }

    def stream_chat(self, body: dict, model: str) -> Iterator[dict]:
        """Stream OpenAI delta/usage dicts via a warm ACP worker."""
        prompt = format_messages_for_agy(body.get("messages") or [])
        if not prompt:
            raise AgyCliError("No messages provided", status=400)

        worker = self._acquire_worker()
        broken = False
        try:
            yield from worker.stream_prompt(prompt, model)
        except AgyCliError as exc:
            broken = exc.status >= 500
            raise
        except Exception:
            broken = True
            raise
        finally:
            self._release_worker(worker, broken=broken)

    def close(self) -> None:
        """Shut down all pooled ACP workers and clean up temp directories."""
        self._closed = True
        while True:
            try:
                worker = self._pool.get_nowait()
            except queue.Empty:
                break
            worker.close()
