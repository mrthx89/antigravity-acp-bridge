"""Backend adapter using persistent ``agy acp --stdio`` workers for low latency."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
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

_TOOL_CALL_TAG = "<openai_tool_calls>"
_TOOL_CALL_CLOSE_TAG = "</openai_tool_calls>"
_TOOL_CALL_RE = re.compile(
    r"<openai_tool_calls>\s*(.*?)\s*</openai_tool_calls>",
    re.DOTALL,
)


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


def _compact_openai_tools(tools: list | None) -> list[dict]:
    compact: list[dict] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        entry: dict = {"name": fn["name"], "description": fn.get("description", "")}
        if isinstance(fn.get("parameters"), dict) and fn["parameters"]:
            entry["parameters"] = fn["parameters"]
        compact.append(entry)
    return compact


def format_messages_for_agy(messages: list[dict]) -> str:
    """Flatten OpenAI ``messages`` (including tool call/result turns) into text."""
    system_parts: list[str] = []
    turns: list[str] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "user")).lower()
        text = content_to_text(msg.get("content"))

        if role in ("system", "developer"):
            if text:
                system_parts.append(text)
            continue

        if role == "tool":
            name = msg.get("name") or msg.get("tool_call_id") or "tool"
            turns.append(f"[Tool Result ({name})]\n{text}")
            continue

        if role == "assistant":
            parts: list[str] = []
            if text:
                parts.append(text)
            tcs = msg.get("tool_calls") or []
            if tcs:
                compact_tcs = []
                for tc in tcs:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    compact_tcs.append({
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "{}"),
                    })
                if compact_tcs:
                    parts.append(f"{_TOOL_CALL_TAG}{json.dumps(compact_tcs)}{_TOOL_CALL_CLOSE_TAG}")
            if parts:
                turns.append("[Assistant]\n" + "\n".join(parts))
            continue

        if text:
            turns.append(f"[User]\n{text}" if len(turns) > 0 or system_parts else text)

    if not turns:
        return ""

    sections: list[str] = []
    if system_parts:
        sections.append("System instructions:\n" + "\n\n".join(system_parts))

    if len(turns) == 1 and not turns[0].startswith("["):
        sections.append(turns[0])
    else:
        sections.append("Conversation history:\n" + "\n\n".join(turns))

    return "\n\n".join(sections)


def format_request_for_agy(body: dict) -> str:
    """Build an ACP prompt from an OpenAI chat body, including external tool specs."""
    base_prompt = format_messages_for_agy(body.get("messages") or [])
    if not base_prompt:
        return ""

    tool_choice = body.get("tool_choice")
    tools = [] if tool_choice == "none" else _compact_openai_tools(body.get("tools"))

    rules = [
        "CRITICAL RUNTIME RULES:",
        "1. NEVER invoke any built-in IDE/agent tools (such as run_command, view_file, write_to_file, list_dir, grep_search, find_by_name, etc.). Built-in tool execution is disabled.",
    ]
    if tools:
        rules.append(
            "2. External OpenAI tools are provided below by the client. When you need to call one or more external tools, respond ONLY with:\n"
            f'{_TOOL_CALL_TAG}[{{"name": "<tool_name>", "arguments": {{...}}}}]{_TOOL_CALL_CLOSE_TAG}\n'
            "Do not add prose outside those tags when calling tools."
        )
        if tool_choice == "required":
            rules.append("3. You MUST call at least one external tool from the list below.")
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            forced_name = (tool_choice.get("function") or {}).get("name")
            if forced_name:
                rules.append(f"3. You MUST call the external tool `{forced_name}`.")
        else:
            rules.append("3. If no external tool is needed, reply directly in plain text.")
        rules.append(f"Available external OpenAI tools:\n{json.dumps(tools, ensure_ascii=False)}")
    else:
        rules.append("2. Respond directly in plain text without calling any tools.")

    return "\n".join(rules) + "\n\n" + base_prompt


def parse_openai_tool_calls(text: str) -> tuple[str | None, list[dict]]:
    """Extract ``<openai_tool_calls>...</openai_tool_calls>`` into OpenAI format."""
    match = _TOOL_CALL_RE.search(text)
    if not match:
        return (text, [])

    raw_json = match.group(1).strip()
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return (text, [])

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return (text, [])

    tool_calls: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name")
        if not name:
            continue
        args = fn.get("arguments", {})
        args_str = json.dumps(args) if isinstance(args, (dict, list)) else str(args)
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": args_str},
        })

    if not tool_calls:
        return (text, [])

    remaining = (text[: match.start()] + text[match.end() :]).strip()
    return (remaining or None, tool_calls)


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

    def _reply_permission_denied(self, msg: dict) -> None:
        options = (msg.get("params") or {}).get("options") or []
        reject_id = next(
            (
                opt.get("optionId")
                for opt in options
                if "reject" in str(opt.get("kind", "")).lower()
                or "deny" in str(opt.get("optionId", "")).lower()
            ),
            None,
        )
        outcome = (
            {"outcome": "selected", "optionId": reject_id}
            if reject_id
            else {"outcome": "cancelled"}
        )
        self._write_json({"jsonrpc": "2.0", "id": msg["id"], "result": {"outcome": outcome}})

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

            if "id" in msg and "method" in msg:
                self._reply_permission_denied(msg)
                yield {"__permission_denied__": True}
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

    def _run_prompt_turn(
        self,
        session_id: str,
        prompt_text: str,
    ) -> Iterator[dict]:
        req_id = self._next_id()
        self._write_json({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": prompt_text}],
            },
        })
        for item in self._read_until_response(req_id):
            if "__rpc_result__" in item:
                yield item
                return
            if item.get("__permission_denied__"):
                yield item
                continue
            upd = (item.get("params") or {}).get("update") or {}
            if upd.get("sessionUpdate") == "agent_message_chunk":
                content = upd.get("content") or {}
                if content.get("type") == "text" and content.get("text"):
                    yield {"text": content["text"]}

    def stream_prompt(
        self,
        prompt: str,
        model: str,
        *,
        has_external_tools: bool = False,
    ) -> Iterator[dict]:
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

            emitted_text = False
            saw_denied_tool = False
            final_result: dict = {}

            for item in self._run_prompt_turn(session_id, prompt):
                if "__rpc_result__" in item:
                    final_result = item["__rpc_result__"]
                elif item.get("__permission_denied__"):
                    saw_denied_tool = True
                elif item.get("text"):
                    emitted_text = True
                    yield {"text": item["text"]}

            # If agy attempted a built-in tool and ended with zero text, ask once in-session
            # for a plain-text or <openai_tool_calls> reply so clients don't get empty responses.
            if not emitted_text:
                recovery_prompt = (
                    f"Built-in tools are disabled. Output your response in plain text or use {_TOOL_CALL_TAG}...{_TOOL_CALL_CLOSE_TAG} from the provided external tools."
                    if has_external_tools
                    else "Built-in tools are disabled here. Please answer directly in plain text without invoking any tools."
                )
                for item in self._run_prompt_turn(session_id, recovery_prompt):
                    if "__rpc_result__" in item:
                        final_result = item["__rpc_result__"]
                    elif item.get("text"):
                        emitted_text = True
                        yield {"text": item["text"]}

            if not emitted_text:
                fallback = (
                    "Unable to execute built-in tools in proxy mode; please rephrase or enable client tools."
                    if saw_denied_tool
                    else "Model completed without text output."
                )
                yield {"text": fallback}

            yield {
                "finish_reason": "stop",
                "usage": parse_agy_usage(final_result.get("usage")),
            }
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
        tool_calls: list[dict] = []

        for item in self.stream_chat(body, model):
            if item.get("text"):
                chunks.append(item["text"])
            if item.get("tool_calls"):
                tool_calls.extend(item["tool_calls"])
            if "finish_reason" in item:
                finish_reason = item["finish_reason"]
            if "usage" in item:
                usage = item["usage"]

        text = "".join(chunks) if chunks else None
        message: dict = {"role": "assistant", "content": text}
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model") or model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }

    def stream_chat(self, body: dict, model: str) -> Iterator[dict]:
        """Stream OpenAI delta/tool_call/usage dicts via a warm ACP worker."""
        prompt = format_request_for_agy(body)
        if not prompt:
            raise AgyCliError("No messages provided", status=400)

        has_external_tools = bool(
            body.get("tool_choice") != "none" and _compact_openai_tools(body.get("tools"))
        )

        worker = self._acquire_worker()
        broken = False
        try:
            raw_gen = worker.stream_prompt(
                prompt, model, has_external_tools=has_external_tools
            )
            if not has_external_tools:
                yield from raw_gen
                return

            # When external tools exist, hold leading text only while it could be <openai_tool_calls>.
            buf = ""
            streaming_plain = False
            emitted_tool_calls = False
            for item in raw_gen:
                if item.get("text"):
                    if streaming_plain:
                        yield item
                        continue
                    buf += item["text"]
                    stripped = buf.lstrip()
                    if not stripped:
                        continue
                    if _TOOL_CALL_TAG.startswith(stripped) or stripped.startswith(_TOOL_CALL_TAG):
                        if _TOOL_CALL_CLOSE_TAG in buf:
                            rem, tcs = parse_openai_tool_calls(buf)
                            buf = ""
                            if tcs:
                                indexed = [dict(tc, index=i) for i, tc in enumerate(tcs)]
                                if rem:
                                    yield {"text": rem}
                                emitted_tool_calls = True
                                yield {"tool_calls": indexed}
                                streaming_plain = True
                            else:
                                streaming_plain = True
                        continue
                    # Clearly not a <openai_tool_calls> block; flush immediately.
                    streaming_plain = True
                    yield {"text": buf}
                    buf = ""
                elif "finish_reason" in item:
                    if buf:
                        rem, tcs = parse_openai_tool_calls(buf)
                        if tcs:
                            indexed = [dict(tc, index=i) for i, tc in enumerate(tcs)]
                            if rem:
                                yield {"text": rem}
                            emitted_tool_calls = True
                            yield {"tool_calls": indexed}
                        else:
                            yield {"text": buf}
                    if emitted_tool_calls:
                        item = dict(item, finish_reason="tool_calls")
                    yield item
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
