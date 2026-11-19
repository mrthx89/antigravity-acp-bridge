"""Backend adapter using persistent ``agy acp --stdio`` workers for low latency."""

from __future__ import annotations

import base64
import collections
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
import urllib.request
import uuid
from typing import Iterator

from .config import BridgeConfig, MODEL_MAP, ProxyConfig
from .translator import content_to_text

logger = logging.getLogger("antigravity_acp_bridge.agy_cli")

_TOOL_CALL_TAG = "<openai_tool_calls>"
_TOOL_CALL_CLOSE_TAG = "</openai_tool_calls>"
_TOOL_CALL_RE = re.compile(
    r"<openai_tool_calls>\s*(.*?)\s*</openai_tool_calls>",
    re.DOTALL,
)
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)
_ACCOUNT_LINK_RE = re.compile(
    r"https://accounts\.google\.com/signin/[^\s<>'\"`]+auth_success_gemini[^\s<>'\"`]*",
    re.IGNORECASE,
)
_TRANSIENT_ACP_HINTS = (
    "429",
    "503",
    "resource_exhausted",
    "capacity",
    "overloaded",
    "temporarily unavailable",
    "exited unexpectedly",
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


def extract_images_for_agy(messages: list[dict]) -> list[dict]:
    """Extract OpenAI image_url / input_image parts into ACP image ContentBlocks."""
    images: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "image_url":
                iu = part.get("image_url")
                url = iu.get("url") if isinstance(iu, dict) else (iu if isinstance(iu, str) else "")
                if not url:
                    continue
                if url.startswith("data:"):
                    try:
                        header, b64 = url.split(",", 1)
                        mime = header[5:].split(";")[0] or "image/png"
                        images.append({"type": "image", "data": b64, "mimeType": mime})
                    except Exception:
                        pass
                elif url.startswith(("http://", "https://")):
                    try:
                        req = urllib.request.Request(url, headers={"User-Agent": "antigravity-acp-bridge"})
                        with urllib.request.urlopen(req, timeout=15) as r:
                            raw = r.read()
                            mime = r.headers.get("Content-Type", "image/png").split(";")[0]
                        images.append({
                            "type": "image",
                            "data": base64.b64encode(raw).decode("ascii"),
                            "mimeType": mime,
                        })
                    except Exception as exc:
                        logger.warning("Failed to fetch image URL '%s': %s", url, exc)
            elif ptype in ("image", "input_image"):
                data = part.get("data") or part.get("image") or ""
                mime = part.get("mimeType") or part.get("media_type") or "image/png"
                if data:
                    images.append({"type": "image", "data": data, "mimeType": mime})
    return images


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
    fence_match = _FENCE_RE.match(raw_json)
    if fence_match:
        raw_json = fence_match.group(1).strip()

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


def _trailing_tag_prefix_len(text: str, tag: str) -> int:
    """Return length of the longest suffix of ``text`` that is a prefix of ``tag``."""
    max_len = min(len(text), len(tag) - 1)
    for size in range(max_len, 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


def _platform_subprocess_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def _is_transient_acp_error(exc: AgyCliError) -> bool:
    if exc.status in (429, 502, 503, 504):
        msg = str(exc).lower()
        return any(hint in msg for hint in _TRANSIENT_ACP_HINTS)
    return False


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
        self._stderr_lines: collections.deque[str] = collections.deque(maxlen=40)
        self._stdout_queue: queue.Queue[str | None] | None = None
        self._use_stdout_reader = False

    def _resolve_cwd(self) -> str:
        if self._config.agy_cwd and os.path.isdir(self._config.agy_cwd):
            return self._config.agy_cwd
        if self._tmp_dir is None:
            try:
                self._tmp_dir = tempfile.TemporaryDirectory(prefix="agy-acp-", ignore_cleanup_errors=True)
            except TypeError:
                self._tmp_dir = tempfile.TemporaryDirectory(prefix="agy-acp-")
        return self._tmp_dir.name

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _recent_stderr(self) -> str:
        if not self._stderr_lines:
            return ""
        return " | ".join(line.strip() for line in list(self._stderr_lines)[-6:] if line.strip())

    def _start_stream_drainers(self, proc: subprocess.Popen) -> None:
        self._stderr_lines.clear()
        stderr = getattr(proc, "stderr", None)
        if stderr is not None:
            def _drain_stderr() -> None:
                try:
                    for raw in stderr:
                        line = raw.rstrip()
                        if line:
                            self._stderr_lines.append(line)
                            logger.debug("agy stderr: %s", line)
                except OSError:
                    pass

            threading.Thread(target=_drain_stderr, name="agy-stderr", daemon=True).start()

        stdout = getattr(proc, "stdout", None)
        # Only spin up a blocking reader thread for real OS pipes (which expose fileno()).
        # In-memory test doubles without fileno() are read synchronously on demand.
        if stdout is not None and hasattr(stdout, "fileno"):
            q: queue.Queue[str | None] = queue.Queue()
            self._stdout_queue = q
            self._use_stdout_reader = True

            def _drain_stdout() -> None:
                try:
                    for raw in stdout:
                        q.put(raw)
                except OSError:
                    pass
                finally:
                    q.put(None)

            threading.Thread(target=_drain_stdout, name="agy-stdout", daemon=True).start()
        else:
            self._stdout_queue = None
            self._use_stdout_reader = False

    def _spawn_and_initialize(self, cmd: list[str]) -> None:
        self._proc = subprocess.Popen(
            cmd,
            cwd=self._cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **_platform_subprocess_kwargs(),
        )
        self._start_stream_drainers(self._proc)
        self._req_id = 0
        self._initialized = False
        self._rpc(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {},
                "clientInfo": {"name": "antigravity-proxy", "version": "2.0.0"},
            },
            timeout=min(30.0, float(self._config.upstream_timeout)),
        )
        self._initialized = True

    def ensure_started(self) -> None:
        """Start and initialize the ACP stdio server if not already running."""
        if self.is_alive() and self._initialized:
            return
        self.close_process()

        self._cwd = self._resolve_cwd()
        preferred_cmd = [
            self._exe,
            "acp",
            "--stdio",
            "--disable-telemetry",
            "--include-default-customizations=false",
            "--cli_log_file",
            os.devnull,
        ]
        try:
            self._spawn_and_initialize(preferred_cmd)
        except AgyCliError as exc:
            err_lower = str(exc).lower()
            # Older/alternate Linux builds of agy may not recognize every optional flag.
            if "flag" in err_lower or "unknown" in err_lower:
                logger.warning(
                    "agy ACP rejected optional flags (%s); retrying with bare `agy acp --stdio`",
                    exc,
                )
                self.close_process()
                self._spawn_and_initialize([self._exe, "acp", "--stdio"])
            else:
                self.close_process()
                raise

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _write_json(self, obj: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps(obj) + "\n")
            self._proc.stdin.flush()
        except OSError as exc:
            detail = self._recent_stderr()
            suffix = f" (stderr: {detail})" if detail else ""
            raise AgyCliError(f"Failed writing to agy ACP stdin: {exc}{suffix}", status=502) from exc

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

    def _read_line(self, deadline: float) -> str | None:
        if self._use_stdout_reader and self._stdout_queue is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = self._recent_stderr()
                suffix = f" (stderr: {detail})" if detail else ""
                self.close_process()
                raise AgyCliError(f"agy ACP timed out waiting for response{suffix}", status=504)
            try:
                return self._stdout_queue.get(timeout=remaining)
            except queue.Empty as exc:
                detail = self._recent_stderr()
                suffix = f" (stderr: {detail})" if detail else ""
                self.close_process()
                raise AgyCliError(
                    f"agy ACP timed out waiting for response{suffix}", status=504
                ) from exc

        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        return line if line else None

    def _read_until_response(self, req_id: int, *, timeout: float) -> Iterator[dict]:
        """Yield notification updates until the JSON-RPC response for ``req_id`` arrives."""
        deadline = time.monotonic() + max(1.0, timeout)
        while True:
            line = self._read_line(deadline)
            if line is None:
                code = self._proc.poll() if self._proc is not None else None
                detail = self._recent_stderr()
                suffix = f" (stderr: {detail})" if detail else ""
                raise AgyCliError(
                    f"agy ACP worker exited unexpectedly (code={code}){suffix}",
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
                    link_match = _ACCOUNT_LINK_RE.search(err_msg) or _ACCOUNT_LINK_RE.search(self._recent_stderr())
                    if link_match:
                        err_msg = f"{err_msg}. Google authentication required: {link_match.group(0)}"
                    raise AgyCliError(f"agy ACP error: {err_msg}", status=502)
                yield {"__rpc_result__": msg.get("result") or {}}
                return

            if "id" in msg and "method" in msg:
                self._reply_permission_denied(msg)
                yield {"__permission_denied__": True}
                continue

            if msg.get("method") == "session/update":
                yield msg

    def _rpc(self, method: str, params: dict, *, timeout: float = 30.0) -> dict:
        req_id = self._next_id()
        self._write_json({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        })
        for item in self._read_until_response(req_id, timeout=timeout):
            if "__rpc_result__" in item:
                return item["__rpc_result__"]
        return {}

    def _run_prompt_turn(
        self,
        session_id: str,
        prompt_text: str,
        image_blocks: list[dict] | None = None,
    ) -> Iterator[dict]:
        req_id = self._next_id()
        prompt_blocks: list[dict] = [{"type": "text", "text": prompt_text}]
        if image_blocks:
            prompt_blocks.extend(image_blocks)
        self._write_json({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": prompt_blocks,
            },
        })
        for item in self._read_until_response(
            req_id, timeout=float(self._config.upstream_timeout)
        ):
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
        image_blocks: list[dict] | None = None,
    ) -> Iterator[dict]:
        """Run one isolated ACP session and yield ``text`` and final ``usage`` dicts."""
        self.ensure_started()
        assert self._cwd is not None

        control_timeout = min(45.0, float(self._config.upstream_timeout))
        session_params: dict = {
            "cwd": self._cwd,
            "mcpServers": [],
            "_meta": {"antigravity": {"disabledTools": ["schedule", "run_workflow"]}},
        }
        try:
            new_res = self._rpc("session/new", session_params, timeout=control_timeout)
        except AgyCliError as exc:
            # Fallback for older ACP builds that reject unrecognized _meta fields
            if "_meta" in str(exc) or "unknown" in str(exc).lower():
                new_res = self._rpc(
                    "session/new", {"cwd": self._cwd, "mcpServers": []}, timeout=control_timeout
                )
            else:
                raise
        session_id = new_res.get("sessionId")
        if not session_id:
            raise AgyCliError(f"agy ACP session/new missing sessionId: {new_res}", status=502)

        try:
            current_model = (new_res.get("models") or {}).get("currentModelId")
            if model and model != current_model:
                try:
                    self._rpc(
                        "session/set_model",
                        {"sessionId": session_id, "modelId": model},
                        timeout=control_timeout,
                    )
                except AgyCliError as exc:
                    raise AgyCliError(f"Invalid or unavailable model '{model}': {exc}", status=400) from exc

            emitted_text = False
            saw_denied_tool = False
            final_result: dict = {}

            head_buf = ""
            head_settled = False

            def _check_banner(text: str) -> None:
                low = text.lower()
                if "terms of service" in low or "violation of terms" in low or "service has been disabled" in low:
                    raise AgyCliError(
                        "Google Antigravity service disabled: account has Terms of Service violation.",
                        status=403,
                    )
                if "verification required" in low or "verify your account" in low:
                    link_match = _ACCOUNT_LINK_RE.search(text) or _ACCOUNT_LINK_RE.search(self._recent_stderr())
                    link_txt = f" Please open: {link_match.group(0)}" if link_match else ""
                    raise AgyCliError(
                        f"Google Account verification required.{link_txt}",
                        status=403,
                    )

            suspicious_prefixes = (
                "this service",
                "terms of service",
                "verification",
                "verify",
                "usage limit",
                "agent execution error",
            )

            for item in self._run_prompt_turn(session_id, prompt, image_blocks=image_blocks):
                if "__rpc_result__" in item:
                    final_result = item["__rpc_result__"]
                elif item.get("__permission_denied__"):
                    saw_denied_tool = True
                elif item.get("text"):
                    chunk_text = item["text"]
                    if not head_settled:
                        head_buf += chunk_text
                        _check_banner(head_buf)
                        low_buf = head_buf.lower().strip()
                        if any(low_buf.startswith(p) for p in suspicious_prefixes):
                            if len(head_buf) >= 160 or "\n" in head_buf:
                                head_settled = True
                                emitted_text = True
                                yield {"text": head_buf}
                                head_buf = ""
                        else:
                            head_settled = True
                            emitted_text = True
                            yield {"text": head_buf}
                            head_buf = ""
                    else:
                        emitted_text = True
                        yield {"text": chunk_text}

            if head_buf:
                _check_banner(head_buf)
                emitted_text = True
                yield {"text": head_buf}
                head_buf = ""

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
                    self._rpc("session/close", {"sessionId": session_id}, timeout=10.0)
                except Exception:
                    pass

    def close_process(self) -> None:
        """Terminate the ACP child process and close stdio pipes."""
        proc = self._proc
        self._proc = None
        self._initialized = False
        self._stdout_queue = None
        self._use_stdout_reader = False
        if proc is None:
            return
        for stream in (
            getattr(proc, "stdin", None),
            getattr(proc, "stdout", None),
            getattr(proc, "stderr", None),
        ):
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
        # 0 or negative means "no cap": every concurrent request may hold its own worker.
        # Positive values bound the pool to that many warm ``agy`` subprocesses.
        self._max_workers: int | None = (
            config.max_concurrency if config.max_concurrency > 0 else None
        )
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._idle: list[tuple[AgyAcpWorker, float]] = []
        self._created = 0
        self._closed = False
        # Idle workers are closed after this many seconds so an unlimited pool does not
        # keep peak-burst memory parked forever. 0 disables the reaper.
        self._idle_timeout = float(config.worker_idle_timeout)
        self._stop = threading.Event()
        self._reaper: threading.Thread | None = None
        if self._idle_timeout > 0:
            self._reaper = threading.Thread(
                target=self._reap_idle_workers, name="agy-idle-reaper", daemon=True
            )
            self._reaper.start()

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
                os.path.expanduser("~/.antigravity/bin/agy"),
                "/usr/local/bin/agy",
                "/usr/bin/agy",
            ])

        for path in candidates:
            if path and os.path.isfile(path):
                return path
        return None

    def preflight_check(self) -> bool:
        """Verify the ``agy`` executable is available and diagnose auth issues early."""
        resolved = self.resolve_bin()
        if not resolved:
            logger.error("`%s` executable not found in PATH or standard locations.", self._config.agy_bin)
            logger.error("Install Antigravity CLI (`agy`) or pass --agy-bin.")
            return False
        logger.info("Using official agy CLI binary: %s", resolved)

        try:
            proc = subprocess.run(
                [resolved, "models"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
                **_platform_subprocess_kwargs(),
            )
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip()
                logger.warning(
                    "`agy models` exited with code %d: %s",
                    proc.returncode,
                    detail[:300] or "(no output)",
                )
                logger.warning(
                    "If running on Linux headless, authenticate first with `agy` or `agy acp --login`."
                )
            else:
                models = [line.split()[0] for line in proc.stdout.splitlines() if line.strip()]
                if models:
                    self._cached_models = models
        except Exception as exc:
            logger.warning("Preflight `agy models` check failed: %s", exc)

        return True

    def warm_up(self) -> None:
        """Pre-start one ACP worker so the first incoming request avoids boot latency."""
        worker = self._acquire_worker()
        broken = False
        try:
            worker.ensure_started()
        except Exception:
            broken = True
            raise
        finally:
            self._release_worker(worker, broken=broken)

    def _acquire_worker(self) -> AgyAcpWorker:
        spawn_exe: str | None = None
        with self._cond:
            while True:
                if self._idle:
                    worker, _idle_since = self._idle.pop()
                    return worker
                if self._max_workers is None or self._created < self._max_workers:
                    spawn_exe = self.resolve_bin()
                    if not spawn_exe:
                        raise AgyCliError(
                            f"`{self._config.agy_bin}` executable not found", status=500
                        )
                    self._created += 1
                    break
                if self._closed:
                    raise AgyCliError("agy worker pool is closed", status=503)
                self._cond.wait()

        return AgyAcpWorker(spawn_exe, self._config)

    def _release_worker(self, worker: AgyAcpWorker, *, broken: bool) -> None:
        should_close = False
        with self._cond:
            if self._closed or broken or not worker.is_alive():
                self._created = max(0, self._created - 1)
                should_close = True
            else:
                self._idle.append((worker, time.monotonic()))
            self._cond.notify()
        if should_close:
            worker.close()

    def _reap_idle_workers(self) -> None:
        """Close workers that sat idle past ``worker_idle_timeout``, shrinking the pool."""
        tick = max(1.0, min(30.0, self._idle_timeout / 2))
        while not self._stop.wait(tick):
            now = time.monotonic()
            expired: list[AgyAcpWorker] = []
            still_warm = 0
            with self._cond:
                kept: list[tuple[AgyAcpWorker, float]] = []
                for worker, idle_since in self._idle:
                    if now - idle_since >= self._idle_timeout:
                        expired.append(worker)
                    else:
                        kept.append((worker, idle_since))
                self._idle = kept
                still_warm = len(kept)
                if expired:
                    self._created = max(0, self._created - len(expired))
                    self._cond.notify_all()

            for worker in expired:
                worker.close()
            if expired:
                logger.info(
                    "Idle reaper closed %d worker(s) after %.0fs idle; %d still warm",
                    len(expired),
                    self._idle_timeout,
                    still_warm,
                )

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

    def _stream_with_tool_parsing(self, raw_gen: Iterator[dict]) -> Iterator[dict]:
        """Stream chunks while detecting ``<openai_tool_calls>`` anywhere in the output."""
        buf = ""
        in_tool_tag = False
        emitted_tool_calls = False
        tool_idx = 0

        for item in raw_gen:
            if item.get("text"):
                buf += item["text"]
                while buf:
                    if not in_tool_tag:
                        start = buf.find(_TOOL_CALL_TAG)
                        if start != -1:
                            if start > 0:
                                yield {"text": buf[:start]}
                                buf = buf[start:]
                            in_tool_tag = True
                            continue

                        keep = _trailing_tag_prefix_len(buf, _TOOL_CALL_TAG)
                        safe_end = len(buf) - keep
                        if safe_end > 0:
                            yield {"text": buf[:safe_end]}
                            buf = buf[safe_end:]
                        break

                    # Inside <openai_tool_calls>: wait until closing tag arrives.
                    close_idx = buf.find(_TOOL_CALL_CLOSE_TAG)
                    if close_idx == -1:
                        break
                    block_end = close_idx + len(_TOOL_CALL_CLOSE_TAG)
                    block = buf[:block_end]
                    buf = buf[block_end:]
                    in_tool_tag = False

                    rem, tcs = parse_openai_tool_calls(block)
                    if tcs:
                        indexed = []
                        for tc in tcs:
                            indexed.append(dict(tc, index=tool_idx))
                            tool_idx += 1
                        if rem:
                            yield {"text": rem}
                        emitted_tool_calls = True
                        yield {"tool_calls": indexed}
                    else:
                        # Malformed JSON inside tag: preserve raw text instead of dropping it.
                        yield {"text": block}

            elif "finish_reason" in item:
                if buf:
                    rem, tcs = parse_openai_tool_calls(buf)
                    if tcs:
                        indexed = []
                        for tc in tcs:
                            indexed.append(dict(tc, index=tool_idx))
                            tool_idx += 1
                        if rem:
                            yield {"text": rem}
                        emitted_tool_calls = True
                        yield {"tool_calls": indexed}
                    else:
                        yield {"text": buf}
                    buf = ""
                if emitted_tool_calls:
                    item = dict(item, finish_reason="tool_calls")
                yield item

    def stream_chat(self, body: dict, model: str) -> Iterator[dict]:
        """Stream OpenAI delta/tool_call/usage dicts via a warm ACP worker."""
        prompt = format_request_for_agy(body)
        if not prompt:
            raise AgyCliError("No messages provided", status=400)

        image_blocks = extract_images_for_agy(body.get("messages") or [])

        has_external_tools = bool(
            body.get("tool_choice") != "none" and _compact_openai_tools(body.get("tools"))
        )

        max_attempts = 2
        for attempt in range(max_attempts):
            worker = self._acquire_worker()
            broken = False
            yielded_any = False
            try:
                raw_gen = worker.stream_prompt(
                    prompt,
                    model,
                    has_external_tools=has_external_tools,
                    image_blocks=image_blocks,
                )
                stream = (
                    self._stream_with_tool_parsing(raw_gen)
                    if has_external_tools
                    else raw_gen
                )
                for item in stream:
                    yielded_any = True
                    yield item
                return
            except AgyCliError as exc:
                broken = exc.status >= 500
                if not yielded_any and attempt + 1 < max_attempts and _is_transient_acp_error(exc):
                    logger.warning(
                        "Transient agy ACP error (attempt %d/%d): %s — retrying with fresh worker",
                        attempt + 1,
                        max_attempts,
                        exc,
                    )
                    time.sleep(1.5)
                    continue
                raise
            except Exception:
                broken = True
                raise
            finally:
                self._release_worker(worker, broken=broken)

    def close(self) -> None:
        """Shut down all pooled ACP workers and clean up temp directories."""
        with self._cond:
            self._closed = True
            self._stop.set()
            to_close = [worker for worker, _ in self._idle]
            self._idle.clear()
            self._created = max(0, self._created - len(to_close))
            self._cond.notify_all()

        reaper = self._reaper
        if reaper is not None and reaper.is_alive() and reaper is not threading.current_thread():
            reaper.join(timeout=5)
        for worker in to_close:
            worker.close()
