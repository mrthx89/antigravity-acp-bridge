"""Backend adapter delegating OpenAI chat completions to the official ``agy`` CLI."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import nullcontext
from typing import Iterator

from .config import MODEL_MAP, ProxyConfig
from .translator import content_to_text

logger = logging.getLogger("antigravity_proxy.agy_cli")


class AgyCliError(Exception):
    """Error raised when invoking or parsing output from the ``agy`` CLI."""

    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


def parse_agy_usage(usage_raw: dict | None) -> dict:
    """Convert ``agy`` CLI usage dict to OpenAI ``usage`` counters."""
    usage = usage_raw or {}
    pt = int(usage.get("input_tokens", 0) or 0)
    ct = int(usage.get("output_tokens", 0) or 0)
    tt = int(usage.get("total_tokens", pt + ct) or (pt + ct))
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
    }


def format_messages_for_agy(messages: list[dict]) -> str:
    """Flatten OpenAI ``messages`` into a single prompt string for ``agy -p``."""
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


class AgyCliClient:
    """Runs non-interactive ``agy -p`` calls and translates outputs to OpenAI format."""

    def __init__(self, config: ProxyConfig) -> None:
        self._config = config
        self._cached_models: list[str] | None = None
        self._semaphore = (
            threading.BoundedSemaphore(config.max_concurrency)
            if config.max_concurrency > 0
            else None
        )

    def _concurrency_guard(self):
        return self._semaphore if self._semaphore is not None else nullcontext()

    def resolve_bin(self) -> str | None:
        """Locate the ``agy`` executable across PATH and standard Windows/Linux dirs."""
        found = shutil.which(self._config.agy_bin)
        if found:
            return found

        if self._config.agy_bin != "agy":
            if os.path.isfile(self._config.agy_bin):
                return self._config.agy_bin
            return None

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

    def _build_command(
        self,
        exe: str,
        prompt: str,
        model: str,
        output_format: str,
    ) -> list[str]:
        cmd = [
            exe,
            "-p",
            prompt,
            "--model",
            model,
            "--output-format",
            output_format,
            "--disable-slash-commands",
            "--print-timeout",
            f"{self._config.upstream_timeout}s",
        ]
        if self._config.project_id_override:
            cmd.extend(["--project", self._config.project_id_override])
        return cmd

    def _work_dir(self) -> tuple[str, tempfile.TemporaryDirectory | None]:
        if self._config.agy_cwd and os.path.isdir(self._config.agy_cwd):
            return self._config.agy_cwd, None
        # Isolate each call in an empty temp dir so workspace files are not loaded.
        tmp = tempfile.TemporaryDirectory(prefix="agy-proxy-")
        return tmp.name, tmp

    def call_chat(self, body: dict, model: str) -> dict:
        """Execute ``agy -p --output-format json`` and return an OpenAI response."""
        exe = self.resolve_bin()
        if not exe:
            raise AgyCliError(f"`{self._config.agy_bin}` executable not found", status=500)

        prompt = format_messages_for_agy(body.get("messages") or [])
        if not prompt:
            raise AgyCliError("No messages provided", status=400)

        cmd = self._build_command(exe, prompt, model, "json")
        with self._concurrency_guard():
            cwd, tmp = self._work_dir()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=cwd,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self._config.upstream_timeout + 15,
                    check=False,
                    **_platform_subprocess_kwargs(),
                )
            except subprocess.TimeoutExpired as exc:
                raise AgyCliError("agy CLI timed out", status=504) from exc
            finally:
                if tmp is not None:
                    tmp.cleanup()

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise AgyCliError(
                f"agy CLI failed (exit {proc.returncode}): {detail[:500]}",
                status=502,
            )

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise AgyCliError(
                f"agy CLI returned invalid JSON: {proc.stdout[:300]}",
                status=502,
            ) from exc

        if payload.get("status") != "SUCCESS":
            raise AgyCliError(
                f"agy CLI returned status={payload.get('status')}: {payload}",
                status=502,
            )

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model") or model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": payload.get("response", ""),
                },
                "finish_reason": "stop",
            }],
            "usage": parse_agy_usage(payload.get("usage")),
        }

    def stream_chat(self, body: dict, model: str) -> Iterator[dict]:
        """Run ``agy -p --output-format stream-json`` and yield delta/usage dicts."""
        exe = self.resolve_bin()
        if not exe:
            raise AgyCliError(f"`{self._config.agy_bin}` executable not found", status=500)

        prompt = format_messages_for_agy(body.get("messages") or [])
        if not prompt:
            raise AgyCliError("No messages provided", status=400)

        cmd = self._build_command(exe, prompt, model, "stream-json")
        with self._concurrency_guard():
            cwd, tmp = self._work_dir()
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **_platform_subprocess_kwargs(),
            )

            emitted_text = False
            try:
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("event")
                    if etype == "step_update":
                        step = event.get("step_update") or {}
                        delta_text = step.get("text_delta")
                        if delta_text:
                            emitted_text = True
                            yield {"text": delta_text}
                    elif etype == "result":
                        res = event.get("result") or {}
                        if res.get("status") != "SUCCESS":
                            raise AgyCliError(
                                f"agy CLI stream status={res.get('status')}",
                                status=502,
                            )
                        if not emitted_text and res.get("response"):
                            emitted_text = True
                            yield {"text": res["response"]}
                        yield {
                            "finish_reason": "stop",
                            "usage": parse_agy_usage(res.get("usage")),
                        }

                returncode = proc.wait(timeout=15)
                if returncode != 0:
                    stderr_text = proc.stderr.read().strip() if proc.stderr else ""
                    raise AgyCliError(
                        f"agy CLI stream failed (exit {returncode}): {stderr_text[:500]}",
                        status=502,
                    )
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5)
                if proc.stdout is not None:
                    proc.stdout.close()
                if proc.stderr is not None:
                    proc.stderr.close()
                if tmp is not None:
                    tmp.cleanup()
