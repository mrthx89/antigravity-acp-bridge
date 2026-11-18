"""Upstream HTTP client for Cloud Code Assist generateContent endpoints."""

from __future__ import annotations

import json
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from ._http import build_antigravity_request, read_http_error_detail
from .auth import TokenManager
from .config import GENERATE_CONTENT_URL, STREAM_GENERATE_CONTENT_URL, UPSTREAM_TIMEOUT
from .sse import parse_remaining, try_parse_one_event


class UpstreamError(Exception):
    """Error raised when the Cloud Code Assist backend fails or rejects a call."""

    def __init__(self, message: str, status: int = 502, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class UpstreamClient:
    """Thin client wrapping non-streaming and SSE streaming upstream calls."""

    def __init__(
        self,
        token_manager: TokenManager,
        timeout: int = UPSTREAM_TIMEOUT,
    ) -> None:
        self._token_manager = token_manager
        self._timeout = timeout

    def _open_upstream(self, url: str, envelope: dict, *, streaming: bool, label: str):
        body = json.dumps(envelope).encode("utf-8")
        token = self._token_manager.get_access_token()
        req = build_antigravity_request(url, body, token, streaming=streaming)
        try:
            return urlopen(req, timeout=self._timeout)
        except HTTPError as exc:
            detail = read_http_error_detail(exc)
            raise UpstreamError(
                f"{label} HTTP {exc.code}: {detail}",
                status=exc.code,
                body=detail,
            ) from exc
        except URLError as exc:
            raise UpstreamError(f"{label} network error: {exc}", status=502) from exc

    def call_generate_content(self, envelope: dict) -> dict:
        """Perform a non-streaming generateContent call and unwrap the response."""
        with self._open_upstream(
            GENERATE_CONTENT_URL,
            envelope,
            streaming=False,
            label="Upstream",
        ) as resp:
            raw = resp.read().decode("utf-8")

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise UpstreamError(
                f"Upstream returned non-JSON: {raw[:500]}",
                status=502,
            ) from exc

        if isinstance(payload, dict) and "response" in payload:
            return payload["response"]
        return payload

    def stream_generate_content(self, envelope: dict) -> Iterator[dict]:
        """Yield parsed JSON event objects from the upstream SSE/JSON stream."""
        resp = self._open_upstream(
            STREAM_GENERATE_CONTENT_URL,
            envelope,
            streaming=True,
            label="Upstream stream",
        )
        try:
            buffer = ""
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                text_chunk = (
                    chunk.decode("utf-8", errors="replace")
                    if isinstance(chunk, bytes)
                    else chunk
                )
                buffer += text_chunk

                while True:
                    parsed = try_parse_one_event(buffer)
                    if parsed is None:
                        break
                    event_obj, consumed = parsed
                    buffer = buffer[consumed:]
                    if event_obj is not None:
                        yield event_obj

            if buffer.strip():
                for obj in parse_remaining(buffer):
                    if obj is not None:
                        yield obj
        finally:
            resp.close()
