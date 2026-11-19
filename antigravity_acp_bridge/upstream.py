"""Upstream HTTP client for Cloud Code Assist generateContent endpoints."""

from __future__ import annotations

import json
import logging
import time
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from ._http import build_antigravity_request, read_http_error_detail
from .auth import TokenManager
from .config import (
    GENERATE_CONTENT_URL,
    STREAM_GENERATE_CONTENT_URL,
    UPSTREAM_RETRY_ATTEMPTS,
    UPSTREAM_RETRY_BASE_DELAY,
    UPSTREAM_RETRY_MAX_DELAY,
    UPSTREAM_RETRY_STATUSES,
    UPSTREAM_TIMEOUT,
)
from .sse import parse_remaining, try_parse_one_event

logger = logging.getLogger("antigravity_proxy.upstream")


def _retry_after_seconds(err: HTTPError) -> float:
    """Return the upstream Retry-After delay in seconds, or 0 when absent."""
    try:
        raw = err.headers.get("Retry-After") if err.headers else None
        if raw:
            return max(0.0, float(str(raw).strip()))
    except (TypeError, ValueError):
        pass
    return 0.0


def _log_retry(label: str, status: object, attempt: int, attempts: int, delay: float) -> None:
    logger.warning(
        "%s HTTP %s (attempt %d/%d) — retrying in %.1fs",
        label,
        status,
        attempt + 1,
        attempts,
        delay,
    )


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
        """Open the upstream request, retrying transient capacity/quota errors.

        daily-cloudcode-pa flaps between 200, 429 RESOURCE_EXHAUSTED and
        503 MODEL_CAPACITY_EXHAUSTED for the same request seconds apart. Those
        are transient server-side conditions, not a bad request: relaying them
        straight to the client turns a hiccup into a hard failure. Retry them
        here, before any bytes reach the client, so the caller only ever sees a
        real answer or a genuinely exhausted model.

        Non-transient statuses (400, 401, 403, 404) are raised immediately —
        retrying those would only delay the caller for nothing.
        """
        body = json.dumps(envelope).encode("utf-8")
        attempts = UPSTREAM_RETRY_ATTEMPTS

        for attempt in range(attempts):
            token = self._token_manager.get_access_token()
            req = build_antigravity_request(url, body, token, streaming=streaming)
            try:
                return urlopen(req, timeout=self._timeout)
            except HTTPError as exc:
                detail = read_http_error_detail(exc)
                error = UpstreamError(
                    f"{label} HTTP {exc.code}: {detail}",
                    status=exc.code,
                    body=detail,
                )
                if exc.code not in UPSTREAM_RETRY_STATUSES or attempt == attempts - 1:
                    raise error from exc

                delay = min(
                    _retry_after_seconds(exc)
                    or (UPSTREAM_RETRY_BASE_DELAY * (2**attempt)),
                    UPSTREAM_RETRY_MAX_DELAY,
                )
                _log_retry(label, exc.code, attempt, attempts, delay)
                time.sleep(delay)
            except URLError as exc:
                error = UpstreamError(f"{label} network error: {exc}", status=502)
                if attempt == attempts - 1:
                    raise error from exc

                delay = min(
                    UPSTREAM_RETRY_BASE_DELAY * (2**attempt), UPSTREAM_RETRY_MAX_DELAY
                )
                _log_retry(label, "network", attempt, attempts, delay)
                time.sleep(delay)

        # Unreachable: every branch above either returns or raises.
        raise UpstreamError(f"{label} failed without a response", status=502)

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
