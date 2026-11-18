"""HTTP server and OpenAI/Gemini request handler for Antigravity Proxy."""

from __future__ import annotations

import hmac
import json
import logging
import re
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .auth import ProjectResolver, TokenManager
from .config import DEFAULT_MODEL, MAX_REQUEST_BODY_BYTES, MODEL_MAP, ProxyConfig
from .translator import (
    build_gemini_request,
    build_passthrough_envelope,
    extract_delta_from_parts,
    extract_usage,
    gemini_response_to_openai,
    map_finish_reason,
    openai_error,
)
from .upstream import UpstreamClient, UpstreamError

logger = logging.getLogger("antigravity_proxy.server")

_GEMINI_PASSTHROUGH_RE = re.compile(r"^/v1beta/models/(.+):generateContent$")
_PUBLIC_HEALTH_PATHS = frozenset({"/health", "/"})


class ProxyHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer carrying shared config, auth, and upstream dependencies."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        config: ProxyConfig,
        token_manager: TokenManager,
        project_resolver: ProjectResolver,
        upstream_client: UpstreamClient,
    ) -> None:
        self.config = config
        self.token_manager = token_manager
        self.project_resolver = project_resolver
        self.upstream_client = upstream_client
        super().__init__(server_address, ProxyHandler)


class ProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler exposing OpenAI-compatible and Gemini passthrough routes."""

    protocol_version = "HTTP/1.1"
    server_version = "AntigravityProxy/2.0"

    def log_message(self, fmt: str, *args) -> None:
        # Suppressed; structured request logging is handled by _log_request().
        return

    @property
    def _proxy(self) -> ProxyHTTPServer:
        return self.server  # type: ignore[return-value]

    def _resolve_cors_origin(self) -> str | None:
        configured = (self._proxy.config.cors_origins or "").strip()
        if not configured or configured.lower() == "none":
            return None
        if configured == "*":
            return "*"
        origin = self.headers.get("Origin", "").strip()
        allowed = [item.strip() for item in configured.split(",") if item.strip()]
        if origin and origin in allowed:
            return origin
        return allowed[0] if allowed else None

    def _send_cors_headers(self) -> None:
        origin = self._resolve_cors_origin()
        if not origin:
            return
        self.send_header("Access-Control-Allow-Origin", origin)
        if origin != "*":
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, x-api-key",
        )
        self.send_header("Access-Control-Max-Age", "86400")

    def _send_json(self, status: int, data: dict) -> None:
        self._last_status = status
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_error_json(
        self,
        code: str | int,
        message: str,
        err_type: str = "api_error",
        http_code: int | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        status = http_code or (code if isinstance(code, int) else 500)
        self._last_status = status
        body = json.dumps(openai_error(message, err_type, code=code)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _check_auth(self, path: str) -> bool:
        expected = self._proxy.config.api_key
        if not expected or path in _PUBLIC_HEALTH_PATHS:
            return True

        provided = ""
        auth_header = self.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            provided = auth_header[7:].strip()
        elif self.headers.get("x-api-key"):
            provided = self.headers.get("x-api-key", "").strip()

        if provided and hmac.compare_digest(provided, expected):
            return True

        self._send_error_json(
            "invalid_api_key",
            "Invalid or missing API key",
            err_type="authentication_error",
            http_code=401,
            extra_headers={"WWW-Authenticate": "Bearer"},
        )
        return False

    def _read_body(self) -> dict | None:
        raw_len = self.headers.get("Content-Length")
        if not raw_len:
            return {}
        try:
            length = int(raw_len)
        except ValueError:
            self._send_error_json(
                "invalid_request",
                "Invalid Content-Length header",
                err_type="invalid_request_error",
                http_code=400,
            )
            return None

        if length < 0:
            self._send_error_json(
                "invalid_request",
                "Negative Content-Length header",
                err_type="invalid_request_error",
                http_code=400,
            )
            return None
        if length > MAX_REQUEST_BODY_BYTES:
            self._send_error_json(
                "payload_too_large",
                f"Request body exceeds {MAX_REQUEST_BODY_BYTES} bytes",
                err_type="invalid_request_error",
                http_code=413,
            )
            return None
        if length == 0:
            return {}

        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            self._send_error_json(
                "invalid_request",
                "Invalid JSON body",
                err_type="invalid_request_error",
                http_code=400,
            )
            return None

        if not isinstance(parsed, dict):
            self._send_error_json(
                "invalid_request",
                "Request body must be a JSON object",
                err_type="invalid_request_error",
                http_code=400,
            )
            return None
        return parsed

    def _log_request(self, method: str, path: str, duration: float) -> None:
        status = getattr(self, "_last_status", 200)
        logger.info("%s %s %s %.3fs", method, path, status, duration)

    def _handle_exception(self, method: str, path: str, exc: Exception) -> None:
        logger.error("%s %s error: %s\n%s", method, path, exc, traceback.format_exc())
        try:
            self._send_error_json("internal_error", str(exc), http_code=500)
        except Exception:
            self._last_status = 500

    # ── HTTP verbs ────────────────────────────────────────────────────────────

    def do_OPTIONS(self) -> None:
        t0 = time.time()
        self._last_status = 204
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self._send_cors_headers()
        self.end_headers()
        self._log_request("OPTIONS", urlparse(self.path).path, time.time() - t0)

    def do_GET(self) -> None:
        t0 = time.time()
        self._last_status = 200
        path = urlparse(self.path).path
        try:
            if not self._check_auth(path):
                return
            if path in ("/v1/models", "/models"):
                self._handle_models()
            elif path in _PUBLIC_HEALTH_PATHS:
                self._send_json(200, {"status": "ok", "service": "antigravity-proxy"})
            else:
                self._send_error_json(
                    "not_found",
                    "Not found",
                    err_type="invalid_request_error",
                    http_code=404,
                )
        except Exception as exc:
            self._handle_exception("GET", path, exc)
        finally:
            self._log_request("GET", path, time.time() - t0)

    def do_POST(self) -> None:
        t0 = time.time()
        self._last_status = 200
        path = urlparse(self.path).path
        try:
            if not self._check_auth(path):
                return
            if path in ("/v1/chat/completions", "/chat/completions"):
                self._handle_chat()
            elif _GEMINI_PASSTHROUGH_RE.match(path):
                self._handle_gemini_passthrough(path)
            else:
                self._send_error_json(
                    "not_found",
                    "Not found",
                    err_type="invalid_request_error",
                    http_code=404,
                )
        except Exception as exc:
            self._handle_exception("POST", path, exc)
        finally:
            self._log_request("POST", path, time.time() - t0)

    # ── Route handlers ────────────────────────────────────────────────────────

    def _handle_models(self) -> None:
        now = int(time.time())
        models = [
            {"id": name, "object": "model", "created": now, "owned_by": "google"}
            for name in MODEL_MAP
        ]
        self._send_json(200, {"object": "list", "data": models})

    def _handle_gemini_passthrough(self, path: str) -> None:
        body = self._read_body()
        if body is None:
            return

        match = _GEMINI_PASSTHROUGH_RE.match(path)
        model = match.group(1) if match else DEFAULT_MODEL

        try:
            project_id = self._proxy.project_resolver.get_project_id()
            envelope = build_passthrough_envelope(model, body, project_id)
        except UpstreamError as exc:
            self._send_error_json("upstream_error", str(exc), http_code=exc.status)
            return
        except RuntimeError as exc:
            self._send_error_json(
                "auth_error",
                str(exc),
                err_type="authentication_error",
                http_code=401,
            )
            return
        except Exception as exc:
            self._send_error_json(
                "request_error",
                f"Failed to build request: {exc}",
                err_type="invalid_request_error",
                http_code=400,
            )
            return

        try:
            gemini_resp = self._proxy.upstream_client.call_generate_content(envelope)
        except UpstreamError as exc:
            self._send_error_json("upstream_error", str(exc), http_code=exc.status)
            return
        except Exception as exc:
            self._send_error_json(
                "upstream_error",
                f"Upstream call failed: {exc}",
                http_code=502,
            )
            return

        self._send_json(200, gemini_resp)

    def _handle_chat(self) -> None:
        body = self._read_body()
        if body is None:
            return

        openai_model = body.get("model") or DEFAULT_MODEL
        stream = bool(body.get("stream", False))
        backend_model, thinking_level = MODEL_MAP.get(
            openai_model, (openai_model, None)
        )

        try:
            project_id = self._proxy.project_resolver.get_project_id()
            envelope = build_gemini_request(
                body,
                backend_model,
                project_id,
                thinking_level=thinking_level,
            )
        except UpstreamError as exc:
            self._send_error_json("upstream_error", str(exc), http_code=exc.status)
            return
        except RuntimeError as exc:
            self._send_error_json(
                "auth_error",
                str(exc),
                err_type="authentication_error",
                http_code=401,
            )
            return
        except Exception as exc:
            self._send_error_json(
                "request_error",
                f"Failed to build request: {exc}",
                err_type="invalid_request_error",
                http_code=400,
            )
            return

        if not envelope.get("request", {}).get("contents"):
            self._send_error_json(
                "invalid_request",
                "No messages provided",
                err_type="invalid_request_error",
                http_code=400,
            )
            return

        if stream:
            self._handle_stream(envelope, openai_model)
        else:
            self._handle_nonstream(envelope, openai_model)

    def _handle_nonstream(self, envelope: dict, openai_model: str) -> None:
        try:
            gemini_resp = self._proxy.upstream_client.call_generate_content(envelope)
        except UpstreamError as exc:
            self._send_error_json("upstream_error", str(exc), http_code=exc.status)
            return
        except Exception as exc:
            self._send_error_json(
                "upstream_error",
                f"Upstream call failed: {exc}",
                http_code=502,
            )
            return

        try:
            openai_resp = gemini_response_to_openai(gemini_resp, openai_model)
        except Exception as exc:
            logger.error("Response transform error: %s\n%s", exc, traceback.format_exc())
            try:
                text = (
                    gemini_resp.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )
            except Exception:
                text = ""
            openai_resp = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": openai_model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text or "[error: could not parse response]",
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        self._send_json(200, openai_resp)

    # ── SSE streaming helpers ─────────────────────────────────────────────────

    def _write_sse_frame(self, payload: bytes) -> bool:
        try:
            self.wfile.write(payload)
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def _send_sse_json(self, obj: dict) -> bool:
        return self._write_sse_frame(f"data: {json.dumps(obj)}\n\n".encode("utf-8"))

    def _send_sse_done(self) -> None:
        self._write_sse_frame(b"data: [DONE]\n\n")

    @staticmethod
    def _make_stream_chunk(
        completion_id: str,
        created: int,
        openai_model: str,
        delta: dict,
        finish_reason: str | None = None,
        usage: dict | None = None,
    ) -> dict:
        chunk: dict = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": openai_model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }
        if usage is not None:
            chunk["usage"] = usage
        return chunk

    def _fallback_to_nonstream_chunk(
        self,
        envelope: dict,
        completion_id: str,
        created: int,
        openai_model: str,
    ) -> bool:
        """Fall back to non-streaming call when streamGenerateContent rejects HTTP 400."""
        try:
            resp = self._proxy.upstream_client.call_generate_content(envelope)
            openai_resp = gemini_response_to_openai(resp, openai_model)
        except Exception:
            return False

        choice = openai_resp["choices"][0]
        msg = choice["message"]
        delta: dict = {}
        if msg.get("content"):
            delta["content"] = msg["content"]
        if msg.get("tool_calls"):
            delta["tool_calls"] = msg["tool_calls"]

        chunk = self._make_stream_chunk(
            completion_id,
            created,
            openai_model,
            delta,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=openai_resp.get("usage"),
        )
        self._send_sse_json(chunk)
        self._send_sse_done()
        return True

    def _handle_stream(self, envelope: dict, openai_model: str) -> None:
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        self._last_status = 200

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self._send_cors_headers()
            self.end_headers()
            self.close_connection = True
        except Exception:
            return

        first_chunk = self._make_stream_chunk(
            completion_id,
            created,
            openai_model,
            {"role": "assistant", "content": ""},
        )
        if not self._send_sse_json(first_chunk):
            return

        try:
            gen = self._proxy.upstream_client.stream_generate_content(envelope)
            self._consume_stream_events(gen, completion_id, created, openai_model)
        except UpstreamError as exc:
            if exc.status == 400 and self._fallback_to_nonstream_chunk(
                envelope, completion_id, created, openai_model
            ):
                return
            self._send_sse_json(
                self._make_stream_chunk(
                    completion_id,
                    created,
                    openai_model,
                    {"content": f"[Error: {exc}]"},
                    finish_reason="stop",
                )
            )
            self._send_sse_done()
        except Exception as exc:
            logger.error("Streaming error: %s", exc)
            if "400" in str(exc) and self._fallback_to_nonstream_chunk(
                envelope, completion_id, created, openai_model
            ):
                return
            self._send_sse_json(
                self._make_stream_chunk(
                    completion_id,
                    created,
                    openai_model,
                    {"content": f"\n[stream error: {exc}]"},
                    finish_reason="stop",
                )
            )
            self._send_sse_done()

    def _consume_stream_events(
        self,
        gen,
        completion_id: str,
        created: int,
        openai_model: str,
    ) -> None:
        finish_reason = "stop"
        seen_tool_names: set[str] = set()
        total_usage: dict | None = None

        for event in gen:
            if not isinstance(event, dict):
                continue
            if event.get("__done__"):
                break

            resp = event.get("response", event)
            if resp.get("usageMetadata"):
                total_usage = resp["usageMetadata"]

            candidates = resp.get("candidates") or []
            if not candidates:
                continue

            cand = candidates[0]
            parts = (cand.get("content") or {}).get("parts") or []
            text_delta, tool_call_deltas, seen_tool_names = extract_delta_from_parts(
                parts, seen_tool_names
            )

            fr = cand.get("finishReason")
            if fr:
                finish_reason = map_finish_reason(cand, bool(tool_call_deltas))

            delta: dict = {}
            if text_delta:
                delta["content"] = text_delta
            if tool_call_deltas:
                delta["tool_calls"] = tool_call_deltas

            if not delta and not fr:
                continue

            chunk = self._make_stream_chunk(
                completion_id, created, openai_model, delta
            )
            if not self._send_sse_json(chunk):
                return

        usage_payload = (
            extract_usage({}, usage_override=total_usage) if total_usage else None
        )
        final_chunk = self._make_stream_chunk(
            completion_id,
            created,
            openai_model,
            {},
            finish_reason=finish_reason,
            usage=usage_payload,
        )
        self._send_sse_json(final_chunk)
        self._send_sse_done()
