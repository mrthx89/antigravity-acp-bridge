"""Integration tests for ProxyHTTPServer, auth, CORS, and routes."""

from __future__ import annotations

import json
import threading
import unittest
from unittest.mock import MagicMock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from antigravity_acp_bridge.cli import main as cli_main
from antigravity_acp_bridge.config import ProxyConfig
from antigravity_acp_bridge.server import ProxyHTTPServer
from antigravity_acp_bridge.upstream import UpstreamError


class ServerIntegrationTests(unittest.TestCase):
    def _start_server(self, config: ProxyConfig, upstream: MagicMock | None = None):
        token_mgr = MagicMock()
        resolver = MagicMock()
        resolver.get_project_id.return_value = "test-project"
        if upstream is None:
            upstream = MagicMock()
        server = ProxyHTTPServer(("127.0.0.1", 0), config, token_mgr, resolver, upstream)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def _request(self, port: int, method: str, path: str, body=None, headers=None):
        url = f"http://127.0.0.1:{port}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = Request(url, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            with urlopen(req, timeout=5) as resp:
                return resp.status, dict(resp.headers), resp.read().decode("utf-8")
        except HTTPError as exc:
            return exc.code, dict(exc.headers), exc.read().decode("utf-8")

    def test_auth_cors_models_chat_and_stream_fallback(self) -> None:
        upstream = MagicMock()
        upstream.call_generate_content.return_value = {
            "candidates": [{"content": {"parts": [{"text": "hello back"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 6},
        }
        upstream.stream_generate_content.side_effect = UpstreamError("bad stream", status=400)

        cfg = ProxyConfig(backend="oauth", api_key="secret-123", cors_origins="https://ui.example.com")
        server, thread = self._start_server(cfg, upstream)
        port = server.server_address[1]
        try:
            # 1. Health stays public
            status, headers, raw = self._request(port, "GET", "/health")
            self.assertEqual(status, 200)
            health_data = json.loads(raw)
            self.assertEqual(health_data["status"], "ok")
            self.assertEqual(health_data["service"], "antigravity-acp-bridge")
            self.assertEqual(health_data["trademark"], "YHDev Bekerja dengan Hati @ 2026")

            # 2. CORS preflight
            status, headers, _ = self._request(
                port,
                "OPTIONS",
                "/v1/chat/completions",
                headers={"Origin": "https://ui.example.com"},
            )
            self.assertEqual(status, 204)
            self.assertEqual(headers.get("Access-Control-Allow-Origin"), "https://ui.example.com")

            # 3. Missing API key rejected
            status, _, raw = self._request(port, "GET", "/v1/models")
            self.assertEqual(status, 401)
            self.assertEqual(json.loads(raw)["error"]["code"], "invalid_api_key")

            # 4. Bearer & x-api-key accepted
            status, _, raw = self._request(
                port,
                "GET",
                "/v1/models",
                headers={"Authorization": "Bearer secret-123"},
            )
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(raw)["object"], "list")

            # 5. Non-stream chat completion
            status, _, raw = self._request(
                port,
                "POST",
                "/v1/chat/completions",
                body={"model": "gemini-3.8-flash-high", "messages": [{"role": "user", "content": "hi"}]},
                headers={"x-api-key": "secret-123"},
            )
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(raw)["choices"][0]["message"]["content"], "hello back")

            # 6. Stream fallback on 400 yields clean SSE chunks without __done__ JSON frame
            status, _, raw = self._request(
                port,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "gemini-3.8-flash-high",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Authorization": "Bearer secret-123"},
            )
            self.assertEqual(status, 200)
            self.assertIn("hello back", raw)
            self.assertIn("data: [DONE]", raw)
            self.assertNotIn("__done__", raw)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_chunked_transfer_encoding_post_body(self) -> None:
        import http.client

        upstream = MagicMock()
        upstream.call_generate_content.return_value = {
            "candidates": [{"content": {"parts": [{"text": "chunked ok"}]}, "finishReason": "STOP"}],
        }
        cfg = ProxyConfig(backend="oauth")
        server, thread = self._start_server(cfg, upstream)
        port = server.server_address[1]
        try:
            payload = json.dumps({
                "model": "gemini-3.8-flash-high",
                "messages": [{"role": "user", "content": "hi"}],
            }).encode("utf-8")
            chunked = f"{len(payload):x}\r\n".encode("ascii") + payload + b"\r\n0\r\n\r\n"
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=chunked,
                headers={"Content-Type": "application/json", "Transfer-Encoding": "chunked"},
            )
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8")
            conn.close()
            self.assertEqual(resp.status, 200)
            self.assertEqual(json.loads(raw)["choices"][0]["message"]["content"], "chunked ok")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_cli_refuses_non_localhost_without_api_key(self) -> None:
        code = cli_main(["--host", "0.0.0.0"])
        self.assertEqual(code, 2)
