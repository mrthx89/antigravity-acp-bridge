"""Unit and server integration tests for the ``agy-cli`` ACP backend."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from antigravity_proxy.agy_cli import (
    AgyAcpWorker,
    AgyCliClient,
    format_messages_for_agy,
    parse_agy_usage,
)
from antigravity_proxy.config import ProxyConfig
from antigravity_proxy.server import ProxyHTTPServer


class _FakeAcpProcess:
    """In-memory fake ``agy acp --stdio`` JSON-RPC subprocess."""

    def __init__(self) -> None:
        self._responses: list[str] = []
        self.sent: list[dict] = []
        self.terminated = False
        self.stdin = _FakeStdin(self)
        self.stdout = _FakeStdout(self)

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True

    def wait(self, timeout=None) -> int:
        self.terminated = True
        return 0

    def handle_request(self, obj: dict) -> None:
        self.sent.append(obj)
        method = obj.get("method")
        req_id = obj.get("id")
        if method == "initialize":
            self._responses.append(json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"protocolVersion": 1},
            }))
        elif method == "session/new":
            self._responses.append(json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "sessionId": "sess-1",
                    "models": {"currentModelId": "gemini-4-argon-medium"},
                },
            }))
        elif method == "session/set_model":
            self._responses.append(json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {},
            }))
        elif method == "session/prompt":
            sid = obj["params"]["sessionId"]
            # Emit thought chunk (should be ignored) + text chunks + permission request + result.
            self._responses.extend([
                json.dumps({
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": sid,
                        "update": {
                            "sessionUpdate": "agent_thought_chunk",
                            "content": {"type": "text", "text": "hidden thought"},
                        },
                    },
                }),
                json.dumps({
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": sid,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "Hello "},
                        },
                    },
                }),
                json.dumps({
                    "jsonrpc": "2.0",
                    "id": 999,
                    "method": "session/request_permission",
                    "params": {},
                }),
                json.dumps({
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": sid,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "world"},
                        },
                    },
                }),
                json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "stopReason": "end_turn",
                        "usage": {"inputTokens": 40, "outputTokens": 2, "totalTokens": 42},
                    },
                }),
            ])
        elif method == "session/close":
            self._responses.append(json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {},
            }))


class _FakeStdin:
    def __init__(self, proc: _FakeAcpProcess) -> None:
        self._proc = proc
        self.closed = False

    def write(self, data: str) -> int:
        for line in data.splitlines():
            if line.strip():
                self._proc.handle_request(json.loads(line))
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FakeStdout:
    def __init__(self, proc: _FakeAcpProcess) -> None:
        self._proc = proc
        self.closed = False

    def readline(self) -> str:
        if not self._proc._responses:
            return ""
        return self._proc._responses.pop(0) + "\n"

    def close(self) -> None:
        self.closed = True


class AgyCliBackendTests(unittest.TestCase):
    def test_format_messages_single_and_multi_turn(self) -> None:
        self.assertEqual(
            format_messages_for_agy([{"role": "user", "content": "Hello"}]),
            "Hello",
        )
        formatted = format_messages_for_agy([
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "2+2?"},
        ])
        self.assertIn("System instructions:\nBe terse.", formatted)
        self.assertIn("[Assistant]\nHello!", formatted)
        self.assertIn("[User]\n2+2?", formatted)

    def test_parse_agy_usage_both_casings(self) -> None:
        self.assertEqual(
            parse_agy_usage({"input_tokens": 100, "output_tokens": 5, "total_tokens": 105}),
            {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
        )
        self.assertEqual(
            parse_agy_usage({"inputTokens": 40, "outputTokens": 2, "totalTokens": 42}),
            {"prompt_tokens": 40, "completion_tokens": 2, "total_tokens": 42},
        )

    def test_resolve_bin_falls_back_to_standard_location(self) -> None:
        cfg = ProxyConfig(backend="agy-cli", agy_bin="agy")
        client = AgyCliClient(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            if os.name == "nt":
                bin_dir = os.path.join(tmpdir, "agy", "bin")
                os.makedirs(bin_dir, exist_ok=True)
                fake_bin = os.path.join(bin_dir, "agy.exe")
                with open(fake_bin, "w", encoding="utf-8") as fh:
                    fh.write("stub")
                env_patch = {"LOCALAPPDATA": tmpdir}
            else:
                bin_dir = os.path.join(tmpdir, ".local", "bin")
                os.makedirs(bin_dir, exist_ok=True)
                fake_bin = os.path.join(bin_dir, "agy")
                with open(fake_bin, "w", encoding="utf-8") as fh:
                    fh.write("stub")
                env_patch = {"HOME": tmpdir}

            with patch("antigravity_proxy.agy_cli.shutil.which", return_value=None), patch.dict(
                os.environ, env_patch, clear=False
            ):
                self.assertEqual(client.resolve_bin(), fake_bin)

    def test_acp_worker_reuses_process_rejects_permissions_and_closes_session(self) -> None:
        fake_proc = _FakeAcpProcess()
        cfg = ProxyConfig(backend="agy-cli", max_concurrency=1)
        client = AgyCliClient(cfg)

        with patch("antigravity_proxy.agy_cli.shutil.which", return_value="agy"), patch(
            "antigravity_proxy.agy_cli.subprocess.Popen", return_value=fake_proc
        ) as m_popen:
            client.warm_up()
            events = list(
                client.stream_chat(
                    {"messages": [{"role": "user", "content": "hi"}]},
                    "gemini-3.8-flash-low",
                )
            )
            resp = client.call_chat(
                {"model": "gemini-3.8-flash-low", "messages": [{"role": "user", "content": "hi"}]},
                "gemini-3.8-flash-low",
            )
            client.close()

        # Process spawned once across warm_up + stream_chat + call_chat!
        self.assertEqual(m_popen.call_count, 1)
        self.assertEqual(events[0], {"text": "Hello "})
        self.assertEqual(events[1], {"text": "world"})
        self.assertEqual(events[2]["usage"]["total_tokens"], 42)
        self.assertEqual(resp["choices"][0]["message"]["content"], "Hello world")

        methods = [m.get("method") for m in fake_proc.sent if "method" in m]
        self.assertEqual(methods.count("initialize"), 1)
        self.assertEqual(methods.count("session/new"), 2)
        self.assertEqual(methods.count("session/set_model"), 2)
        self.assertEqual(methods.count("session/close"), 2)

        # Server-initiated permission request id=999 must be rejected on both calls.
        perm_replies = [m for m in fake_proc.sent if m.get("id") == 999 and "error" in m]
        self.assertEqual(len(perm_replies), 2)
        self.assertTrue(fake_proc.terminated)

    def test_server_routes_in_agy_cli_mode(self) -> None:
        agy_client = MagicMock()
        agy_client.list_models.return_value = ["gemini-3.8-flash-low", "gemini-4-hydrogen-high"]
        agy_client.call_chat.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "gemini-3.8-flash-low",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }
        agy_client.stream_chat.return_value = iter([
            {"text": "Hi"},
            {"text": "!"},
            {"finish_reason": "stop", "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
        ])

        cfg = ProxyConfig(backend="agy-cli", api_key="cli-secret")
        server = ProxyHTTPServer(("127.0.0.1", 0), cfg, agy_client=agy_client)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]

        def req(method, path, body=None, headers=None):
            data = json.dumps(body).encode("utf-8") if body is not None else None
            r = Request(f"http://127.0.0.1:{port}{path}", data=data, method=method)
            if data is not None:
                r.add_header("Content-Type", "application/json")
            for k, v in (headers or {}).items():
                r.add_header(k, v)
            try:
                with urlopen(r, timeout=5) as resp:
                    return resp.status, resp.read().decode("utf-8")
            except HTTPError as exc:
                return exc.code, exc.read().decode("utf-8")

        try:
            auth = {"Authorization": "Bearer cli-secret"}
            st, raw = req("GET", "/v1/models", headers=auth)
            self.assertEqual(st, 200)
            self.assertEqual([m["id"] for m in json.loads(raw)["data"]], ["gemini-3.8-flash-low", "gemini-4-hydrogen-high"])

            st, raw = req(
                "POST",
                "/v1/chat/completions",
                body={"model": "gemini-3.8-flash-low", "messages": [{"role": "user", "content": "hi"}]},
                headers=auth,
            )
            self.assertEqual(st, 200)
            self.assertEqual(json.loads(raw)["choices"][0]["message"]["content"], "Hi!")

            st, raw = req(
                "POST",
                "/v1/chat/completions",
                body={"model": "gemini-3.8-flash-low", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
                headers=auth,
            )
            self.assertEqual(st, 200)
            self.assertIn("Hi", raw)
            self.assertIn("data: [DONE]", raw)

            st, _ = req(
                "POST",
                "/v1beta/models/gemini-3.8-flash-low:generateContent",
                body={"contents": []},
                headers=auth,
            )
            self.assertEqual(st, 501)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
