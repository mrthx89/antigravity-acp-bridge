"""Unit and server integration tests for the ``agy-cli`` backend."""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from antigravity_proxy.agy_cli import (
    AgyCliClient,
    format_messages_for_agy,
    parse_agy_usage,
)
from antigravity_proxy.config import ProxyConfig
from antigravity_proxy.server import ProxyHTTPServer


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

    def test_parse_agy_usage(self) -> None:
        self.assertEqual(
            parse_agy_usage({"input_tokens": 100, "output_tokens": 5, "total_tokens": 105}),
            {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
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

    def test_call_chat_and_stream_chat(self) -> None:
        cfg = ProxyConfig(backend="agy-cli", agy_bin="agy")
        client = AgyCliClient(cfg)

        fake_proc = SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "status": "SUCCESS",
                "response": "OK\n",
                "usage": {"input_tokens": 50, "output_tokens": 2, "total_tokens": 52},
            }),
            stderr="",
        )
        with patch("antigravity_proxy.agy_cli.shutil.which", return_value="C:\\agy.exe"), patch(
            "antigravity_proxy.agy_cli.subprocess.run", return_value=fake_proc
        ):
            resp = client.call_chat(
                {"model": "gemini-3.8-flash-low", "messages": [{"role": "user", "content": "hi"}]},
                "gemini-3.8-flash-low",
            )
        self.assertEqual(resp["choices"][0]["message"]["content"], "OK\n")
        self.assertEqual(resp["usage"]["total_tokens"], 52)

        ndjson = "\n".join([
            json.dumps({"event": "init", "init": {"model": "gemini-3.8-flash-low"}}),
            json.dumps({
                "event": "step_update",
                "step_update": {"step_type": "agent_response", "text_delta": "Hello "},
            }),
            json.dumps({
                "event": "step_update",
                "step_update": {"step_type": "agent_response", "text_delta": "world"},
            }),
            json.dumps({
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "response": "Hello world",
                    "usage": {"input_tokens": 20, "output_tokens": 2, "total_tokens": 22},
                },
            }),
        ])
        fake_popen = MagicMock()
        fake_popen.stdout = io.StringIO(ndjson + "\n")
        fake_popen.stderr = io.StringIO("")
        fake_popen.wait.return_value = 0
        fake_popen.poll.return_value = 0

        with patch("antigravity_proxy.agy_cli.shutil.which", return_value="C:\\agy.exe"), patch(
            "antigravity_proxy.agy_cli.subprocess.Popen", return_value=fake_popen
        ):
            events = list(
                client.stream_chat(
                    {"messages": [{"role": "user", "content": "hi"}]},
                    "gemini-3.8-flash-low",
                )
            )
        self.assertEqual(events[0], {"text": "Hello "})
        self.assertEqual(events[1], {"text": "world"})
        self.assertEqual(events[2]["finish_reason"], "stop")
        self.assertEqual(events[2]["usage"]["total_tokens"], 22)

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
