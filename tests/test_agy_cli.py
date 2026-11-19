"""Unit and server integration tests for the ``agy-cli`` ACP backend."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from antigravity_acp_bridge.agy_cli import (
    AgyAcpWorker,
    AgyCliClient,
    AgyCliError,
    format_messages_for_agy,
    format_request_for_agy,
    parse_agy_usage,
    parse_openai_tool_calls,
)
from antigravity_acp_bridge.config import BridgeConfig
from antigravity_acp_bridge.server import BridgeHTTPServer


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
        cfg = BridgeConfig(backend="agy-cli", agy_bin="agy")
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

            with patch("antigravity_acp_bridge.agy_cli.shutil.which", return_value=None), patch.dict(
                os.environ, env_patch, clear=False
            ):
                self.assertEqual(client.resolve_bin(), fake_bin)

    def test_acp_worker_reuses_process_rejects_permissions_and_closes_session(self) -> None:
        fake_proc = _FakeAcpProcess()
        cfg = BridgeConfig(backend="agy-cli", max_concurrency=1)
        client = AgyCliClient(cfg)

        with patch("antigravity_acp_bridge.agy_cli.shutil.which", return_value="agy"), patch(
            "antigravity_acp_bridge.agy_cli.subprocess.Popen", return_value=fake_proc
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

        # Server-initiated permission request id=999 must be rejected via ACP outcome.
        perm_replies = [m for m in fake_proc.sent if m.get("id") == 999 and "result" in m]
        self.assertEqual(len(perm_replies), 2)
        self.assertEqual(perm_replies[0]["result"]["outcome"]["outcome"], "cancelled")
        self.assertTrue(fake_proc.terminated)

    def test_tool_call_bridging_format_and_parse(self) -> None:
        body = {
            "messages": [
                {"role": "user", "content": "cek server"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": '{"command":"uptime"}'}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "name": "terminal", "content": "up 3 days"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "terminal", "description": "Run cmd", "parameters": {"type": "object"}},
                }
            ],
        }
        prompt = format_request_for_agy(body)
        self.assertIn("CRITICAL RUNTIME RULES", prompt)
        self.assertIn("[Tool Result (terminal, id=call_1)]", prompt)
        self.assertIn("up 3 days", prompt)

        rem, tcs = parse_openai_tool_calls(
            '<openai_tool_calls>[{"name": "terminal", "arguments": {"command": "free -h"}}]</openai_tool_calls>'
        )
        self.assertIsNone(rem)
        self.assertEqual(len(tcs), 1)
        self.assertEqual(tcs[0]["function"]["name"], "terminal")
        self.assertEqual(json.loads(tcs[0]["function"]["arguments"]), {"command": "free -h"})

        # Fenced JSON inside <openai_tool_calls> must also parse cleanly.
        rem2, tcs2 = parse_openai_tool_calls(
            'Checking now.\n<openai_tool_calls>\n```json\n[{"name": "terminal", "arguments": {"command": "df -h"}}]\n```\n</openai_tool_calls>'
        )
        self.assertEqual(rem2, "Checking now.")
        self.assertEqual(len(tcs2), 1)
        self.assertEqual(json.loads(tcs2[0]["function"]["arguments"]), {"command": "df -h"})

    def test_stream_with_tool_parsing_handles_prose_prefix_and_malformed_json(self) -> None:
        client = AgyCliClient(BridgeConfig(backend="agy-cli", worker_idle_timeout=0))
        try:
            # Prose before <openai_tool_calls> split across chunk boundaries
            raw_chunks = iter([
                {"text": "Let me check. <openai_"},
                {"text": 'tool_calls>[{"name":"terminal","arguments":{"command":"uptime"}}]</openai_tool_calls>'},
                {"finish_reason": "stop", "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}},
            ])
            events = list(client._stream_with_tool_parsing(raw_chunks))
            texts = [e["text"] for e in events if "text" in e]
            tcs = [e["tool_calls"] for e in events if "tool_calls" in e]
            self.assertEqual("".join(texts), "Let me check. ")
            self.assertEqual(len(tcs), 1)
            self.assertEqual(tcs[0][0]["function"]["name"], "terminal")
            self.assertEqual(events[-1]["finish_reason"], "tool_calls")

            # Malformed JSON inside tag must be preserved as text, not dropped
            bad_chunks = iter([
                {"text": "<openai_tool_calls>{not valid json}</openai_tool_calls>"},
                {"finish_reason": "stop", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            ])
            bad_events = list(client._stream_with_tool_parsing(bad_chunks))
            self.assertEqual(bad_events[0]["text"], "<openai_tool_calls>{not valid json}</openai_tool_calls>")
            self.assertEqual(bad_events[-1]["finish_reason"], "stop")
        finally:
            client.close()

    def test_acp_worker_falls_back_on_unknown_flag_and_surfaces_stderr(self) -> None:
        import io

        class _FlagRejectProc:
            def __init__(self) -> None:
                self.stdin = io.StringIO()
                self.stdout = io.StringIO("")
                self.stderr = io.StringIO("flag provided but not defined: -include-default-customizations\n")

            def poll(self):
                return 2

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 2

        good_proc = _FakeAcpProcess()
        spawned_cmds: list[list[str]] = []

        def _fake_popen(cmd, **kwargs):
            spawned_cmds.append(list(cmd))
            return _FlagRejectProc() if len(spawned_cmds) == 1 else good_proc

        worker = AgyAcpWorker("agy", BridgeConfig(backend="agy-cli"))
        try:
            with patch("antigravity_acp_bridge.agy_cli.subprocess.Popen", side_effect=_fake_popen):
                worker.ensure_started()
            self.assertEqual(len(spawned_cmds), 2)
            self.assertEqual(spawned_cmds[1], ["agy", "acp", "--stdio"])
        finally:
            worker.close()

    def test_acp_worker_timeout_raises_504(self) -> None:
        import queue as _q

        worker = AgyAcpWorker("agy", BridgeConfig(backend="agy-cli", upstream_timeout=1))
        worker._use_stdout_reader = True
        worker._stdout_queue = _q.Queue()
        worker._stderr_lines.append("waiting for keyring")
        with self.assertRaises(AgyCliError) as ctx:
            worker._read_line(time.monotonic() + 0.05)
        self.assertEqual(ctx.exception.status, 504)
        self.assertIn("waiting for keyring", str(ctx.exception))

    def test_server_stream_returns_http_error_on_early_acp_failure(self) -> None:
        agy_client = MagicMock()
        agy_client.stream_chat.side_effect = AgyCliError("agy not logged in", status=502)

        cfg = BridgeConfig(backend="agy-cli", api_key="k")
        server = BridgeHTTPServer(("127.0.0.1", 0), cfg, agy_client=agy_client)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            req = Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps({
                    "model": "gemini-3.8-flash-low",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                }).encode("utf-8"),
                headers={"Authorization": "Bearer k", "Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as ctx:
                urlopen(req, timeout=5)
            self.assertEqual(ctx.exception.code, 502)
            body = json.loads(ctx.exception.read().decode("utf-8"))
            self.assertIn("agy not logged in", body["error"]["message"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_acp_empty_response_triggers_recovery_turn(self) -> None:
        class _RecoveryProc(_FakeAcpProcess):
            def handle_request(self, obj: dict) -> None:
                if obj.get("method") == "session/prompt":
                    self.sent.append(obj)
                    sid = obj["params"]["sessionId"]
                    req_id = obj["id"]
                    prompt_txt = obj["params"]["prompt"][0]["text"]
                    if "Built-in tools are disabled here." in prompt_txt:
                        self._responses.extend([
                            json.dumps({
                                "jsonrpc": "2.0",
                                "method": "session/update",
                                "params": {
                                    "sessionId": sid,
                                    "update": {
                                        "sessionUpdate": "agent_message_chunk",
                                        "content": {"type": "text", "text": "Recovered answer"},
                                    },
                                },
                            }),
                            json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn", "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12}}}),
                        ])
                    else:
                        self._responses.extend([
                            json.dumps({"jsonrpc": "2.0", "id": 888, "method": "session/request_permission", "params": {"options": [{"optionId": "deny", "kind": "reject_once"}]}}),
                            json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn", "usage": {"inputTokens": 10, "outputTokens": 1, "totalTokens": 11}}}),
                        ])
                else:
                    super().handle_request(obj)

        fake_proc = _RecoveryProc()
        client = AgyCliClient(BridgeConfig(backend="agy-cli", max_concurrency=1))
        with patch("antigravity_acp_bridge.agy_cli.shutil.which", return_value="agy"), patch(
            "antigravity_acp_bridge.agy_cli.subprocess.Popen", return_value=fake_proc
        ):
            resp = client.call_chat(
                {"messages": [{"role": "user", "content": "cek server"}]},
                "gemini-3.8-flash-high",
            )
            client.close()
        self.assertEqual(resp["choices"][0]["message"]["content"], "Recovered answer")

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

        cfg = BridgeConfig(backend="agy-cli", api_key="cli-secret")
        server = BridgeHTTPServer(("127.0.0.1", 0), cfg, agy_client=agy_client)
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


class _LiveWorker:
    """Stand-in for AgyAcpWorker that reports a live process (no subprocess needed)."""

    def __init__(self) -> None:
        self.closed = False

    def is_alive(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


class PoolLimitSemanticsTest(unittest.TestCase):
    """Contract for --max-concurrency: >0 caps the warm pool, <=0 means unlimited."""

    @staticmethod
    def _client(limit: int) -> AgyCliClient:
        return AgyCliClient(BridgeConfig(backend="agy-cli", max_concurrency=limit))

    def test_positive_limit_sets_pool_cap(self) -> None:
        client = self._client(3)
        try:
            self.assertEqual(client._max_workers, 3)
        finally:
            client.close()

    def test_zero_and_negative_mean_unlimited(self) -> None:
        for limit in (0, -1, -42):
            client = self._client(limit)
            try:
                self.assertIsNone(client._max_workers, f"limit {limit} should mean no cap")
            finally:
                client.close()

    def test_unlimited_acquire_never_blocks_or_reuses(self) -> None:
        client = self._client(0)
        try:
            with patch("antigravity_acp_bridge.agy_cli.shutil.which", return_value="agy"), patch(
                "antigravity_acp_bridge.agy_cli.AgyAcpWorker", side_effect=lambda *a, **k: _LiveWorker()
            ):
                workers = [client._acquire_worker() for _ in range(5)]
                self.assertEqual(len({id(w) for w in workers}), 5)
                self.assertEqual(client._created, 5)
                for worker in workers:
                    client._release_worker(worker, broken=False)
                # Released live workers come back out of the LIFO pool for reuse.
                self.assertIs(client._acquire_worker(), workers[-1])
        finally:
            client.close()

    def test_capped_pool_blocks_when_exhausted(self) -> None:
        client = self._client(1)
        try:
            with patch("antigravity_acp_bridge.agy_cli.shutil.which", return_value="agy"), patch(
                "antigravity_acp_bridge.agy_cli.AgyAcpWorker", side_effect=lambda *a, **k: _LiveWorker()
            ):
                first = client._acquire_worker()
                acquired: list = []
                thread = threading.Thread(
                    target=lambda: acquired.append(client._acquire_worker()), daemon=True
                )
                thread.start()
                thread.join(timeout=0.3)
                self.assertTrue(
                    thread.is_alive(), "second acquire must block while the only worker is out"
                )
                client._release_worker(first, broken=False)
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())
                self.assertEqual(acquired, [first])
        finally:
            client.close()


class IdleReaperTests(unittest.TestCase):
    """Idle workers must be closed so an unlimited pool does not park burst memory."""

    @staticmethod
    def _client(timeout: float, limit: int = 0) -> AgyCliClient:
        return AgyCliClient(
            BridgeConfig(backend="agy-cli", max_concurrency=limit, worker_idle_timeout=timeout)
        )

    def test_reaper_disabled_when_timeout_is_zero(self) -> None:
        client = self._client(0)
        try:
            self.assertIsNone(client._reaper)
        finally:
            client.close()

    def test_expired_idle_workers_are_closed(self) -> None:
        client = self._client(0.2)
        try:
            workers: list[_LiveWorker] = []
            with patch("antigravity_acp_bridge.agy_cli.shutil.which", return_value="agy"), patch(
                "antigravity_acp_bridge.agy_cli.AgyAcpWorker",
                side_effect=lambda *a, **k: workers.append(_LiveWorker()) or workers[-1],
            ):
                held = [client._acquire_worker() for _ in range(2)]
                for worker in held:
                    client._release_worker(worker, broken=False)
                self.assertEqual(client._created, 2)

                deadline = time.time() + 6
                while client._created > 0 and time.time() < deadline:
                    time.sleep(0.05)

            self.assertEqual(client._created, 0)
            self.assertEqual(len(workers), 2)
            self.assertTrue(all(w.closed for w in workers))
        finally:
            client.close()

    def test_busy_workers_survive_the_reaper(self) -> None:
        client = self._client(0.2)
        try:
            with patch("antigravity_acp_bridge.agy_cli.shutil.which", return_value="agy"), patch(
                "antigravity_acp_bridge.agy_cli.AgyAcpWorker", side_effect=lambda *a, **k: _LiveWorker()
            ):
                busy = client._acquire_worker()
                time.sleep(1.4)  # longer than the 1 s reaper tick
                self.assertTrue(busy.is_alive())
                self.assertEqual(client._created, 1)
                client._release_worker(busy, broken=False)
        finally:
            client.close()

    def test_extract_images_from_messages(self) -> None:
        from antigravity_acp_bridge.agy_cli import extract_images_for_agy

        msgs = [
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,QUJDREVGRw=="},
                    },
                    {
                        "type": "image",
                        "data": "SEVMTE8=",
                        "mimeType": "image/png",
                    },
                ],
            },
        ]
        images = extract_images_for_agy(msgs)
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0], {"type": "image", "data": "QUJDREVGRw==", "mimeType": "image/jpeg"})
        self.assertEqual(images[1], {"type": "image", "data": "SEVMTE8=", "mimeType": "image/png"})


class HeavyHermesReliabilityTests(unittest.TestCase):
    def test_aborted_stream_closes_dirty_worker_instead_of_reusing(self) -> None:
        class _StreamingWorker(_LiveWorker):
            def stream_prompt(self, prompt, model, *, has_external_tools=False, image_blocks=None):
                yield {"text": "first chunk"}
                yield {"text": "second chunk"}
                yield {"finish_reason": "stop", "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}

        workers: list[_StreamingWorker] = []
        client = AgyCliClient(BridgeConfig(backend="agy-cli", max_concurrency=2, worker_idle_timeout=0))
        try:
            with patch("antigravity_acp_bridge.agy_cli.shutil.which", return_value="agy"), patch(
                "antigravity_acp_bridge.agy_cli.AgyAcpWorker",
                side_effect=lambda *a, **k: workers.append(_StreamingWorker()) or workers[-1],
            ):
                gen = client.stream_chat(
                    {"messages": [{"role": "user", "content": "hi"}]},
                    "gemini-3.8-flash-low",
                )
                first = next(gen)
                self.assertEqual(first, {"text": "first chunk"})
                # Simulate client disconnect mid-stream (e.g. Hermes Ctrl+C or timeout).
                gen.close()

            self.assertEqual(len(workers), 1)
            self.assertTrue(workers[0].closed)
            self.assertEqual(len(client._idle), 0)
            self.assertEqual(client._created, 0)
        finally:
            client.close()

    def test_parse_multiple_tool_call_blocks_claude_input_and_none_args(self) -> None:
        raw = (
            'Checking both.\n'
            '<openai_tool_calls>[{"name": "read_file", "input": {"path": "a.py"}}]</openai_tool_calls>\n'
            '<openai_tool_calls>[{"name": "noop", "arguments": null}]</openai_tool_calls>'
        )
        rem, tcs = parse_openai_tool_calls(raw)
        self.assertEqual(rem, "Checking both.")
        self.assertEqual(len(tcs), 2)
        self.assertEqual(tcs[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(tcs[0]["function"]["arguments"]), {"path": "a.py"})
        self.assertEqual(tcs[1]["function"]["name"], "noop")
        self.assertEqual(json.loads(tcs[1]["function"]["arguments"]), {})

    def test_response_format_json_object_and_schema(self) -> None:
        prompt_obj = format_request_for_agy({
            "messages": [{"role": "user", "content": "Return status"}],
            "response_format": {"type": "json_object"},
        })
        self.assertIn("valid JSON object", prompt_obj)

        prompt_schema = format_request_for_agy({
            "messages": [{"role": "user", "content": "Return status"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "res", "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}}},
            },
        })
        self.assertIn("JSON Schema", prompt_schema)
        self.assertIn('"ok"', prompt_schema)

    def test_banner_guard_allows_coding_prose_and_blocks_real_google_banners(self) -> None:
        class _ChunkedTextProc(_FakeAcpProcess):
            def __init__(self, chunks: list[str], error_msg: str | None = None) -> None:
                super().__init__()
                self._chunks = chunks
                self._error_msg = error_msg

            def handle_request(self, obj: dict) -> None:
                if obj.get("method") == "session/prompt":
                    self.sent.append(obj)
                    req_id = obj.get("id")
                    sid = obj["params"]["sessionId"]
                    if self._error_msg is not None:
                        self._responses.append(json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {"code": -32000, "message": self._error_msg},
                        }))
                        return
                    for txt in self._chunks:
                        self._responses.append(json.dumps({
                            "jsonrpc": "2.0",
                            "method": "session/update",
                            "params": {
                                "sessionId": sid,
                                "update": {
                                    "sessionUpdate": "agent_message_chunk",
                                    "content": {"type": "text", "text": txt},
                                },
                            },
                        }))
                    self._responses.append(json.dumps({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"usage": {"inputTokens": 5, "outputTokens": 5, "totalTokens": 10}},
                    }))
                    return
                super().handle_request(obj)

        cfg = BridgeConfig(backend="agy-cli")

        # 1. Normal coding response mentioning Terms of Service & Verification required must NOT 403.
        fake_ok = _ChunkedTextProc([
            "Berikut adalah halaman Terms of Service dan perbaikan pesan Verification required di auth.py.",
        ])
        with patch("antigravity_acp_bridge.agy_cli.subprocess.Popen", return_value=fake_ok):
            worker = AgyAcpWorker("agy", cfg)
            try:
                events = list(worker.stream_prompt("Buat halaman Terms of Service", "gemini-3.8-flash-low"))
            finally:
                worker.close()
        text = "".join(e.get("text", "") for e in events)
        self.assertIn("Terms of Service", text)

        # 2. Real Google ToS ban banner streamed in tiny chunks MUST raise HTTP 403.
        fake_ban = _ChunkedTextProc([
            "This ",
            "service has been disabled due to a violation of Google Terms of Service.",
        ])
        with patch("antigravity_acp_bridge.agy_cli.subprocess.Popen", return_value=fake_ban):
            worker = AgyAcpWorker("agy", cfg)
            try:
                with self.assertRaises(AgyCliError) as ctx:
                    list(worker.stream_prompt("hi", "gemini-3.8-flash-low"))
                self.assertEqual(ctx.exception.status, 403)
            finally:
                worker.close()

        # 3. Real Google account verification banner MUST raise HTTP 403 with link.
        verify_url = "https://accounts.google.com/signin/continue?sarp=1&auth_success_gemini=1"
        fake_verify = _ChunkedTextProc([
            f"Verification required. Please verify your account to continue: {verify_url}",
        ])
        with patch("antigravity_acp_bridge.agy_cli.subprocess.Popen", return_value=fake_verify):
            worker = AgyAcpWorker("agy", cfg)
            try:
                with self.assertRaises(AgyCliError) as ctx:
                    list(worker.stream_prompt("hi", "gemini-3.8-flash-low"))
                self.assertEqual(ctx.exception.status, 403)
                self.assertIn(verify_url, str(ctx.exception))
            finally:
                worker.close()

        # 4. Upstream rate-limit JSON-RPC error MUST map to HTTP 429 for 9router/Hermes.
        fake_429 = _ChunkedTextProc([], error_msg="RESOURCE_EXHAUSTED: rate limit exceeded")
        with patch("antigravity_acp_bridge.agy_cli.subprocess.Popen", return_value=fake_429):
            worker = AgyAcpWorker("agy", cfg)
            try:
                with self.assertRaises(AgyCliError) as ctx:
                    list(worker.stream_prompt("hi", "gemini-3.8-flash-low"))
                self.assertEqual(ctx.exception.status, 429)
            finally:
                worker.close()

    def test_transient_failed_to_start_retries_and_rate_limit_maps_429(self) -> None:
        attempts = {"count": 0}

        class _FlakyWorker(_LiveWorker):
            def stream_prompt(self, prompt, model, *, has_external_tools=False, image_blocks=None):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise AgyCliError("agy ACP error: Failed to start Antigravity conversation", status=502)
                yield {"text": "recovered!"}
                yield {"finish_reason": "stop", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

        client = AgyCliClient(BridgeConfig(backend="agy-cli", max_concurrency=2, worker_idle_timeout=0))
        try:
            with patch("antigravity_acp_bridge.agy_cli.shutil.which", return_value="agy"), patch(
                "antigravity_acp_bridge.agy_cli.AgyAcpWorker", side_effect=lambda *a, **k: _FlakyWorker()
            ), patch("antigravity_acp_bridge.agy_cli.time.sleep"):
                resp = client.call_chat({"messages": [{"role": "user", "content": "hi"}]}, "gemini-3.8-flash-low")
            self.assertEqual(attempts["count"], 2)
            self.assertEqual(resp["choices"][0]["message"]["content"], "recovered!")
        finally:
            client.close()

    def test_closed_pool_does_not_spawn_new_worker(self) -> None:
        client = AgyCliClient(BridgeConfig(backend="agy-cli", max_concurrency=2, worker_idle_timeout=0))
        client.close()
        with self.assertRaises(AgyCliError) as ctx:
            client._acquire_worker()
        self.assertEqual(ctx.exception.status, 503)
