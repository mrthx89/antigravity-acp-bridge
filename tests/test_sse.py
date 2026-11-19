"""Unit tests for SSE and JSON stream parsing."""

from __future__ import annotations

import unittest

from antigravity_acp_bridge.sse import (
    parse_remaining,
    parse_sse_block,
    try_parse_one_event,
    try_parse_one_json_object,
)


class SseParsingTests(unittest.TestCase):
    def test_parse_sse_block_json_and_done(self) -> None:
        self.assertEqual(
            parse_sse_block("event: message\ndata: {\"a\": 1}"),
            {"a": 1},
        )
        self.assertEqual(parse_sse_block("data: [DONE]"), {"__done__": True})
        self.assertIsNone(parse_sse_block(": heartbeat"))

    def test_try_parse_one_event_sse_and_json_array(self) -> None:
        buf = 'data: {"x": 1}\n\ndata: {"x": 2}\n\n'
        obj, consumed = try_parse_one_event(buf)
        self.assertEqual(obj, {"x": 1})
        obj2, _ = try_parse_one_event(buf[consumed:])
        self.assertEqual(obj2, {"x": 2})

        arr = '[{"text": "hi"}, {"text": "there"}]'
        first, used = try_parse_one_event(arr)
        self.assertEqual(first, {"text": "hi"})
        second, _ = try_parse_one_event(arr[used:])
        self.assertEqual(second, {"text": "there"})

    def test_try_parse_one_json_object_with_braces_in_strings(self) -> None:
        raw = '{"msg": "hello } world", "n": 2}{"next": true}'
        obj, consumed = try_parse_one_json_object(raw)
        self.assertEqual(obj, {"msg": "hello } world", "n": 2})
        self.assertEqual(raw[consumed:], '{"next": true}')

    def test_parse_remaining_handles_sse_and_bare_list(self) -> None:
        self.assertEqual(
            list(parse_remaining("data: {\"k\": 1}\ndata: [DONE]")),
            [{"k": 1}, {"__done__": True}],
        )
        self.assertEqual(list(parse_remaining('[{"a": 1}, {"b": 2}]')), [{"a": 1}, {"b": 2}])
