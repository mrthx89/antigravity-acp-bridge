"""Unit tests for OpenAI <-> Gemini/Antigravity translation."""

from __future__ import annotations

import json
import unittest

from antigravity_acp_bridge.translator import (
    SKIP_THOUGHT_SIGNATURE,
    build_gemini_request,
    build_passthrough_envelope,
    content_to_text,
    decode_tool_call_id,
    encode_tool_call_id,
    extract_delta_from_parts,
    gemini_response_to_openai,
    openai_tools_to_gemini,
    tool_choice_to_gemini,
)


class TranslatorTests(unittest.TestCase):
    def test_content_to_text_variants(self) -> None:
        self.assertEqual(content_to_text(None), "")
        self.assertEqual(content_to_text("hello"), "hello")
        self.assertEqual(
            content_to_text([
                {"type": "text", "text": "line 1"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                {"type": "input_text", "text": "line 2"},
            ]),
            "line 1\n[image: https://example.com/a.png]\nline 2",
        )

    def test_tool_call_id_round_trip(self) -> None:
        encoded = encode_tool_call_id("fc99", "sigABC")
        self.assertEqual(encoded, "call_fc99|sigABC")
        self.assertEqual(decode_tool_call_id(encoded), ("fc99", "sigABC"))
        self.assertEqual(decode_tool_call_id("call_onlyid"), ("onlyid", ""))

    def test_tools_and_tool_choice_conversion(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Fetch weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        gemini_tools = openai_tools_to_gemini(tools)
        self.assertEqual(gemini_tools[0]["functionDeclarations"][0]["name"], "get_weather")
        self.assertEqual(tool_choice_to_gemini("auto"), "AUTO")
        self.assertEqual(tool_choice_to_gemini("none"), "NONE")
        self.assertEqual(tool_choice_to_gemini("required"), "ANY")
        self.assertEqual(
            tool_choice_to_gemini({"type": "function", "function": {"name": "get_weather"}}),
            "ANY",
        )

    def test_build_gemini_request_round_trips_tool_history(self) -> None:
        body = {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_fc1|sig1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Jakarta"}'},
                        },
                        {
                            "id": "call_no_sig",
                            "type": "function",
                            "function": {"name": "ping", "arguments": "{}"},
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_fc1|sig1",
                    "name": "get_weather",
                    "content": '{"temp": 30}',
                },
            ],
            "temperature": 0.2,
            "max_completion_tokens": 256,
        }
        env = build_gemini_request(
            body,
            "gemini-4-hydrogen-high",
            "proj-1",
            thinking_level="HIGH",
        )
        self.assertEqual(env["project"], "proj-1")
        self.assertEqual(env["model"], "gemini-4-hydrogen-high")
        inner = env["request"]
        self.assertEqual(inner["systemInstruction"]["parts"][0]["text"], "Be brief.")
        self.assertEqual(inner["generationConfig"]["maxOutputTokens"], 256)
        self.assertEqual(inner["generationConfig"]["thinkingConfig"]["thinkingLevel"], "HIGH")
        model_parts = inner["contents"][1]["parts"]
        self.assertEqual(model_parts[0]["thoughtSignature"], "sig1")
        self.assertEqual(model_parts[1]["thoughtSignature"], SKIP_THOUGHT_SIGNATURE)
        tool_resp = inner["contents"][2]["parts"][0]["functionResponse"]
        self.assertEqual(tool_resp["id"], "fc1")
        self.assertEqual(tool_resp["response"], {"temp": 30})

    def test_build_passthrough_envelope_maps_model(self) -> None:
        env = build_passthrough_envelope(
            "claude-sonnet-4.6",
            {"contents": [{"parts": [{"text": "hi"}]}]},
            "proj-2",
        )
        self.assertEqual(env["model"], "claude-sonnet-4-6")
        self.assertEqual(env["project"], "proj-2")
        self.assertEqual(len(env["request"]["safetySettings"]), 4)

    def test_gemini_response_to_openai_and_streaming_delta(self) -> None:
        gemini_resp = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Checking..."},
                            {
                                "functionCall": {"id": "fc7", "name": "lookup", "args": {"q": "a"}},
                                "thoughtSignature": "sig7",
                            },
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        }
        out = gemini_response_to_openai(gemini_resp, "gemini-3.8-flash-high")
        choice = out["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["tool_calls"][0]["id"], "call_fc7|sig7")
        self.assertEqual(json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]), {"q": "a"})
        self.assertEqual(out["usage"]["total_tokens"], 15)

        text, deltas, names = extract_delta_from_parts(
            gemini_resp["candidates"][0]["content"]["parts"], set()
        )
        self.assertEqual(text, "Checking...")
        self.assertEqual(deltas[0]["index"], 0)
        self.assertEqual(names, {"lookup"})
