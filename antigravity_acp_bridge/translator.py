"""OpenAI <-> Gemini/Antigravity request and response translators."""

from __future__ import annotations

import json
import time
import uuid

from .config import (
    DEFAULT_SAFETY_SETTINGS,
    DEFAULT_SYSTEM_INSTRUCTION,
    GEMINI_FINISH_REASON_MAP,
    MODEL_MAP,
)

SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator"


def openai_error(
    message: str,
    err_type: str = "api_error",
    code: str | int | None = None,
) -> dict:
    """Format an error payload matching OpenAI's API error schema."""
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": None,
            "code": code,
        }
    }


def encode_tool_call_id(fc_id: str, thought_sig: str) -> str:
    """Encode Gemini functionCall.id and thoughtSignature into an OpenAI tool_call.id."""
    if not (fc_id or thought_sig):
        return ""
    return f"call_{fc_id}|{thought_sig}"


def decode_tool_call_id(tool_call_id: str) -> tuple[str, str]:
    """Decode ``call_<fc_id>|<thought_signature>`` back into ``(fc_id, thought_sig)``."""
    if not tool_call_id or not tool_call_id.startswith("call_"):
        return ("", "")
    raw = tool_call_id[5:]
    if "|" in raw:
        fc_id, thought_sig = raw.split("|", 1)
        return (fc_id, thought_sig)
    return (raw, "")


def parse_tool_arguments(args_raw) -> dict:
    """Parse OpenAI tool-call arguments string/object into a dict for Gemini."""
    try:
        parsed = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (json.JSONDecodeError, TypeError):
        return {"raw": str(args_raw)}


def content_to_text(content) -> str:
    """Normalize an OpenAI message ``content`` field to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    chunks: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            chunks.append(str(block))
            continue
        btype = block.get("type", "text")
        if btype in ("text", "input_text"):
            chunks.append(block.get("text", ""))
        elif btype == "image_url":
            iu = block.get("image_url")
            url = iu.get("url", "") if isinstance(iu, dict) else (iu if isinstance(iu, str) else "")
            label = f"{url[:60]}..." if url.startswith("data:") else url
            chunks.append(f"[image: {label}]")
        elif btype == "input_image":
            chunks.append("[image provided]")
        else:
            chunks.append(f"[{btype}: {json.dumps(block.get('text', ''))[:100]}]")
    return "\n".join(c for c in chunks if c)


def openai_tools_to_gemini(tools: list | None) -> list[dict]:
    """Convert OpenAI ``tools`` array to Gemini ``functionDeclarations`` format."""
    if not tools:
        return []
    decls: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if not name:
            continue
        decl: dict = {
            "name": name,
            "description": func.get("description", ""),
        }
        params = func.get("parameters")
        if isinstance(params, dict) and params:
            decl["parameters"] = params
        decls.append(decl)
    return [{"functionDeclarations": decls}] if decls else []


def tool_choice_to_gemini(tool_choice) -> str | None:
    """Convert OpenAI ``tool_choice`` to Gemini ``functionCallingConfig.mode``."""
    simple = {"auto": "AUTO", "none": "NONE", "required": "ANY"}
    if isinstance(tool_choice, str):
        return simple.get(tool_choice)
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return "ANY"
    return None


def build_antigravity_envelope(
    project_id: str,
    backend_model: str,
    inner_request: dict,
) -> dict:
    """Wrap an inner Gemini request payload in the Cloud Code Assist envelope."""
    return {
        "project": project_id,
        "model": backend_model,
        "request": inner_request,
        "requestType": "agent",
        "userAgent": "antigravity",
        "requestId": f"agent/{uuid.uuid4()}/{int(time.time() * 1000)}/{uuid.uuid4()}/1",
    }


def build_passthrough_envelope(
    model: str,
    gemini_body: dict,
    project_id: str,
) -> dict:
    """Wrap a native Gemini ``generateContent`` body in the Antigravity envelope."""
    entry = MODEL_MAP.get(model)
    backend_model = entry[0] if entry else model
    thinking_level = entry[1] if entry else None

    inner = dict(gemini_body)
    if thinking_level:
        gc = inner.setdefault("generationConfig", {})
        tc = gc.setdefault("thinkingConfig", {})
        tc.setdefault("thinkingLevel", thinking_level)

    inner.setdefault("safetySettings", list(DEFAULT_SAFETY_SETTINGS))
    return build_antigravity_envelope(project_id, backend_model, inner)


def _convert_tool_message(msg: dict) -> dict:
    tool_call_id = msg.get("tool_call_id", "")
    content_text = content_to_text(msg.get("content"))
    response_obj: dict = {}
    if content_text:
        try:
            parsed = json.loads(content_text)
            response_obj = parsed if isinstance(parsed, dict) else {"result": parsed}
        except (json.JSONDecodeError, TypeError):
            response_obj = {"result": content_text}

    fn_name = msg.get("name") or tool_call_id or "tool_result"
    fc_id, _ = decode_tool_call_id(tool_call_id)
    func_resp: dict = {"name": fn_name, "response": response_obj}
    if fc_id:
        func_resp["id"] = fc_id
    return {"role": "function", "parts": [{"functionResponse": func_resp}]}


def _convert_assistant_message(msg: dict) -> dict | None:
    parts: list[dict] = []
    text = content_to_text(msg.get("content"))
    if text:
        parts.append({"text": text})

    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        fn_name = fn.get("name", "")
        if not fn_name:
            continue
        fc_id, thought_sig = decode_tool_call_id(tc.get("id", "") or "")
        fc_obj: dict = {
            "name": fn_name,
            "args": parse_tool_arguments(fn.get("arguments", "{}")),
        }
        if fc_id:
            fc_obj["id"] = fc_id
        # thoughtSignature sits next to functionCall; sentinel skips validation when absent.
        parts.append({
            "functionCall": fc_obj,
            "thoughtSignature": thought_sig or SKIP_THOUGHT_SIGNATURE,
        })

    return {"role": "model", "parts": parts} if parts else None


def _convert_user_message(msg: dict) -> dict | None:
    parts: list[dict] = []
    text = content_to_text(msg.get("content"))
    if text:
        parts.append({"text": text})
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        fn_name = fn.get("name", "")
        if fn_name:
            parts.append({
                "functionCall": {
                    "name": fn_name,
                    "args": parse_tool_arguments(fn.get("arguments", "{}")),
                }
            })
    return {"role": "user", "parts": parts} if parts else None


def _build_generation_config(
    body: dict,
    backend_model: str,
    thinking_level: str | None,
) -> dict:
    gen_config: dict = {}
    float_fields = (
        ("temperature", "temperature"),
        ("top_p", "topP"),
        ("presence_penalty", "presencePenalty"),
        ("frequency_penalty", "frequencyPenalty"),
    )
    for src_key, dst_key in float_fields:
        if body.get(src_key) is not None:
            gen_config[dst_key] = float(body[src_key])

    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    if max_tokens is not None:
        gen_config["maxOutputTokens"] = int(max_tokens)

    if body.get("stop"):
        stops = body["stop"]
        gen_config["stopSequences"] = [stops] if isinstance(stops, str) else stops
    if body.get("seed") is not None:
        gen_config["seed"] = int(body["seed"])

    if backend_model in ("claude-opus-4-6-thinking", "claude-sonnet-4-6"):
        thinking_cfg = gen_config.setdefault("thinkingConfig", {})
        thinking_cfg.setdefault("includeThoughts", False)
    if thinking_level:
        thinking_cfg = gen_config.setdefault("thinkingConfig", {})
        thinking_cfg.setdefault("thinkingLevel", thinking_level)
        thinking_cfg.setdefault("includeThoughts", False)

    return gen_config


def build_gemini_request(
    body: dict,
    backend_model: str,
    project_id: str,
    thinking_level: str | None = None,
) -> dict:
    """Transform an OpenAI chat completion body into an Antigravity request envelope."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        messages = []

    system_texts: list[str] = []
    contents: list[dict] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        if role in ("system", "developer"):
            text = content_to_text(msg.get("content"))
            if text:
                system_texts.append(text)
        elif role == "tool":
            contents.append(_convert_tool_message(msg))
        elif role == "assistant":
            converted = _convert_assistant_message(msg)
            if converted:
                contents.append(converted)
        else:
            converted = _convert_user_message(msg)
            if converted:
                contents.append(converted)

    system_text = "\n\n".join(system_texts) if system_texts else DEFAULT_SYSTEM_INSTRUCTION
    inner: dict = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_text}]},
        "safetySettings": list(DEFAULT_SAFETY_SETTINGS),
    }

    tools = openai_tools_to_gemini(body.get("tools"))
    if tools:
        inner["tools"] = tools

    tool_choice = body.get("tool_choice")
    mode = tool_choice_to_gemini(tool_choice)
    if mode:
        tc_config: dict = {"functionCallingConfig": {"mode": mode}}
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            fn_name = (tool_choice.get("function") or {}).get("name")
            if fn_name:
                tc_config["functionCallingConfig"]["allowedFunctionNames"] = [fn_name]
        inner["toolConfig"] = tc_config

    gen_config = _build_generation_config(body, backend_model, thinking_level)
    if gen_config:
        inner["generationConfig"] = gen_config

    return build_antigravity_envelope(project_id, backend_model, inner)


def _format_function_call_part(part: dict) -> dict | None:
    fc = part.get("functionCall") or {}
    name = fc.get("name", "")
    if not name:
        return None
    args = fc.get("args", {})
    fc_id = fc.get("id", "")
    thought_sig = part.get("thoughtSignature") or part.get("thought_signature") or ""
    return {
        "id": encode_tool_call_id(fc_id, thought_sig),
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args) if isinstance(args, (dict, list)) else str(args),
        },
    }


def extract_parts(parts: list | None) -> tuple[str, list[dict], str | None]:
    """Extract ``(text, tool_calls, finish_reason_hint)`` from Gemini ``parts``."""
    text_chunks: list[str] = []
    tool_calls: list[dict] = []

    for part in parts or []:
        if not isinstance(part, dict):
            continue
        if part.get("text"):
            text_chunks.append(part["text"])
        if "functionCall" in part:
            tc = _format_function_call_part(part)
            if tc:
                tool_calls.append(tc)
        if "executableCode" in part:
            code = (part["executableCode"] or {}).get("code", "")
            if code:
                text_chunks.append(f"```python\n{code}\n```")
        if "codeExecutionResult" in part:
            out = (part["codeExecutionResult"] or {}).get("output", "")
            if out:
                text_chunks.append(f"```\n{out}\n```")

    finish_hint = "tool_calls" if tool_calls else None
    return "".join(text_chunks), tool_calls, finish_hint


def map_finish_reason(candidate: dict, has_tool_calls: bool) -> str:
    """Map Gemini ``finishReason`` / ``stopReason`` to OpenAI ``finish_reason``."""
    if has_tool_calls:
        return "tool_calls"
    raw = (candidate.get("finishReason") or candidate.get("stopReason") or "").upper()
    return GEMINI_FINISH_REASON_MAP.get(raw, "stop")


def extract_usage(response: dict, usage_override: dict | None = None) -> dict:
    """Extract OpenAI-format token usage counters from a Gemini response."""
    usage = usage_override or response.get("usageMetadata") or {}
    if not usage:
        cands = response.get("candidates") or []
        if cands:
            usage = cands[0].get("usageMetadata") or {}
    pt = usage.get("promptTokenCount", 0) or 0
    ct = usage.get("candidatesTokenCount", 0) or usage.get("completionTokenCount", 0) or 0
    tt = usage.get("totalTokenCount", pt + ct) or (pt + ct)
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
    }


def gemini_response_to_openai(response: dict, model: str) -> dict:
    """Transform an unwrapped Gemini ``generateContent`` response to OpenAI format."""
    candidates = response.get("candidates") or []
    text = ""
    tool_calls: list[dict] = []
    finish_reason = "stop"

    if candidates:
        cand = candidates[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text, tool_calls, _ = extract_parts(parts)
        finish_reason = map_finish_reason(cand, bool(tool_calls))
        for tc in tool_calls:
            if not tc["id"]:
                tc["id"] = f"call_{uuid.uuid4().hex[:24]}"
    else:
        block_reason = ((response.get("promptFeedback") or {}).get("blockReason") or "").upper()
        if block_reason:
            finish_reason = "content_filter"
            text = f"[Content blocked: {block_reason}]"

    message: dict = {"role": "assistant", "content": text if text else None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": extract_usage(response),
    }


def extract_delta_from_parts(
    parts: list | None,
    prev_tool_names: set[str],
) -> tuple[str, list[dict], set[str]]:
    """Extract incremental ``(text_delta, tool_call_deltas, seen_tool_names)``."""
    text_chunks: list[str] = []
    tool_call_deltas: list[dict] = []
    current_names = set(prev_tool_names)

    for part in parts or []:
        if not isinstance(part, dict):
            continue
        if part.get("text"):
            text_chunks.append(part["text"])
        if "functionCall" in part:
            tc = _format_function_call_part(part)
            if tc:
                current_names.add(tc["function"]["name"])
                tc["index"] = len(current_names) - 1
                if not tc["id"]:
                    tc["id"] = f"call_{uuid.uuid4().hex[:24]}"
                tool_call_deltas.append(tc)

    return "".join(text_chunks), tool_call_deltas, current_names
