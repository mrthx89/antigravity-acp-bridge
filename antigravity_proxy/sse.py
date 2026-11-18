"""Incremental SSE and JSON stream parsers for Cloud Code Assist responses."""

from __future__ import annotations

import json
from typing import Iterator


def parse_sse_block(block: str) -> dict | None:
    """Parse one SSE event block into a JSON object, a done sentinel, or None."""
    data_lines: list[str] = []
    for raw_line in block.split("\n"):
        line = raw_line.rstrip("\r")
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        # Ignore SSE comments/heartbeats and metadata fields (event, id, retry).

    if not data_lines:
        return None

    data_str = "\n".join(data_lines).strip()
    if data_str == "[DONE]":
        return {"__done__": True}
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        return None


def try_parse_one_json_object(buffer: str) -> tuple[dict | None, int]:
    """Parse one complete top-level JSON object from ``buffer`` using brace counting.

    Returns ``(obj, consumed_chars)`` or ``(None, 0)`` if the object is incomplete.
    """
    start = buffer.find("{")
    if start == -1:
        return (None, 0)

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(buffer)):
        ch = buffer[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = buffer[start : idx + 1]
                try:
                    return (json.loads(candidate), idx + 1)
                except json.JSONDecodeError:
                    continue

    return (None, 0)


def try_parse_one_event(buffer: str) -> tuple[dict | None, int] | None:
    """Attempt to parse one SSE event or JSON stream element from ``buffer``.

    Returns ``(event_obj_or_None, chars_consumed)`` or ``None`` if incomplete.
    """
    if "data:" in buffer[:10] or buffer.startswith(("\n", ":")):
        for sep in ("\n\n", "\r\n\r\n"):
            idx = buffer.find(sep)
            if idx != -1:
                block = buffer[:idx]
                return (parse_sse_block(block), idx + len(sep))
        nl = buffer.find("\n")
        if nl != -1 and buffer.strip().startswith("data:"):
            return (parse_sse_block(buffer[:nl]), nl + 1)
        return None

    stripped = buffer.lstrip()
    if stripped.startswith("]"):
        return (None, buffer.index("]") + 1)
    if stripped.startswith((",", "[", "{")):
        obj, consumed = try_parse_one_json_object(buffer)
        if obj is not None:
            return (obj, consumed)
        return None

    nl = buffer.find("\n")
    if nl != -1:
        return (None, nl + 1)
    return None


def parse_remaining(buffer: str) -> Iterator[dict]:
    """Flush any remaining complete SSE lines or bare JSON objects at stream end."""
    stripped = buffer.strip()
    if not stripped:
        return

    for raw_line in stripped.split("\n"):
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            yield {"__done__": True}
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            pass

    if stripped.startswith(("[", "{")):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, list):
                yield from obj
            else:
                yield obj
        except json.JSONDecodeError:
            pass
