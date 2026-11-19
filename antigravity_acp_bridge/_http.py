"""Shared HTTP request/error helpers for Google Cloud Code Assist calls."""

from __future__ import annotations

from urllib.error import HTTPError
from urllib.request import Request

from .config import ANTIGRAVITY_USER_AGENT, CLIENT_METADATA, GOOG_API_CLIENT


def read_http_error_detail(err: HTTPError) -> str:
    """Safely decode the body of an HTTPError for logging/diagnostics."""
    try:
        return err.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def build_antigravity_request(
    url: str,
    body: bytes,
    access_token: str,
    *,
    streaming: bool = False,
) -> Request:
    """Construct a urllib Request carrying standard Antigravity headers."""
    req = Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", ANTIGRAVITY_USER_AGENT)
    req.add_header("X-Goog-Api-Client", GOOG_API_CLIENT)
    req.add_header("Client-Metadata", CLIENT_METADATA)
    if streaming:
        req.add_header("Accept", "text/event-stream")
    return req
