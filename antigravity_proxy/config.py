"""Configuration, constants, and model mappings for Antigravity Proxy."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

# OAuth client credentials embedded in the Antigravity CLI binary.
DEFAULT_CLIENT_ID = (
    "1071006060591-tmhssin2h21lcre2" "35vtolojh4g403ep.apps.googleusercontent.com"
)
DEFAULT_CLIENT_SECRET = "GOCSPX-K58FWR486LdL" "J1mLB8sXC4z6qDAf"

DEFAULT_TOKEN_FILE = os.path.expanduser(
    "~/.gemini/antigravity-cli/antigravity-oauth-token"
)
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
CLOUDCODE_BASE = "https://daily-cloudcode-pa.googleapis.com"
LOAD_CODEASSIST_URL = f"{CLOUDCODE_BASE}/v1internal:loadCodeAssist"
GENERATE_CONTENT_URL = f"{CLOUDCODE_BASE}/v1internal:generateContent"
STREAM_GENERATE_CONTENT_URL = (
    f"{CLOUDCODE_BASE}/v1internal:streamGenerateContent?alt=sse"
)

ANTIGRAVITY_USER_AGENT = (
    "antigravity/cli/1.5.10 "
    "(aidev_client; os_type=linux; arch=amd64; cl=1023297289; auth_method=consumer)"
)
GOOG_API_CLIENT = "google-cloud-sdk vscode_cloudshelleditor/0.1"
CLIENT_METADATA = json.dumps(
    {"ideType": "ANTIGRAVITY", "platform": "MACOS", "pluginType": "GEMINI"},
    separators=(",", ":"),
)

DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful AI assistant."
UPSTREAM_TIMEOUT = 300  # seconds
TOKEN_REFRESH_SKEW = 120  # refresh this many seconds before expiry

# Transient upstream conditions worth retrying before failing the client call.
# daily-cloudcode-pa returns these when a model's capacity is momentarily gone;
# the same request usually succeeds a few seconds later.
UPSTREAM_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
UPSTREAM_RETRY_ATTEMPTS = 4
UPSTREAM_RETRY_BASE_DELAY = 2.0  # seconds; doubles per attempt
UPSTREAM_RETRY_MAX_DELAY = 30.0  # cap for Retry-After and backoff
MAX_REQUEST_BODY_BYTES = 20 * 1024 * 1024  # 20 MiB guard against oversized bodies

LOCALHOST_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# NOTE: HARM_CATEGORY_CIVIC_INTEGRITY is rejected by this API variant even
# though it appears in error responses — only these 4 categories are accepted.
DEFAULT_SAFETY_SETTINGS = [
    {"category": cat, "threshold": "BLOCK_NONE"}
    for cat in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]

GEMINI_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "MALFORMED_FUNCTION_CALL": "stop",
    "IMAGE_SAFETY": "content_filter",
    "LANGUAGE": "content_filter",
    "OTHER": "stop",
}

_THINKING_LEVELS = {
    "high": "HIGH",
    "xhigh": "HIGH",
    "medium": "MEDIUM",
    "low": "LOW",
}


def _lvl(suffix: str) -> str | None:
    return _THINKING_LEVELS.get(suffix)


# OpenAI-facing model name -> (backend_model_id, thinking_level)
MODEL_MAP: dict[str, tuple[str, str | None]] = {
    # Gemini 3.8 Flash
    "gemini-3.8-flash-high": ("gemini-3.8-flash-high", _lvl("high")),
    "gemini-3.8-flash-medium": ("gemini-3.8-flash-medium", _lvl("medium")),
    "gemini-3.8-flash-low": ("gemini-3.8-flash-low", _lvl("low")),
    # Gemini 4 Hydrogen
    "gemini-4-hydrogen-high": ("gemini-4-hydrogen-high", _lvl("high")),
    "gemini-4-hydrogen-xhigh": ("gemini-4-hydrogen-xhigh", _lvl("xhigh")),
    "gemini-4-hydrogen-medium": ("gemini-4-hydrogen-medium", _lvl("medium")),
    "gemini-4-hydrogen-low": ("gemini-4-hydrogen-low", _lvl("low")),
    # Gemini 4 Argon
    "gemini-4-argon-high": ("gemini-4-argon-high", _lvl("high")),
    "gemini-4-argon-medium": ("gemini-4-argon-medium", _lvl("medium")),
    "gemini-4-argon-low": ("gemini-4-argon-low", _lvl("low")),
    # Gemini 1 Pro Orchestrator
    "gemini-pro-orchestrator-g1-pro": ("gemini-pro-orchestrator-g1-pro", _lvl("high")),
    # GPT-OSS 120B
    "gpt-oss-120b": ("gpt-oss-120b-medium", None),
    "gpt-oss-120b-medium": ("gpt-oss-120b-medium", None),
    # Claude Sonnet / Opus 4.6
    "claude-sonnet-4-6": ("claude-sonnet-4-6", None),
    "claude-sonnet-4.6": ("claude-sonnet-4-6", None),
    "claude-opus-4.6": ("claude-opus-4-6-thinking", None),
    # Claude Sonnet 5.5
    "claude-sonnet-5-5-low": ("claude-sonnet-5-5-low", None),
    "claude-sonnet-5-5-medium": ("claude-sonnet-5-5-medium", None),
    "claude-sonnet-5-5-high": ("claude-sonnet-5-5-high", None),
    # Claude Opus 5.5
    "claude-opus-5-5-low": ("claude-opus-5-5-low", None),
    "claude-opus-5-5-medium": ("claude-opus-5-5-medium", None),
    "claude-opus-5-5-high": ("claude-opus-5-5-high", None),
}

DEFAULT_MODEL = "gemini-3.8-flash-high"


@dataclass(frozen=True)
class ProxyConfig:
    """Runtime configuration for the proxy server."""

    host: str = "127.0.0.1"
    port: int = 8877
    api_key: str | None = None
    allow_unauthenticated: bool = False
    cors_origins: str = "*"
    token_file: str = DEFAULT_TOKEN_FILE
    client_id: str = DEFAULT_CLIENT_ID
    client_secret: str = DEFAULT_CLIENT_SECRET
    project_id_override: str | None = None
    upstream_timeout: int = UPSTREAM_TIMEOUT

    @property
    def is_localhost(self) -> bool:
        """Return True if bound strictly to a loopback interface."""
        return self.host in LOCALHOST_HOSTS

    @classmethod
    def from_env(cls, **overrides) -> ProxyConfig:
        """Build a ProxyConfig seeded from environment variables."""
        defaults = {
            "host": os.environ.get("ANTIGRAVITY_PROXY_HOST", "127.0.0.1"),
            "port": int(os.environ.get("ANTIGRAVITY_PROXY_PORT", "8877")),
            "api_key": os.environ.get("ANTIGRAVITY_PROXY_API_KEY") or None,
            "cors_origins": os.environ.get("ANTIGRAVITY_CORS_ORIGINS", "*"),
            "token_file": os.path.expanduser(
                os.environ.get("ANTIGRAVITY_TOKEN_FILE", DEFAULT_TOKEN_FILE)
            ),
            "client_id": os.environ.get("ANTIGRAVITY_CLIENT_ID", DEFAULT_CLIENT_ID),
            "client_secret": os.environ.get(
                "ANTIGRAVITY_CLIENT_SECRET", DEFAULT_CLIENT_SECRET
            ),
            "project_id_override": os.environ.get("ANTIGRAVITY_PROJECT_ID") or None,
        }
        defaults.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**defaults)
