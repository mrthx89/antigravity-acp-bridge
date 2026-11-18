"""OAuth token lifecycle management and Cloud Code Assist project discovery."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ._http import build_antigravity_request, read_http_error_detail
from .config import LOAD_CODEASSIST_URL, OAUTH_TOKEN_URL, TOKEN_REFRESH_SKEW, ProxyConfig

logger = logging.getLogger("antigravity_proxy.auth")


def parse_expiry(expiry_raw) -> float:
    """Parse an RFC3339 or numeric expiry value into a Unix timestamp.

    The Antigravity CLI stores expiry as an RFC3339 string with nanosecond
    precision (e.g. ``2026-06-30T12:55:03.123456789Z``). Python's
    ``datetime.fromisoformat`` supports up to 6 fractional digits, so sub-second
    precision is truncated to microseconds.
    """
    if not expiry_raw:
        return 0.0
    if isinstance(expiry_raw, (int, float)):
        return float(expiry_raw)

    raw = str(expiry_raw).strip()
    try:
        if "." in raw:
            head, tail = raw.split(".", 1)
            tz_part = ""
            for idx, ch in enumerate(tail):
                if ch in "Z+-":
                    tz_part = tail[idx:]
                    tail = tail[:idx]
                    break
            raw = f"{head}.{tail[:6]}{tz_part}"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0


def deep_find(obj, key: str):
    """Recursively search nested dicts/lists for ``key`` and return the first match."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = deep_find(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = deep_find(item, key)
            if found is not None:
                return found
    return None


class TokenManager:
    """Thread-safe reader, refresher, and disk persister for Antigravity OAuth tokens."""

    def __init__(self, config: ProxyConfig) -> None:
        self._config = config
        self._lock = threading.Lock()

    @property
    def token_file(self) -> str:
        return self._config.token_file

    def read_token_from_disk(self) -> dict | None:
        """Read and parse the OAuth token JSON file, returning None on error."""
        try:
            with open(self._config.token_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read token file %s: %s", self._config.token_file, exc)
            return None

    def write_token_to_disk(self, data: dict) -> None:
        """Atomically persist updated token data to disk with restrictive permissions."""
        tmp_path = f"{self._config.token_file}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp_path, self._config.token_file)
            try:
                os.chmod(self._config.token_file, 0o600)
            except OSError:
                pass
        except OSError as exc:
            logger.warning("Could not write token file %s: %s", self._config.token_file, exc)

    def refresh_access_token(self, refresh_token: str) -> dict:
        """Exchange a refresh token for a fresh access token via Google OAuth."""
        body = json.dumps({
            "client_id": self._config.client_id,
            "client_secret": self._config.client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }).encode("utf-8")
        req = Request(OAUTH_TOKEN_URL, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = read_http_error_detail(exc)
            raise RuntimeError(f"OAuth refresh failed (HTTP {exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"OAuth refresh network error: {exc}") from exc

        expires_in = int(payload.get("expires_in", 3600))
        exp_dt = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        return {
            "access_token": payload["access_token"],
            "refresh_token": payload.get("refresh_token", refresh_token),
            "token_type": payload.get("token_type", "Bearer"),
            "expiry": exp_dt.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
        }

    def get_access_token(self) -> str:
        """Return a valid access token, refreshing and persisting if near expiry."""
        with self._lock:
            data = self.read_token_from_disk()
            if not data:
                raise RuntimeError(
                    f"No OAuth token found at {self._config.token_file}. "
                    "Run `agy` to authenticate first."
                )

            token_obj = data.get("token") or {}
            access_token = token_obj.get("access_token")
            expiry_ts = parse_expiry(token_obj.get("expiry"))
            needs_refresh = (
                not access_token
                or expiry_ts == 0.0
                or (expiry_ts - time.time()) < TOKEN_REFRESH_SKEW
            )

            if needs_refresh:
                refresh_token = token_obj.get("refresh_token")
                if not refresh_token:
                    raise RuntimeError(
                        "No refresh_token available; re-run `agy` to authenticate."
                    )
                logger.info("Access token expired or missing — refreshing...")
                new_tok = self.refresh_access_token(refresh_token)
                data["token"] = new_tok
                self.write_token_to_disk(data)
                access_token = new_tok["access_token"]

            return access_token

    def preflight_check(self) -> bool:
        """Validate that the OAuth token file exists, parses, and has a refresh_token."""
        if not os.path.isfile(self._config.token_file):
            logger.error("Token file not found at %s", self._config.token_file)
            logger.error("Run the Antigravity CLI (`agy`) to authenticate first.")
            return False
        data = self.read_token_from_disk()
        if not data:
            logger.error("Token file exists but could not be parsed.")
            return False
        token_obj = data.get("token") or {}
        if not token_obj.get("refresh_token"):
            logger.error("No refresh_token in token file. Re-run `agy` to authenticate.")
            return False
        logger.info("Token file OK (auth_method=%s)", data.get("auth_method", "unknown"))
        return True


class ProjectResolver:
    """Thread-safe, cached Cloud Code Assist project ID resolver."""

    def __init__(self, config: ProxyConfig, token_manager: TokenManager) -> None:
        self._config = config
        self._token_manager = token_manager
        self._lock = threading.Lock()
        self._cached_project_id: str | None = None

    def load_code_assist(self, access_token: str) -> str:
        """Query v1internal:loadCodeAssist to discover the cloudaicompanion project ID."""
        # Empty JSON body is intentional: the backend rejects metadata fields here,
        # while Client-Metadata is sent via HTTP headers.
        req = build_antigravity_request(LOAD_CODEASSIST_URL, b"{}", access_token)
        try:
            with urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = read_http_error_detail(exc)
            raise RuntimeError(f"loadCodeAssist failed (HTTP {exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"loadCodeAssist network error: {exc}") from exc

        project_id = (
            payload.get("cloudaicompanionProject")
            or payload.get("cloudaicompanion_project")
            or deep_find(payload, "cloudaicompanionProject")
        )
        if not project_id:
            raise RuntimeError(
                f"loadCodeAssist did not return a project ID. Response: {payload}"
            )
        return project_id

    def get_project_id(self) -> str:
        """Return cached project ID, discovering or reading override if needed."""
        with self._lock:
            if self._cached_project_id:
                return self._cached_project_id

            if self._config.project_id_override:
                logger.info("Using project ID override: %s", self._config.project_id_override)
                pid = self._config.project_id_override
            else:
                token = self._token_manager.get_access_token()
                pid = self.load_code_assist(token)
                logger.info("Discovered project ID: %s", pid)

            self._cached_project_id = pid
            return pid
