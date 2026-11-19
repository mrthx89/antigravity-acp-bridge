"""Unit tests for OAuth token lifecycle and Cloud Code Assist discovery."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from antigravity_acp_bridge.auth import ProjectResolver, TokenManager, deep_find, parse_expiry
from antigravity_acp_bridge.config import ProxyConfig


class AuthHelpersTests(unittest.TestCase):
    def test_parse_expiry_handles_nanoseconds_and_numbers(self) -> None:
        ts = parse_expiry("2026-06-30T12:55:03.123456789Z")
        expected = datetime(2026, 6, 30, 12, 55, 3, 123456, tzinfo=timezone.utc).timestamp()
        self.assertAlmostEqual(ts, expected, places=5)
        self.assertEqual(parse_expiry(1700000000), 1700000000.0)
        self.assertEqual(parse_expiry("not-a-date"), 0.0)
        self.assertEqual(parse_expiry(None), 0.0)

    def test_deep_find_nested_dict_and_list(self) -> None:
        payload = {"outer": [{"inner": {"cloudaicompanionProject": "proj-123"}}]}
        self.assertEqual(deep_find(payload, "cloudaicompanionProject"), "proj-123")
        self.assertIsNone(deep_find(payload, "missing"))


class TokenManagerTests(unittest.TestCase):
    def test_preflight_and_cached_token_return(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime(
            "%Y-%m-%dT%H:%M:%S.000000Z"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = os.path.join(tmpdir, "token.json")
            with open(token_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "auth_method": "oauth",
                        "token": {
                            "access_token": "valid-access",
                            "refresh_token": "valid-refresh",
                            "expiry": future,
                        },
                    },
                    fh,
                )

            cfg = ProxyConfig(token_file=token_path)
            mgr = TokenManager(cfg)
            self.assertTrue(mgr.preflight_check())
            self.assertEqual(mgr.get_access_token(), "valid-access")

    def test_expired_token_triggers_refresh_and_persists(self) -> None:
        past = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime(
            "%Y-%m-%dT%H:%M:%S.000000Z"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = os.path.join(tmpdir, "token.json")
            with open(token_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "token": {
                            "access_token": "stale",
                            "refresh_token": "refresh-1",
                            "expiry": past,
                        }
                    },
                    fh,
                )

            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "access_token": "fresh-access",
                "expires_in": 3600,
                "token_type": "Bearer",
            }).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp

            cfg = ProxyConfig(token_file=token_path)
            mgr = TokenManager(cfg)
            with patch("antigravity_acp_bridge.auth.urlopen", return_value=mock_resp):
                token = mgr.get_access_token()

            self.assertEqual(token, "fresh-access")
            persisted = mgr.read_token_from_disk()
            self.assertEqual(persisted["token"]["access_token"], "fresh-access")
            self.assertEqual(persisted["token"]["refresh_token"], "refresh-1")


class ProjectResolverTests(unittest.TestCase):
    def test_override_bypasses_network_and_caches(self) -> None:
        cfg = ProxyConfig(project_id_override="custom-proj")
        mgr = MagicMock()
        resolver = ProjectResolver(cfg, mgr)
        self.assertEqual(resolver.get_project_id(), "custom-proj")
        mgr.get_access_token.assert_not_called()

    def test_discovers_and_caches_project_id(self) -> None:
        cfg = ProxyConfig()
        mgr = MagicMock()
        mgr.get_access_token.return_value = "tok"
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"nested": {"cloudaicompanionProject": "discovered-proj"}}
        ).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        resolver = ProjectResolver(cfg, mgr)
        with patch("antigravity_acp_bridge.auth.urlopen", return_value=mock_resp) as m_open:
            self.assertEqual(resolver.get_project_id(), "discovered-proj")
            self.assertEqual(resolver.get_project_id(), "discovered-proj")
            self.assertEqual(m_open.call_count, 1)
