from __future__ import annotations

import unittest
from unittest.mock import patch

from secretary.config import SecretaryConfig, build_screenpipe_command


class ConfigTests(unittest.TestCase):
    def test_managed_screenpipe_command_uses_verified_privacy_flags(self) -> None:
        command = SecretaryConfig().screenpipe_command

        self.assertIn("--disable-audio", command)
        self.assertIn("--disable-clipboard-capture", command)
        self.assertIn("--ignored-windows", command)
        self.assertEqual(command[-4:], ("--ignored-windows", "1Password", "--ignored-windows", "KeePass"))

    def test_custom_excluded_apps_are_forwarded_to_managed_command(self) -> None:
        command = build_screenpipe_command(("Vault", "Secrets"))

        self.assertEqual(command[-4:], ("--ignored-windows", "Vault", "--ignored-windows", "Secrets"))

    def test_policy_thresholds_and_shadow_mode_are_configurable(self) -> None:
        with patch.dict("os.environ", {
            "POLICY_WATCH_MIN_CONFIDENCE": "0.77",
            "POLICY_NOTIFY_MIN_INTERRUPT_SCORE": "0.91",
            "POLICY_NOTIFY_MIN_WATCH_EVIDENCE": "4",
            "SECRETARY_SHADOW_MODE": "true",
        }, clear=False):
            config = SecretaryConfig.from_environment()
        self.assertEqual(config.model_watch_min_confidence, 0.77)
        self.assertEqual(config.model_notify_min_interrupt_score, 0.91)
        self.assertEqual(config.model_notify_min_watch_evidence, 4)
        self.assertTrue(config.shadow_mode)
