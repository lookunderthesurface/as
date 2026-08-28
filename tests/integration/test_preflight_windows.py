from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path

from secretary.config import SecretaryConfig
from secretary.main import run_preflight


@unittest.skipUnless(os.name == "nt", "Windows-specific preflight assertion")
class WindowsPreflightTests(unittest.TestCase):
    def test_windows_is_reported_as_an_ok_platform_check(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = SecretaryConfig(
                project_root=root,
                database_path=root / "data" / "state.db",
                log_directory=root / "logs",
                capture_provider="mock",
            )
            output = io.StringIO()
            self.assertTrue(run_preflight(config, output))
            self.assertIn("[OK] Windows", output.getvalue())
