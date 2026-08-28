from __future__ import annotations

import os
import unittest

from secretary.capture.lifecycle import ScreenpipeLifecycleManager
from secretary.capture.screenpipe import ScreenpipeCaptureProvider
from secretary.config import SecretaryConfig
from secretary.engine import SecretaryEngine
from secretary.memory.store import MemoryStore
from secretary.notifications.mock import MockNotificationProvider
from secretary.platform.windows.job_object import WindowsJobObject


@unittest.skipUnless(
    os.name == "nt" and os.getenv("SECRETARY_LIVE_TESTS") == "1" and os.getenv("SCREENPIPE_API_KEY"),
    "opt-in: Windows + SECRETARY_LIVE_TESTS=1 + SCREENPIPE_API_KEY required",
)
class WindowsLiveTests(unittest.TestCase):
    def test_existing_external_screenpipe_is_reused_and_left_running(self) -> None:
        config = SecretaryConfig.from_environment()
        provider = ScreenpipeCaptureProvider(config.screenpipe_base_url, config.screenpipe_api_key)
        if not provider.ready():
            self.skipTest("no external Screenpipe already running; refusing to start one for external-instance test")
        manager = ScreenpipeLifecycleManager(provider, mode="external", command=config.screenpipe_command, ready_timeout=1)
        status = manager.start()
        self.assertEqual(status.capture_status, "READY", status.error)
        self.assertFalse(status.owned_by_secretary)
        manager.pause()
        manager.quit()
        self.assertTrue(provider.ready())

    def test_managed_lifecycle_real_screenpipe(self) -> None:
        config = SecretaryConfig.from_environment()
        provider = ScreenpipeCaptureProvider(config.screenpipe_base_url, config.screenpipe_api_key)
        if provider.health():
            self.skipTest("external Screenpipe already running; refusing to control it")
        manager = ScreenpipeLifecycleManager(
            provider,
            mode="managed",
            command=config.screenpipe_command,
            job_factory=WindowsJobObject,
            ready_timeout=45,
        )
        try:
            status = manager.start()
            self.assertEqual(status.capture_status, "READY", status.error)
            self.assertTrue(status.owned_by_secretary)
            self.assertTrue(provider.authenticated_search(limit=1))
            self.assertTrue(provider.audio_disabled())
            # Exercise the real capture -> normalization -> mock-inference ->
            # policy -> SQLite path without persisting raw Screenpipe text.
            engine = SecretaryEngine(
                config,
                store=MemoryStore(":memory:"),
                notifier=MockNotificationProvider(),
            )
            try:
                results = [engine.process(dict(item)) for item in provider.poll()]
                self.assertTrue(all(result.event is None or result.event.app for result in results))
            finally:
                engine.close()
            paused = manager.pause()
            self.assertEqual(paused.capture_status, "PAUSED")
            resumed = manager.resume()
            self.assertEqual(resumed.capture_status, "READY", resumed.error)
            self.assertTrue(resumed.owned_by_secretary)
        finally:
            manager.quit()
        self.assertFalse(provider.health())
