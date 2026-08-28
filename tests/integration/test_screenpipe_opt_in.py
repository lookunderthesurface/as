from __future__ import annotations

import os
import unittest

from secretary.capture.screenpipe import ScreenpipeCaptureProvider


@unittest.skipUnless(os.getenv("SCREENPIPE_API_KEY"), "opt-in: set SCREENPIPE_API_KEY to run Screenpipe integration tests")
class ScreenpipeIntegrationTests(unittest.TestCase):
    def test_health_and_authenticated_search(self) -> None:
        provider = ScreenpipeCaptureProvider(api_key=os.environ["SCREENPIPE_API_KEY"])
        self.assertTrue(provider.health())
        self.assertTrue(provider.authenticated_search(limit=1))
        self.assertTrue(provider.audio_disabled())

