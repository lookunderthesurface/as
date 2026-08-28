from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from secretary.config import SecretaryConfig
from secretary.engine import SecretaryEngine
from secretary.inference.schema import InferenceEvent, InferenceResult, SecretaryAssessment


class RecordingProvider:
    name = "recording"
    model = None

    def __init__(self) -> None:
        self.requests = []

    def analyze(self, request):
        self.requests.append(request)
        return InferenceResult(
            event=InferenceEvent(event_type="activity", activity="desktop", summary="ordered activity", confidence=0.9),
            secretary=SecretaryAssessment(),
            provider=self.name,
        )


def raw(timestamp: str, frame_id: int) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "frame_id": frame_id,
        "foreground_app": "Code.exe",
        "window_title": f"frame {frame_id}",
        "event_source": "test",
        "text": f"frame {frame_id}",
    }


class EventOrderingTests(unittest.TestCase):
    def run_batch(self, items):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = RecordingProvider()
            engine = SecretaryEngine(
                SecretaryConfig(project_root=root, database_path=root / "state.db", log_directory=root / "logs"),
                inference_provider=provider,
            )
            try:
                engine.process_coalesced(items)
                self.assertEqual(len(provider.requests), 1)
                return provider.requests[0]
            finally:
                engine.close()

    def assert_ordered(self, items) -> None:
        request = self.run_batch(items)
        timestamps = [event["timestamp"] for event in request.recent_events]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(request.current_event.timestamp.isoformat(), max(timestamps))

    def test_ascending_input(self) -> None:
        self.assert_ordered([
            raw("2026-08-28T12:00:01Z", 1),
            raw("2026-08-28T12:00:03Z", 2),
            raw("2026-08-28T12:00:05Z", 3),
            raw("2026-08-28T12:00:09Z", 4),
        ])

    def test_descending_input(self) -> None:
        self.assert_ordered([
            raw("2026-08-28T12:00:09Z", 4),
            raw("2026-08-28T12:00:05Z", 3),
            raw("2026-08-28T12:00:03Z", 2),
            raw("2026-08-28T12:00:01Z", 1),
        ])

    def test_mixed_input(self) -> None:
        request = self.run_batch([
            raw("2026-08-28T12:00:05Z", 3),
            raw("2026-08-28T12:00:01Z", 1),
            raw("2026-08-28T12:00:09Z", 4),
            raw("2026-08-28T12:00:03Z", 2),
        ])
        self.assertEqual(request.current_event.timestamp.isoformat(), "2026-08-28T12:00:09+00:00")
        self.assertEqual([item["timestamp"] for item in request.recent_events], [
            "2026-08-28T12:00:01+00:00",
            "2026-08-28T12:00:03+00:00",
            "2026-08-28T12:00:05+00:00",
            "2026-08-28T12:00:09+00:00",
        ])

    def test_same_timestamp_has_deterministic_tie_breaker(self) -> None:
        request = self.run_batch([
            raw("2026-08-28T12:00:05Z", 2),
            raw("2026-08-28T12:00:05Z", 1),
        ])
        self.assertEqual([item["timestamp"] for item in request.recent_events], [
            "2026-08-28T12:00:05+00:00",
            "2026-08-28T12:00:05+00:00",
        ])
        self.assertEqual(request.current_event.frame_id, 2)
