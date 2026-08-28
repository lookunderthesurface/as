from __future__ import annotations

import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from secretary.config import SecretaryConfig
from secretary.engine import SecretaryEngine
from secretary.events.normalize import normalize_fixture_item
from secretary.inference.mock import MockInferenceProvider
from secretary.main import build_parser, run_live, run_preflight, run_recent_decisions, run_replay, run_session_report
from secretary.notifications.mock import MockNotificationProvider
from secretary.notifications.shadow import ShadowNotificationProvider


def event(timestamp: str, app: str, text: str, window: str = "") -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "foreground_app": app,
        "window_title": window,
        "event_source": "test",
        "text": text,
    }


class PipelineTests(unittest.TestCase):
    def make_engine(self) -> tuple[SecretaryEngine, MockNotificationProvider, tempfile.TemporaryDirectory[str]]:
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        config = SecretaryConfig(
            project_root=root,
            database_path=root / "data" / "state.db",
            log_directory=root / "logs",
            excluded_apps=("1Password", "KeePass"),
        )
        notifier = MockNotificationProvider()
        return SecretaryEngine(config, notifier=notifier), notifier, temp

    def test_repeated_failure_reaches_notification(self) -> None:
        engine, notifier, temp = self.make_engine()
        try:
            results = [
                engine.process(event("2026-08-28T10:00:00Z", "WindowsTerminal.exe", "pytest FAILED")),
                engine.process(event("2026-08-28T10:00:40Z", "WindowsTerminal.exe", "pytest FAILED")),
                engine.process(event("2026-08-28T10:01:20Z", "chrome.exe", "pytest documentation", "pytest docs")),
                engine.process(event("2026-08-28T10:02:00Z", "WindowsTerminal.exe", "pytest FAILED")),
                engine.process(event("2026-08-28T10:02:40Z", "WindowsTerminal.exe", "pytest FAILED")),
            ]
            self.assertEqual([r.decision.action.value for r in results], ["REMEMBER", "WATCH", "WATCH", "INVESTIGATE", "NOTIFY"])
            self.assertEqual(len(notifier.notifications), 1)
            self.assertEqual(engine.store.count_failures("test-failure:python"), 4)
        finally:
            engine.close()
            temp.cleanup()

    def test_screenpipe_fixture_is_normalized_at_the_boundary(self) -> None:
        fixture_path = Path(__file__).parents[1] / "fixtures" / "screenpipe_terminal.json"
        raw = json.loads(fixture_path.read_text(encoding="utf-8"))
        normalized = normalize_fixture_item(raw)
        self.assertEqual(normalized.source, "screenpipe")
        self.assertEqual(normalized.foreground_app, "WindowsTerminal.exe")
        self.assertEqual(normalized.frame_id, 1001)
        self.assertEqual(normalized.text, "python -m unittest")

    def test_flattened_screenpipe_projection_is_normalized_at_the_boundary(self) -> None:
        normalized = normalize_fixture_item({
            "type": "OCR",
            "content.app_name": "WindowsTerminal.exe",
            "content.window_name": "PowerShell",
            "content.text": "pytest FAILED",
            "content.event_source": "screen",
            "content.frame_id": 1002,
            "content.timestamp": "2026-08-28T08:00:01Z",
        })
        self.assertEqual(normalized.foreground_app, "WindowsTerminal.exe")
        self.assertEqual(normalized.text, "pytest FAILED")
        self.assertEqual(normalized.event_source, "screen")
        self.assertEqual(normalized.frame_id, 1002)

    def test_normal_coding_does_not_notify(self) -> None:
        engine, notifier, temp = self.make_engine()
        try:
            result = engine.process(event("2026-08-28T10:00:00Z", "Code.exe", "editing secretary"))
            self.assertEqual(result.decision.action.value, "IGNORE")
            self.assertEqual(notifier.notifications, [])
        finally:
            engine.close()
            temp.cleanup()

    def test_local_inference_is_the_deterministic_mock_boundary(self) -> None:
        engine, _, temp = self.make_engine()
        try:
            self.assertIsInstance(engine.inference, MockInferenceProvider)
            self.assertIs(engine.extractor.inference, engine.inference)
            self.assertEqual(MockInferenceProvider.ACTION_SPACE[-1], "NOTIFY")
        finally:
            engine.close()
            temp.cleanup()

    def test_real_capture_failure_is_degraded_not_fake_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = SecretaryConfig(
                project_root=root,
                database_path=root / "data" / "state.db",
                log_directory=root / "logs",
                screenpipe_api_key=None,
                screenpipe_mode="managed",
                capture_provider="screenpipe",
            )
            args = build_parser().parse_args(["run", "--once"])
            output = io.StringIO()
            import contextlib
            with contextlib.redirect_stdout(output):
                result = run_live(config, args)
            self.assertEqual(result, 2)
            self.assertIn("capture_status=DEGRADED", output.getvalue())
            self.assertNotIn("MockCaptureProvider", output.getvalue())

    def test_privacy_event_is_not_persisted_or_model_processed(self) -> None:
        engine, notifier, temp = self.make_engine()
        try:
            result = engine.process(event("2026-08-28T10:00:00Z", "KeePass.exe", "password=DO_NOT_STORE"))
            self.assertTrue(result.privacy_suppressed)
            self.assertEqual(engine.store.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0], 0)
            self.assertEqual(engine.store.connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 0)
            log_text = (Path(temp.name) / "logs" / "secretary.log").read_text(encoding="utf-8")
            self.assertNotIn("DO_NOT_STORE", log_text)
            self.assertEqual(notifier.notifications, [])
        finally:
            engine.close()
            temp.cleanup()

    def test_cross_app_context_is_retained_without_raw_text(self) -> None:
        engine, _, temp = self.make_engine()
        try:
            engine.process(event("2026-08-28T10:00:00Z", "Code.exe", "editing"))
            engine.process(event("2026-08-28T10:00:20Z", "WindowsTerminal.exe", "python -m unittest"))
            engine.process(event("2026-08-28T10:00:40Z", "chrome.exe", "GitHub discussion"))
            snapshot = engine.state.snapshot()
            self.assertEqual(snapshot["active_apps"], ["Code.exe", "WindowsTerminal.exe", "chrome.exe"])
            self.assertEqual(engine.store.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0], 3)
            self.assertNotIn("GitHub discussion", json.dumps(snapshot))
        finally:
            engine.close()
            temp.cleanup()

    def test_pause_stops_processing(self) -> None:
        engine, _, temp = self.make_engine()
        try:
            engine.pause()
            result = engine.process(event("2026-08-28T10:00:00Z", "Code.exe", "editing"))
            self.assertEqual(result.decision.action.value, "IGNORE")
            self.assertEqual(engine.store.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0], 0)
            engine.resume()
            engine.process(event("2026-08-28T10:00:01Z", "Code.exe", "editing"))
            self.assertEqual(engine.store.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0], 1)
        finally:
            engine.close()
            temp.cleanup()

    def test_replay_and_preflight_helpers(self) -> None:
        output = io.StringIO()
        preflight_config = SecretaryConfig.from_environment(Path(tempfile.gettempdir()))
        preflight_config.screenpipe_api_key = "test-only"
        preflight_config.capture_provider = "mock"
        self.assertTrue(run_preflight(preflight_config, output))
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            scenario = temp / "scenario.jsonl"
            scenario.write_text((Path(__file__).parents[2] / "scenarios" / "repeated_failure.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
            config = SecretaryConfig(project_root=temp, database_path=temp / "data" / "state.db", log_directory=temp / "logs")
            replay_output = io.StringIO()
            import contextlib
            with contextlib.redirect_stdout(replay_output):
                self.assertEqual(run_replay(scenario, config), 0)
            self.assertIn("NOTIFY", replay_output.getvalue())

    def test_shadow_records_would_notify_without_delivery(self) -> None:
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        config = SecretaryConfig(project_root=root, database_path=root / "data" / "state.db", log_directory=root / "logs")
        notifier = ShadowNotificationProvider()
        engine = SecretaryEngine(config, notifier=notifier)
        try:
            for index, timestamp in enumerate(("10:00:00", "10:00:40", "10:01:20", "10:02:00", "10:02:40")):
                engine.process(event(f"2026-08-28T{timestamp}Z", "WindowsTerminal.exe", "pytest FAILED"))
            self.assertEqual(len(notifier.notifications), 1)
            trace = engine.store.recent_decision_traces(1)[0]
            self.assertEqual(trace["policy_action"], "NOTIFY")
            self.assertEqual(trace["final_action"], "WOULD_NOTIFY")
            self.assertEqual(engine.store.connection.execute("SELECT COUNT(*) FROM notifications").fetchone()[0], 1)
        finally:
            engine.close()
            report_output = io.StringIO()
            self.assertEqual(run_session_report(config, report_output), 0)
            self.assertIn("WOULD_NOTIFY", report_output.getvalue())
            recent_output = io.StringIO()
            self.assertEqual(run_recent_decisions(config, limit=1, output=recent_output), 0)
            self.assertIn("Final: WOULD_NOTIFY", recent_output.getvalue())
            temp.cleanup()

    def test_shadow_mode_refuses_mock_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = SecretaryConfig(project_root=root, database_path=root / "state.db", log_directory=root / "logs")
            args = build_parser().parse_args(["run", "--shadow", "--once"])
            output = io.StringIO()
            import contextlib
            with contextlib.redirect_stdout(output):
                result = run_live(config, args)
            self.assertEqual(result, 2)
            self.assertIn("requires INFERENCE_PROVIDER=ollama", output.getvalue())
