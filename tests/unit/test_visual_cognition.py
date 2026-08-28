from __future__ import annotations

import io
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from secretary.config import SecretaryConfig
from secretary.events.normalize import normalize_fixture_item
from secretary.engine import SecretaryEngine
from secretary.main import run_current_state, run_gui_trajectory
from secretary.memory.store import MemoryStore
from secretary.vision.cognition import VisualCognition
from secretary.vision.keyframe import KEYFRAME_SAME, KEYFRAME_STRUCTURED, KEYFRAME_VISUAL, VisualKeyframeScheduler
from secretary.vision.mock_gui import MockGUIPerceptionProvider
from secretary.vision.perception import GUIPerceptionRequest, render_gui_perception_prompt
from secretary.vision.state import SemanticGUIState, compute_gui_state_delta
from secretary.vision.world import DesktopWorldState, SemanticEvent, SemanticTrajectory


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "gui"


def std_event(text: str, app: str = "WindowsTerminal.exe", window: str = "PowerShell") -> object:
    return normalize_fixture_item({
        "timestamp": NOW.isoformat(),
        "foreground_app": app,
        "window_title": window,
        "event_source": "command",
        "text": text,
    })


def event_at(text: str, timestamp: datetime, app: str = "WindowsTerminal.exe") -> object:
    return normalize_fixture_item({
        "timestamp": timestamp.isoformat(),
        "foreground_app": app,
        "window_title": "PowerShell",
        "event_source": "command",
        "text": text,
    })


class GUIStateTests(unittest.TestCase):
    def test_failed_to_recovered_produces_recovery_delta(self) -> None:
        failed = SemanticGUIState(timestamp=NOW, application="VSCode", activity="debugging", progress="stalled", errors=("pytest FAILED",))
        passed = SemanticGUIState(timestamp=NOW + timedelta(minutes=2), application="VSCode", activity="debugging", progress="recovered")
        delta = compute_gui_state_delta(failed, passed)
        self.assertTrue(delta.recovery)
        self.assertIn("progress", delta.changed_fields)
        self.assertFalse(delta.regression)

    def test_recovered_to_stalled_produces_regression_delta(self) -> None:
        passed = SemanticGUIState(timestamp=NOW, application="VSCode", activity="debugging", progress="recovered")
        failed = SemanticGUIState(timestamp=NOW + timedelta(minutes=2), application="VSCode", activity="debugging", progress="stalled", errors=("boom",))
        delta = compute_gui_state_delta(passed, failed)
        self.assertTrue(delta.regression)
        self.assertIn("errors", delta.changed_fields)

    def test_application_switch_delta(self) -> None:
        vscode = SemanticGUIState(timestamp=NOW, application="VSCode", activity="coding")
        browser = SemanticGUIState(timestamp=NOW + timedelta(minutes=1), application="Chrome.exe", activity="research")
        delta = compute_gui_state_delta(vscode, browser)
        self.assertIn("application", delta.changed_fields)
        self.assertNotIn("progress", delta.changed_fields)

    def test_identical_state_produces_empty_delta(self) -> None:
        first = SemanticGUIState(timestamp=NOW, application="VSCode", activity="coding", progress="running")
        second = SemanticGUIState(timestamp=NOW + timedelta(seconds=10), application="VSCode", activity="coding", progress="running")
        delta = compute_gui_state_delta(first, second)
        self.assertEqual(delta.changed_fields, ())


class KeyframeTests(unittest.TestCase):
    def test_identical_frames_are_same(self) -> None:
        scheduler = VisualKeyframeScheduler()
        first = scheduler.evaluate(std_event("pytest running"), now=NOW)
        # First observation ever = a real keyframe (fresh desktop session).
        second = scheduler.evaluate(std_event("pytest running"), now=NOW + timedelta(seconds=5))
        self.assertEqual(first.level, KEYFRAME_VISUAL)
        self.assertEqual(second.level, KEYFRAME_SAME)

    def test_app_switch_is_visual_keyframe(self) -> None:
        scheduler = VisualKeyframeScheduler()
        scheduler.evaluate(std_event("editing", "Code.exe"), now=NOW)
        decision = scheduler.evaluate(std_event("browsing docs", "Chrome.exe"), now=NOW + timedelta(seconds=5))
        self.assertEqual(decision.level, KEYFRAME_VISUAL)
        self.assertIn("application_changed", decision.reason)

    def test_new_error_is_visual_keyframe(self) -> None:
        scheduler = VisualKeyframeScheduler()
        scheduler.evaluate(std_event("npm install"), now=NOW)
        decision = scheduler.evaluate(std_event("pytest FAILED"), now=NOW + timedelta(seconds=5))
        self.assertEqual(decision.level, KEYFRAME_VISUAL)
        self.assertIn("new_error_observed", decision.reason)

    def test_text_append_alone_is_structured(self) -> None:
        scheduler = VisualKeyframeScheduler()
        scheduler.evaluate(std_event("dotnet build"), now=NOW)
        decision = scheduler.evaluate(std_event("dotnet build 12 modules compiled"), now=NOW + timedelta(seconds=30))
        self.assertEqual(decision.level, KEYFRAME_STRUCTURED)

    def test_visual_content_app_respects_cooldown(self) -> None:
        """Real-shadow regression: visual_content_app must not fire every frame."""
        scheduler = VisualKeyframeScheduler(min_visual_interval_seconds=45.0)
        raw = {
            "timestamp": NOW.isoformat(),
            "foreground_app": "Code.exe",
            "window_title": "attention.py",
            "event_source": "screen",
            "text": "editing attention layout",
        }
        first = scheduler.evaluate(normalize_fixture_item(raw), now=NOW)
        self.assertEqual(first.level, KEYFRAME_VISUAL)
        # Same window, text-only change 10s later: throttled to STRUCTURED.
        throttled = scheduler.evaluate(
            normalize_fixture_item({**raw, "timestamp": (NOW + timedelta(seconds=10)).isoformat(), "text": "editing attention layout v2"}),
            now=NOW + timedelta(seconds=10),
        )
        self.assertEqual(throttled.level, KEYFRAME_STRUCTURED)
        self.assertIn("visual_cooldown_throttled", throttled.reason)
        # After the cooldown elapses, a plain visual-content frame is visual again.
        later = scheduler.evaluate(
            normalize_fixture_item({**raw, "timestamp": (NOW + timedelta(minutes=2)).isoformat()}),
            now=NOW + timedelta(minutes=2),
        )
        self.assertEqual(later.level, KEYFRAME_VISUAL)

    def test_high_priority_signals_bypass_cooldown(self) -> None:
        scheduler = VisualKeyframeScheduler(min_visual_interval_seconds=45.0)
        base = {
            "timestamp": NOW.isoformat(),
            "foreground_app": "Code.exe",
            "window_title": "attention.py",
            "event_source": "screen",
            "text": "editing",
        }
        scheduler.evaluate(normalize_fixture_item(base), now=NOW)
        error_soon = scheduler.evaluate(
            normalize_fixture_item({**base, "timestamp": (NOW + timedelta(seconds=5)).isoformat(), "text": "pytest FAILED"}),
            now=NOW + timedelta(seconds=5),
        )
        self.assertEqual(error_soon.level, KEYFRAME_VISUAL)
        self.assertIn("new_error_observed", error_soon.reason)

    def test_visual_required_is_always_visual(self) -> None:
        scheduler = VisualKeyframeScheduler()
        raw = {
            "timestamp": NOW.isoformat(),
            "foreground_app": "Code.exe",
            "window_title": "UI layout editor",
            "event_source": "screen",
            "text": "inspect layout",
            "visual_required": True,
        }
        decision = scheduler.evaluate(normalize_fixture_item(raw), now=NOW)
        self.assertEqual(decision.level, KEYFRAME_VISUAL)

    def test_periodic_refresh_keeps_perception_alive(self) -> None:
        scheduler = VisualKeyframeScheduler(forced_visual_interval_seconds=120, min_visual_interval_seconds=60)
        scheduler.evaluate(std_event("work in progress"), now=NOW)
        decision = scheduler.evaluate(std_event("work still in progress"), now=NOW + timedelta(minutes=5))
        self.assertEqual(decision.level, KEYFRAME_VISUAL)
        self.assertIn("periodic", decision.reason)


class TrajectoryTests(unittest.TestCase):
    def test_same_activity_merges_into_one_step(self) -> None:
        trajectory = SemanticTrajectory(merge_after_seconds=300)
        trajectory.append(SemanticEvent(NOW, "editing: test failed", activity="debugging"))
        trajectory.append(SemanticEvent(NOW + timedelta(minutes=2), "editing: test failed again", activity="debugging"))
        self.assertEqual(len(trajectory.snapshot()), 1)
        self.assertEqual(trajectory.snapshot()[-1].label, "editing: test failed again")

    def test_different_activity_keeps_separate_steps(self) -> None:
        trajectory = SemanticTrajectory(merge_after_seconds=300)
        trajectory.append(SemanticEvent(NOW, "editing attention.py", activity="coding", application="VSCode"))
        trajectory.append(SemanticEvent(NOW + timedelta(minutes=2), "documentation search", activity="research", application="Chrome.exe"))
        self.assertEqual(len(trajectory.snapshot()), 2)


class CognitionIntegrationTests(unittest.TestCase):
    def test_failed_to_passed_updates_world_and_recovers(self) -> None:
        world = DesktopWorldState()
        cognition = VisualCognition(MockGUIPerceptionProvider(), world=world)
        failed = std_event("pytest FAILED with AssertionError")
        cognition.on_accepted_event(failed, NOW)
        self.assertEqual(world.current_gui.progress, "stalled")
        self.assertEqual(world.current_gui.errors, ("pytest FAILED with AssertionError",))
        passed = std_event("pytest all tests passed", "")
        passed = normalize_fixture_item({
            "timestamp": (NOW + timedelta(minutes=2)).isoformat(),
            "foreground_app": "WindowsTerminal.exe",
            "window_title": "PowerShell",
            "event_source": "command",
            "text": "pytest all tests passed",
        })
        cognition.on_accepted_event(passed, NOW + timedelta(minutes=2))
        self.assertEqual(world.current_gui.progress, "recovered")
        self.assertTrue(world.last_delta.recovery)

    def test_excluded_app_never_reaches_perception(self) -> None:
        class RaisedPerception(MockGUIPerceptionProvider):
            def perceive(self, request):
                raise AssertionError("excluded app must never be perceived")

        cognition = VisualCognition(RaisedPerception(), excluded_apps=("1Password",))
        event = normalize_fixture_item({
            "timestamp": NOW.isoformat(),
            "foreground_app": "1Password",
            "window_title": "Vault",
            "event_source": "screen",
            "text": "master password",
            "image_path": str(FIXTURES / "vscode_failed_terminal.ppm"),
        })
        update = cognition.on_accepted_event(event, NOW, skip_vision=False)
        self.assertTrue(update.is_same)
        self.assertIsNone(cognition.current_gui_state)

    def test_same_frame_skips_structured_update(self) -> None:
        world = DesktopWorldState()
        cognition = VisualCognition(MockGUIPerceptionProvider(), world=world)
        first = std_event("pytest running")
        cognition.on_accepted_event(first, NOW)
        second = std_event("pytest running")
        update = cognition.on_accepted_event(second, NOW + timedelta(seconds=5))
        self.assertTrue(update.is_same)
        self.assertEqual(len(world.trajectory.snapshot()), 1)

    def test_vision_keyframe_calls_perception_with_previous_state(self) -> None:
        captured: list[GUIPerceptionRequest] = []

        class CapturingPerception(MockGUIPerceptionProvider):
            def perceive(self, request):
                captured.append(request)
                return super().perceive(request)

        world = DesktopWorldState()
        cognition = VisualCognition(CapturingPerception(), world=world)
        first = normalize_fixture_item({
            "timestamp": NOW.isoformat(),
            "foreground_app": "Code.exe",
            "window_title": "attention.py",
            "event_source": "screen",
            "text": "editing",
            "image_path": str(FIXTURES / "vscode_failed_terminal.ppm"),
        })
        cognition.on_accepted_event(first, NOW, skip_vision=False)
        second = normalize_fixture_item({
            "timestamp": (NOW + timedelta(minutes=1)).isoformat(),
            "foreground_app": "WindowsTerminal.exe",
            "window_title": "PowerShell",
            "event_source": "command",
            "text": "pytest FAILED",
            "image_path": str(FIXTURES / "vscode_failed_terminal.ppm"),
        })
        cognition.on_accepted_event(second, NOW + timedelta(minutes=1), skip_vision=False)
        self.assertEqual(len(captured), 2)
        self.assertIsNotNone(captured[1].previous_state)
        self.assertIn("CURRENT MOMENT", render_gui_perception_prompt(captured[1]))

    def test_skip_vision_still_keeps_structured_state(self) -> None:
        world = DesktopWorldState()
        cognition = VisualCognition(MockGUIPerceptionProvider(), world=world)
        failed = std_event("pytest FAILED")
        update = cognition.on_accepted_event(failed, NOW, skip_vision=True)
        self.assertFalse(update.used_vision)
        self.assertEqual(world.current_gui.progress, "stalled")

    def test_negated_words_do_not_fabricate_recovery(self) -> None:
        world = DesktopWorldState()
        cognition = VisualCognition(MockGUIPerceptionProvider(), world=world)
        dodgy = std_event("download unsuccessful, retrying...")
        cognition.on_accepted_event(dodgy, NOW, skip_vision=True)
        self.assertEqual(world.current_gui.progress, "running")
        self.assertEqual(world.current_gui.errors, ())

    def test_zero_error_line_is_not_a_failure(self) -> None:
        from secretary.vision.structured import structured_gui_state
        from secretary.vision.keyframe import KeyframeDecision

        event = std_event("0 errors, build ok")
        state = structured_gui_state(event, KeyframeDecision(KEYFRAME_SAME, "test"))
        self.assertEqual(state.errors, ())
        self.assertNotEqual(state.progress, "stalled")

    def test_failed_perception_is_reported_without_crashing(self) -> None:
        class BrokenPerception(MockGUIPerceptionProvider):
            def perceive(self, request):
                from secretary.vision.perception import GUIPerceptionOutput
                return GUIPerceptionOutput.safe(provider=self.name, error_type="transport_error")

        world = DesktopWorldState()
        cognition = VisualCognition(BrokenPerception(), world=world)
        event = normalize_fixture_item({
            "timestamp": NOW.isoformat(),
            "foreground_app": "Code.exe",
            "window_title": "attention.py",
            "event_source": "screen",
            "text": "editing",
            "image_path": str(FIXTURES / "vscode_failed_terminal.ppm"),
        })
        update = cognition.on_accepted_event(event, NOW, skip_vision=False)
        self.assertTrue(update.used_vision)
        self.assertTrue(update.perception_failed)
        self.assertIsNone(world.current_gui)


class StorePersistenceTests(unittest.TestCase):
    def test_gui_states_roundtrip_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            state = SemanticGUIState(
                timestamp=NOW,
                application="VSCode",
                window="attention.py",
                activity="debugging",
                topic="attention layout",
                progress="stalled",
                errors=("pytest FAILED",),
                confidence=0.9,
                perception_mode="vision",
                keyframe_reason="new_error_observed",
            )
            store.record_gui_state(
                session_id=None,
                event_timestamp=NOW,
                application=state.application,
                window=state.window,
                activity=state.activity,
                topic=state.topic,
                progress=state.progress,
                task_hint=state.task_hint,
                errors=state.errors,
                confidence=state.confidence,
                perception_mode=state.perception_mode,
                keyframe_reason=state.keyframe_reason,
                changed_fields=("progress",),
                recovery=False,
                regression=False,
                trajectory_label="debugging: stalled",
                perception_latency_ms=None,
                generation_id=3,
            )
            latest = store.latest_gui_state()
            self.assertIsNotNone(latest)
            if latest is not None:
                self.assertEqual(latest["progress"], "stalled")
                self.assertEqual(latest["errors"], ["pytest FAILED"])
                self.assertEqual(latest["changed_fields"], ["progress"])
            stats = store.gui_perception_stats()
            self.assertEqual(stats["gui_states"], 1)
            self.assertEqual(stats["vision_perceptions"], 1)
            store.close()

    def test_empty_gui_stats_are_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            stats = store.gui_perception_stats()
            self.assertEqual(stats["gui_states"], 0)
            self.assertEqual(stats["latency"]["count"], 0)
            store.close()

    def test_trajectory_events_roundtrip_filters_by_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            store.record_gui_trajectory_event(
                session_id=None,
                event_timestamp=NOW,
                label="debugging: stalled",
                activity="debugging",
                application="WindowsTerminal.exe",
                topic="attention layout",
                importance=1.0,
            )
            store.record_gui_trajectory_event(
                session_id=None,
                event_timestamp=NOW + timedelta(minutes=10),
                label="debugging: recovered",
                activity="debugging",
                application="WindowsTerminal.exe",
                topic="attention layout",
                importance=1.0,
            )
            events = store.recent_trajectory_events()
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]["label"], "debugging: stalled")
            # Idempotent re-persist of the same event is ignored by the unique index.
            self.assertEqual(store.record_gui_trajectory_event(
                session_id=None,
                event_timestamp=NOW,
                label="debugging: stalled",
                activity="debugging",
                application="WindowsTerminal.exe",
                topic="attention layout",
                importance=1.0,
            ), 0)
            self.assertEqual(len(store.recent_trajectory_events()), 2)
            store.close()


class DiagnosisCLITests(unittest.TestCase):
    def test_current_state_and_trajectory_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SecretaryConfig(
                project_root=root,
                database_path=root / "data" / "state.db",
                log_directory=root / "logs",
            )
            store = MemoryStore(config.database_path)
            store.record_gui_state(
                session_id=None,
                event_timestamp=NOW,
                application="VSCode",
                window="attention.py",
                activity="debugging",
                topic="attention layout",
                progress="stalled",
                task_hint="fix failing work",
                errors=("pytest FAILED",),
                confidence=0.9,
                perception_mode="vision",
                keyframe_reason="new_error_observed",
                changed_fields=("progress",),
                recovery=False,
                regression=False,
                trajectory_label="debugging: stalled",
            )
            store.record_gui_trajectory_event(
                session_id=None,
                event_timestamp=NOW,
                label="debugging: stalled",
                activity="debugging",
                application="WindowsTerminal.exe",
                topic="attention layout",
                importance=1.0,
            )
            store.close()
            output = io.StringIO()
            self.assertEqual(run_current_state(config, output), 0)
            text = output.getvalue()
            self.assertIn("VSCode", text)
            self.assertIn("debugging", text)
            self.assertIn("stalled", text)
            self.assertNotIn("screenshot", text.lower())
            output = io.StringIO()
            self.assertEqual(run_gui_trajectory(config, minutes=60, output=output), 0)
            self.assertIn("debugging: stalled", output.getvalue())


class EngineGUIRecoveryTests(unittest.TestCase):
    """Success criterion: FAILED -> fix -> PASSED must resolve the WATCH."""

    def make_config(self, root: Path) -> SecretaryConfig:
        return SecretaryConfig(
            project_root=root,
            database_path=root / "data" / "state.db",
            log_directory=root / "logs",
            inference_provider="mock",
            cloud_provider="mock",
        )

    def test_repeated_failure_then_recovery_resolves_watch_with_gui_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = SecretaryEngine(self.make_config(root))
            try:
                base = "2026-08-28T12:0{}:00Z"
                # Failure group builds an active WATCH.
                for minute in range(0, 4):
                    engine.process({
                        "timestamp": base.format(minute),
                        "foreground_app": "WindowsTerminal.exe",
                        "window_title": "PowerShell",
                        "event_source": "command",
                        "text": "pytest FAILED",
                    })
                self.assertIsNotNone(engine.watch.active)
                active_signature = engine.watch.active.signature
                # User fixes the code; the terminal now reports success.
                recovery = engine.process({
                    "timestamp": "2026-08-28T12:05:00Z",
                    "foreground_app": "WindowsTerminal.exe",
                    "window_title": "PowerShell",
                    "event_source": "command",
                    "text": "pytest all tests passed",
                    "visual_required": True,
                    "image_path": str(FIXTURES / "vscode_passed_terminal.ppm"),
                })
                self.assertEqual(engine.watch.active, None)
                self.assertEqual(recovery.decision.reason_code, "WATCH_RESOLVED")
                self.assertEqual(engine.watch.resolved_at(active_signature) is not None, True)
                gui_history = engine.store.recent_gui_states(10)
                # The final state plus enough transitions to see the recovery.
                self.assertTrue(any(item["delta_recovery"] for item in gui_history))
                self.assertTrue(any(item["perception_mode"] == "vision" for item in gui_history))
            finally:
                engine.close()

    def test_cross_app_debugging_is_understood_as_one_task(self) -> None:
        """Success criterion 4: VSCode -> terminal -> browser -> VSCode = one task."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = SecretaryEngine(self.make_config(root))
            try:
                steps = [
                    ("2026-08-28T14:00:00Z", "WindowsTerminal.exe", "PowerShell", "pytest FAILED"),
                    ("2026-08-28T14:05:00Z", "Code.exe", "attention.py", "editing attention layout"),
                    ("2026-08-28T14:08:00Z", "Chrome.exe", "PyTorch docs", "reading non-determinism docs"),
                    ("2026-08-28T14:12:00Z", "WindowsTerminal.exe", "PowerShell", "pytest FAILED again"),
                ]
                for timestamp, app, window, text in steps:
                    engine.process({
                        "timestamp": timestamp,
                        "foreground_app": app,
                        "window_title": window,
                        "event_source": "command",
                        "text": text,
                    })
                labels = [item.label for item in engine.cognition.world.trajectory.snapshot()]
                self.assertTrue(any("failed" in label or "stalled" in label for label in labels))
                self.assertTrue(any("research" in label or "documentation" in label or "observed" in label for label in labels))
                # The activity of the persisted final GUI state still names the task.
                state = engine.cognition.current_gui_state
                self.assertIsNotNone(state)
                self.assertEqual(str(state.activity), "terminal")
                self.assertEqual(str(state.progress), "stalled")
            finally:
                engine.close()


if __name__ == "__main__":
    unittest.main()
