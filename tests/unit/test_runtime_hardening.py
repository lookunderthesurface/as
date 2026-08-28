from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from secretary.config import SecretaryConfig
from secretary.context.working_state import WorkingState
from secretary.engine import SecretaryEngine
from secretary.inference.context import InferenceContextBuilder
from secretary.inference.mock import MockInferenceProvider
from secretary.inference.ollama import OllamaInferenceProvider
from secretary.inference.schema import InferenceRequest
from secretary.inference.stale import ResultFreshness, assess_result
from secretary.instance import InstanceLock
from secretary.main import run_benchmark, run_recent_decisions
from secretary.memory.store import MemoryStore
from secretary.paths import AppPaths


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


def raw(app: str, text: str, timestamp: str = "2026-08-28T12:00:00Z", **extra):
    value = {"timestamp": timestamp, "foreground_app": app, "window_title": "PowerShell", "event_source": "command", "text": text}
    value.update(extra)
    return value


class RuntimeHardeningTests(unittest.TestCase):
    def config(self, root: Path, **overrides) -> SecretaryConfig:
        values = dict(project_root=root, database_path=root / "data" / "state.db", log_directory=root / "logs")
        values.update(overrides)
        return SecretaryConfig(**values)

    def test_stale_result_is_discarded_without_policy_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = SecretaryEngine(self.config(root, inference_min_interval_seconds=0, inference_stale_result_seconds=0))
            try:
                _, work = engine.prepare_inference_batch([raw("Code.exe", "editing")])
                assert work is not None
                engine.submit_inference(work)
                unrelated = raw("chrome.exe", "unrelated research", "2026-08-28T12:00:01Z")
                unrelated["window_title"] = "Chrome research"
                unrelated["event_source"] = "navigation"
                engine.note_generation(3, [unrelated])
                result = engine.run_scheduled_inference(work)
                assert result is not None
                self.assertEqual(result.decision.reason_code, "STALE_RESULT")
                self.assertEqual(engine.store.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0], 0)
                self.assertEqual(engine.store.recent_decision_traces(1)[0]["reason_code"], "STALE_RESULT")
                self.assertEqual(engine.counters_snapshot()["inference_results_stale_discarded"], 1)
            finally:
                engine.close()

    def test_slightly_stale_result_is_low_risk_and_does_not_hijack_objective(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = SecretaryEngine(self.config(root, inference_min_interval_seconds=0, inference_stale_result_seconds=0))
            engine.state.current_objective = "review unrelated work"
            try:
                _, work = engine.prepare_inference_batch([raw("WindowsTerminal.exe", "pytest FAILED")])
                assert work is not None
                engine.submit_inference(work)
                engine.note_generation(2, [raw("WindowsTerminal.exe", "pytest still visible", "2026-08-28T12:00:01Z")])
                result = engine.run_scheduled_inference(work)
                assert result is not None
                self.assertEqual(result.decision.reason_code, "SLIGHTLY_STALE_LOW_RISK")
                self.assertEqual(result.decision.action.value, "REMEMBER")
                self.assertEqual(engine.state.current_objective, "review unrelated work")
                self.assertEqual(engine.watch.snapshot(), [])
            finally:
                engine.close()

    def test_mock_does_not_treat_browser_error_documentation_as_failure_and_detects_recovery(self) -> None:
        provider = MockInferenceProvider()
        docs_raw = raw("chrome.exe", "Error handling documentation")
        docs_raw["window_title"] = "Chrome docs"
        docs = provider.analyze(InferenceRequest(current_event=__import__("secretary.events.normalize", fromlist=["normalize_fixture_item"]).normalize_fixture_item(docs_raw)))
        self.assertNotEqual(docs.event.event_type, "failure")
        recovery = provider.analyze(InferenceRequest(current_event=__import__("secretary.events.normalize", fromlist=["normalize_fixture_item"]).normalize_fixture_item(raw("WindowsTerminal.exe", "pytest PASSED after fix; resolved"))))
        self.assertEqual(recovery.event.event_type, "recovery")

    def test_watch_recovery_clears_active_watch_and_failure_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = SecretaryEngine(self.config(root))
            try:
                engine.process(raw("WindowsTerminal.exe", "pytest FAILED", "2026-08-28T12:00:00Z"))
                engine.process(raw("WindowsTerminal.exe", "pytest FAILED", "2026-08-28T12:00:40Z"))
                self.assertIsNotNone(engine.watch.active)
                recovery = engine.process(raw("WindowsTerminal.exe", "pytest PASSED after fix; resolved", "2026-08-28T12:01:20Z"))
                self.assertEqual(recovery.decision.reason_code, "WATCH_RESOLVED")
                self.assertIsNone(engine.watch.active)
                self.assertEqual(engine.state.current_objective, None)
            finally:
                engine.close()

    def test_context_metadata_is_bounded_and_recorded(self) -> None:
        builder = InferenceContextBuilder(500)
        event = __import__("secretary.events.normalize", fromlist=["normalize_fixture_item"]).normalize_fixture_item(raw("Code.exe", "x"))
        request = builder.build(event, recent_events=[{"text": "old" * 1000}], active_hypotheses=[{"id": "watch"}])
        self.assertLessEqual(len(request.context_text), 500)
        self.assertEqual(request.context_chars, len(request.context_text))
        self.assertEqual(request.context_event_count, 1)
        self.assertEqual(request.context_watch_count, 1)

    def test_schema_migration_crash_recovery_retention_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "legacy.db"
            connection = sqlite3.connect(db)
            connection.executescript("CREATE TABLE sessions (id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT); CREATE TABLE events (id INTEGER PRIMARY KEY, timestamp TEXT, source TEXT, app TEXT, event_type TEXT, activity TEXT, summary TEXT, importance REAL, novelty REAL, confidence REAL, failure_signature TEXT); CREATE TABLE decision_traces (id INTEGER PRIMARY KEY, session_id INTEGER, created_at TEXT NOT NULL, event_timestamp TEXT NOT NULL, foreground_app TEXT NOT NULL, event_type TEXT NOT NULL, candidate_action TEXT NOT NULL, candidate_confidence REAL NOT NULL, candidate_importance REAL NOT NULL, interrupt_score REAL NOT NULL, deterministic_evidence INTEGER NOT NULL, watch_id TEXT, watch_evidence INTEGER NOT NULL, policy_action TEXT NOT NULL, final_action TEXT NOT NULL, suppression_reason TEXT, inference_latency_ms REAL, inference_mode TEXT, reason TEXT NOT NULL, summary TEXT NOT NULL, cloud_escalation_candidate INTEGER NOT NULL)")
            connection.execute("INSERT INTO sessions(started_at) VALUES (?)", ("2026-08-27T00:00:00+00:00",))
            connection.commit()
            connection.close()
            store = MemoryStore(db)
            self.assertIn("status", {row[1] for row in store.connection.execute("PRAGMA table_info(sessions)")})
            self.assertEqual(store.connection.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0], "8")
            new_session = store.start_session()
            self.assertEqual(store.connection.execute("SELECT status FROM sessions WHERE id=1").fetchone()[0], "ABORTED")
            store.end_session(new_session, {"counters": {"raw_screenpipe_items": 2}})
            report = store.latest_session_report()
            assert report is not None
            self.assertEqual(report["counters"]["raw_screenpipe_items"], 2)
            store.close()

    def test_instance_lock_is_exclusive_and_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "secretary.lock"
            first, second = InstanceLock(path), InstanceLock(path)
            self.assertTrue(first.acquire())
            self.assertFalse(second.acquire())
            first.release()
            self.assertTrue(second.acquire())
            second.release()

    def test_app_paths_support_explicit_source_and_runtime_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            paths = AppPaths.from_environment(source)
            self.assertEqual(paths.data_dir, source / "data")
            self.assertEqual(paths.prompt_directory, source / "prompts")
            runtime = Path(directory) / "runtime"
            with patch.dict("os.environ", {"SECRETARY_DATA_DIR": str(runtime)}, clear=False):
                configured = AppPaths.from_environment()
            self.assertEqual(configured.runtime_root, runtime)
            self.assertEqual(configured.database_path, runtime / "state.db")

    def test_benchmark_and_recent_decision_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            output = io.StringIO()
            self.assertEqual(run_benchmark(config, output), 0)
            self.assertIn("10 scenarios", output.getvalue())
            engine = SecretaryEngine(config)
            try:
                engine.process(raw("Code.exe", "editing"))
            finally:
                engine.close()
            output = io.StringIO()
            self.assertEqual(run_recent_decisions(config, limit=3, output=output, action="IGNORE"), 0)
            self.assertIn("reason_code=", output.getvalue())

    def test_stale_assessment_uses_fakeable_clock(self) -> None:
        event = __import__("secretary.events.normalize", fromlist=["normalize_fixture_item"]).normalize_fixture_item(raw("Code.exe", "editing"))
        request = InferenceRequest(current_event=event, created_at=NOW, generation_id=1, activity_snapshot=("Code.exe",))
        assessment = assess_result(request, current_generation=2, current_activity=("Code.exe",), stale_seconds=1, now=NOW + timedelta(seconds=1.5))
        self.assertEqual(assessment.freshness, ResultFreshness.SLIGHTLY_STALE)

    def test_ollama_status_recovers_after_transient_provider_failure(self) -> None:
        calls = 0

        def transport(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("temporary")
            return {"message": {"content": json.dumps({
                "event": {"event_type": "activity", "activity": "desktop", "summary": "ok", "importance": 0.1, "novelty": 0.1, "confidence": 0.9},
                "secretary": {"candidate_action": "IGNORE", "interrupt_score": 0.0, "reason": "quiet"},
            })}}

        event = __import__("secretary.events.normalize", fromlist=["normalize_fixture_item"]).normalize_fixture_item(raw("Code.exe", "editing"))
        provider = OllamaInferenceProvider(text_model="test-model", http_post=transport)
        self.assertEqual(provider.analyze(InferenceRequest(event)).error_type, "connection_error")
        self.assertEqual(provider.status().status.value, "DEGRADED")
        self.assertIsNone(provider.analyze(InferenceRequest(event)).error_type)
        self.assertEqual(provider.status().status.value, "READY")
