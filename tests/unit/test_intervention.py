from __future__ import annotations

import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from secretary.config import SecretaryConfig
from secretary.context.working_state import WorkingState
from secretary.engine import SecretaryEngine
from secretary.inference.schema import Action
from secretary.main import build_parser, run_feedback, run_recent_interventions, run_secretary_profile
from secretary.memory.intervention import PreferenceKind
from secretary.memory.retrieval import retrieve_relevant_intervention_preferences
from secretary.memory.store import MemoryStore
from secretary.events.schema import sanitize_failure_signature, sanitize_semantic_label
from secretary.perception.extractor import ExtractedEvent
from secretary.policy.hard_rules import HardRules
from secretary.policy.proactive import PolicyThresholds, ProactivePolicy
from secretary.policy.watch import WatchManager


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


def raw(timestamp: str, text: str, app: str = "WindowsTerminal.exe") -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "foreground_app": app,
        "window_title": "PowerShell",
        "event_source": "command",
        "text": text,
    }


def extracted(*, action: Action = Action.NOTIFY, failure: str | None = "test-failure:python") -> ExtractedEvent:
    return ExtractedEvent(
        timestamp=NOW,
        event_type="failure" if failure else "research",
        activity="terminal" if failure else "research",
        app="WindowsTerminal.exe" if failure else "Chrome.exe",
        summary="A bounded semantic summary",
        importance=0.9,
        novelty=0.5,
        confidence=0.95,
        failure_signature=failure,
        topic=failure or "pytest",
        candidate_action=action,
        interrupt_score=0.9,
    )


class InterventionTests(unittest.TestCase):
    def make_config(self, root: Path, **overrides: object) -> SecretaryConfig:
        values: dict[str, object] = {
            "project_root": root,
            "database_path": root / "data" / "state.db",
            "log_directory": root / "logs",
        }
        values.update(overrides)
        return SecretaryConfig(**values)

    def test_engine_records_episode_and_explicit_feedback_as_preference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = SecretaryEngine(self.make_config(root))
            try:
                result = engine.process(raw("2026-08-28T12:00:00Z", "pytest FAILED"))
                self.assertEqual(result.decision.action, Action.REMEMBER)
                episode = engine.store.recent_intervention_episodes(1)[0]
                feedback = engine.store.record_intervention_feedback(int(episode["id"]), "dont-remind")
                self.assertEqual(feedback["reaction"], "EXPLICIT_NEGATIVE")
                self.assertEqual(engine.store.get_intervention_episode(int(episode["id"]))["explicit_feedback"], "DONT_REMIND")
                preferences = engine.store.active_intervention_preferences()
                self.assertEqual(len(preferences), 1)
                self.assertEqual(preferences[0]["source"], "EXPLICIT_USER")
                self.assertEqual(preferences[0]["preference"], PreferenceKind.AVOID_ISOLATED.value)
                next_result = engine.process(raw("2026-08-28T12:00:40Z", "pytest FAILED"))
                self.assertEqual(next_result.decision.action, Action.WATCH)
                trace = engine.store.recent_decision_traces(1)[0]
                self.assertEqual(json.loads(trace["preference_ids"]), [int(preferences[0]["id"])])
                self.assertEqual(trace["preference_effect"], "USER_PREFERS_SILENCE_FOR_ISOLATED_ERROR")
            finally:
                engine.close()

    def test_preference_update_supersedes_previous_preference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            first = store.record_intervention_episode(
                session_id=None,
                event_timestamp=NOW,
                situation_type="debugging",
                activity="terminal",
                event_type="failure",
                topic="test-failure:python",
                failure_signature="test-failure:python",
                candidate_action="WATCH",
                final_action="NOTIFY",
            )
            store.record_intervention_feedback(first, "dont-remind")
            second = store.record_intervention_episode(
                session_id=None,
                event_timestamp=NOW,
                situation_type="debugging",
                activity="terminal",
                event_type="failure",
                topic="test-failure:python",
                failure_signature="test-failure:python",
                candidate_action="WATCH",
                final_action="INVESTIGATE",
            )
            store.record_intervention_feedback(second, "more-proactive")
            rows = store.connection.execute(
                "SELECT preference, status, supersedes_id FROM intervention_preferences ORDER BY id"
            ).fetchall()
            self.assertEqual([(row[0], row[1]) for row in rows], [
                (PreferenceKind.AVOID_ISOLATED.value, "SUPERSEDED"),
                (PreferenceKind.MORE_PROACTIVE.value, "ACTIVE"),
            ])
            self.assertEqual(rows[1][2], 1)
            store.close()

    def test_untrusted_failure_signature_is_opaque_in_persisted_preference(self) -> None:
        secret = "password=TOP_SECRET api_key=do-not-store"
        self.assertNotIn("TOP_SECRET", str(sanitize_failure_signature(secret)))
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            episode_id = store.record_intervention_episode(
                session_id=None,
                event_timestamp=NOW,
                situation_type="debugging",
                activity="terminal",
                event_type="failure",
                topic="general",
                failure_signature=secret,
                candidate_action="WATCH",
                final_action="WATCH",
            )
            store.record_intervention_feedback(episode_id, "dont-remind")
            persisted_episode = store.get_intervention_episode(episode_id)
            self.assertNotIn("TOP_SECRET", str(persisted_episode["failure_signature"]))
            preference = store.active_intervention_preferences()[0]
            self.assertNotIn("TOP_SECRET", json.dumps(preference))
            self.assertNotIn("TOP_SECRET", str(preference["content"]))
            store.close()

    def test_forget_disables_scoped_preference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            episode_id = store.record_intervention_episode(
                session_id=None,
                event_timestamp=NOW,
                situation_type="debugging",
                activity="terminal",
                event_type="failure",
                topic="test-failure:python",
                failure_signature="test-failure:python",
                candidate_action="WATCH",
                final_action="WATCH",
            )
            store.record_intervention_feedback(episode_id, "dont-remind")
            store.record_intervention_feedback(episode_id, "forget")
            self.assertEqual(store.active_intervention_preferences(), [])
            status = store.connection.execute("SELECT status FROM intervention_preferences").fetchone()[0]
            self.assertEqual(status, "DISABLED")
            store.close()

    def test_relevant_preferences_are_retrieved_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            episode_id = store.record_intervention_episode(
                session_id=None,
                event_timestamp=NOW,
                situation_type="debugging",
                activity="terminal",
                event_type="failure",
                topic="test-failure:python",
                failure_signature="test-failure:python",
                candidate_action="WATCH",
                final_action="WATCH",
            )
            store.record_intervention_feedback(episode_id, "more-proactive")
            matches = retrieve_relevant_intervention_preferences(store, extracted(action=Action.WATCH))
            self.assertEqual([item["preference"] for item in matches], [PreferenceKind.MORE_PROACTIVE.value])
            self.assertGreater(int(matches[0]["match_score"]), 0)
            store.close()

    def test_preference_does_not_cross_match_another_failure_signature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            episode_id = store.record_intervention_episode(
                session_id=None,
                event_timestamp=NOW,
                situation_type="debugging",
                activity="terminal",
                event_type="failure",
                topic="test-failure:python",
                failure_signature="test-failure:python",
                candidate_action="WATCH",
                final_action="WATCH",
            )
            store.record_intervention_feedback(episode_id, "dont-remind")
            other = extracted(action=Action.WATCH, failure="npm-failure:node")
            self.assertEqual(retrieve_relevant_intervention_preferences(store, other), [])
            store.close()

    def test_silence_preference_blocks_isolated_model_notify(self) -> None:
        watch = WatchManager()
        policy = ProactivePolicy(watch, HardRules(max_notifications_per_hour=10), PolicyThresholds())
        watch.observe_model(extracted(action=Action.WATCH), NOW, "initial watch")
        decision = policy.decide(
            extracted(action=Action.NOTIFY),
            WorkingState(current_objective="resolve failure"),
            1,
            NOW,
            preferences=[{
                "id": 7,
                "status": "ACTIVE",
                "preference": PreferenceKind.AVOID_ISOLATED.value,
            }],
        )
        self.assertNotEqual(decision.action, Action.NOTIFY)
        self.assertEqual(decision.action, Action.REMEMBER)
        self.assertEqual(decision.preference_ids, (7,))
        self.assertEqual(decision.preference_effect, "USER_PREFERS_SILENCE_FOR_ISOLATED_ERROR")

    def test_early_warning_preference_creates_silent_watch(self) -> None:
        watch = WatchManager()
        policy = ProactivePolicy(watch, HardRules(max_notifications_per_hour=10), PolicyThresholds())
        decision = policy.decide(
            extracted(action=Action.IGNORE),
            WorkingState(),
            1,
            NOW,
            preferences=[{
                "id": 8,
                "status": "ACTIVE",
                "preference": PreferenceKind.EARLIER_WARNING.value,
            }],
        )
        self.assertEqual(decision.action, Action.WATCH)
        self.assertEqual(decision.reason_code, "USER_PREFERS_EARLY_WARNING")
        self.assertEqual(watch.active.evidence, 1)

    def test_timing_preference_requires_more_watch_evidence(self) -> None:
        watch = WatchManager()
        policy = ProactivePolicy(watch, HardRules(max_notifications_per_hour=10), PolicyThresholds())
        failure = extracted(action=Action.WATCH)
        for offset in range(3):
            watch.observe_failure(failure, NOW.replace(second=offset), "prior evidence")
        decision = policy.decide(
            extracted(action=Action.IGNORE),
            WorkingState(current_objective="resolve failure"),
            4,
            NOW,
            preferences=[{
                "id": 9,
                "status": "ACTIVE",
                "preference": PreferenceKind.TIMING_SENSITIVE.value,
            }],
        )
        self.assertEqual(decision.action, Action.INVESTIGATE)
        self.assertEqual(decision.reason_code, "USER_PREFERS_BETTER_TIMING")

    def test_watch_resolution_updates_intervention_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = SecretaryEngine(self.make_config(root))
            try:
                engine.process(raw("2026-08-28T12:00:00Z", "pytest FAILED"))
                engine.process(raw("2026-08-28T12:00:40Z", "pytest FAILED"))
                recovery = engine.process(raw("2026-08-28T12:01:20Z", "pytest PASSED after fix; resolved"))
                self.assertEqual(recovery.decision.reason_code, "WATCH_RESOLVED")
                episodes = engine.store.recent_intervention_episodes(10)
                resolved = [item for item in episodes if item["watch_id"] == recovery.decision.watch_id]
                self.assertTrue(resolved)
                self.assertTrue(all(item["status"] == "RESOLVED" and item["outcome"] == "RESOLVED" for item in resolved))
            finally:
                engine.close()

    def test_watch_instance_id_prevents_old_transition_from_closing_new_watch(self) -> None:
        watch = WatchManager(expiration_minutes=1)
        event = extracted(action=Action.WATCH)
        watch.observe_failure(event, NOW, "first")
        first_id = watch.active.watch_id
        watch.expire(NOW.replace(minute=2))
        watch.observe_failure(event, NOW.replace(minute=2), "second")
        second_id = watch.active.watch_id
        self.assertNotEqual(first_id, second_id)
        transition = watch.drain_transitions()[0]
        self.assertEqual(transition["watch_id"], first_id)
        self.assertEqual(watch.active.watch_id, second_id)

    def test_starting_a_new_store_session_expires_unresumed_intervention_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            first_session = store.start_session()
            episode_id = store.record_intervention_episode(
                session_id=first_session,
                event_timestamp=NOW,
                situation_type="debugging",
                activity="terminal",
                event_type="failure",
                watch_id="watch-1",
                status="ACTIVE",
                candidate_action="WATCH",
                final_action="WATCH",
            )
            store.connection.execute("UPDATE sessions SET owner_pid = ? WHERE id = ?", (999999999, first_session))
            store.connection.commit()
            second_session = store.start_session()
            self.assertNotEqual(first_session, second_session)
            episode = store.get_intervention_episode(episode_id)
            self.assertEqual(episode["status"], "EXPIRED")
            self.assertEqual(episode["outcome"], "EXPIRED")
            store.close()

    def test_ordinary_activity_while_watching_is_not_an_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = SecretaryEngine(self.make_config(root))
            try:
                engine.process(raw("2026-08-28T12:00:00Z", "pytest FAILED"))
                engine.process(raw("2026-08-28T12:00:40Z", "pytest FAILED"))
                before = engine.store.connection.execute("SELECT COUNT(*) FROM intervention_episodes").fetchone()[0]
                engine.process(raw("2026-08-28T12:01:00Z", "still editing code"))
                after = engine.store.connection.execute("SELECT COUNT(*) FROM intervention_episodes").fetchone()[0]
                self.assertEqual(after, before)
            finally:
                engine.close()

    def test_privacy_and_stale_paths_do_not_create_intervention_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = SecretaryEngine(self.make_config(root, inference_min_interval_seconds=0, inference_stale_result_seconds=0))
            try:
                engine.process(raw("2026-08-28T12:00:00Z", "secret", "KeePass.exe"))
                self.assertEqual(engine.store.connection.execute("SELECT COUNT(*) FROM intervention_episodes").fetchone()[0], 0)
                _, work = engine.prepare_inference_batch([raw("2026-08-28T12:00:01Z", "pytest FAILED")])
                self.assertIsNotNone(work)
                assert work is not None
                engine.submit_inference(work)
                unrelated = raw("2026-08-28T12:00:02Z", "unrelated research", "chrome.exe")
                unrelated["window_title"] = "Chrome research"
                unrelated["event_source"] = "navigation"
                engine.note_generation(3, [unrelated])
                result = engine.run_scheduled_inference(work)
                self.assertIsNotNone(result)
                self.assertEqual(result.decision.reason_code, "STALE_RESULT")
                self.assertEqual(engine.store.connection.execute("SELECT COUNT(*) FROM intervention_episodes").fetchone()[0], 0)
            finally:
                engine.close()

    def test_feedback_and_profile_cli_are_inspectable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            engine = SecretaryEngine(config)
            engine.process(raw("2026-08-28T12:00:00Z", "pytest FAILED"))
            episode_id = int(engine.store.recent_intervention_episodes(1)[0]["id"])
            engine.close()
            output = io.StringIO()
            self.assertEqual(run_feedback(config, episode_id, "dont-remind", output), 0)
            self.assertIn("Active preference", output.getvalue())
            output = io.StringIO()
            self.assertEqual(run_recent_interventions(config, 5, output), 0)
            self.assertIn(f"#{episode_id}", output.getvalue())
            output = io.StringIO()
            self.assertEqual(run_secretary_profile(config, output), 0)
            self.assertIn("Avoid interrupting", output.getvalue())
            parsed = build_parser().parse_args(["feedback", "--episode-id", str(episode_id), "dont-remind"])
            self.assertEqual(parsed.episode_option, episode_id)
            self.assertEqual(parsed.episode_id_arg, "dont-remind")

    def test_credential_shaped_labels_are_redacted_by_semantic_sanitizer(self) -> None:
        for label in (
            "read client_secret=abc123 from the config",
            "refresh token expired access_token=eyJhbGciOi",
            "Authorization: Bearer q0secret.1token.value",
            "curl -H 'x-api-key: sk-1234567890abcdef' --url https://x",
            "-----BEGIN PRIVATE KEY-----\nABCD\n-----END PRIVATE KEY-----",
        ):
            self.assertEqual(sanitize_semantic_label(label, 300), "[redacted]", label)

    def test_forget_erases_preference_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            episode_id = store.record_intervention_episode(
                session_id=None,
                event_timestamp=NOW,
                situation_type="debugging",
                activity="terminal",
                event_type="failure",
                topic="test-failure:python",
                failure_signature="test-failure:python",
                candidate_action="WATCH",
                final_action="WATCH",
            )
            store.record_intervention_feedback(episode_id, "dont-remind")
            store.record_intervention_feedback(episode_id, "forget")
            row = store.connection.execute(
                "SELECT status, content FROM intervention_preferences"
            ).fetchone()
            self.assertEqual(row[0], "DISABLED")
            self.assertEqual(row[1], "")
            store.close()

    def test_decision_reason_is_redacted_before_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            store.record_decision("WATCH", "credential client_secret=leaked passed to policy", 2)
            store.record_decision_trace(
                session_id=None,
                event_timestamp=NOW,
                foreground_app="WindowsTerminal.exe",
                event_type="failure",
                candidate_action="NOTIFY",
                candidate_confidence=0.9,
                candidate_importance=0.8,
                interrupt_score=0.7,
                deterministic_evidence=2,
                watch_id=None,
                watch_evidence=0,
                policy_action="WATCH",
                final_action="WATCH",
                suppression_reason=None,
                inference_latency_ms=None,
                inference_mode="mock",
                reason="summary contains access_token=eyJhbGciOi and jwt payload",
                summary="plain summary",
            )
            decisions = store.connection.execute("SELECT reason FROM assistant_decisions").fetchall()
            traces = store.connection.execute("SELECT reason FROM decision_traces").fetchall()
            self.assertEqual(decisions[0][0], "[redacted]")
            self.assertEqual(traces[0][0], "[redacted]")
            store.close()
