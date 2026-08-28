from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from secretary.config import SecretaryConfig
from secretary.context.working_state import WorkingState
from secretary.engine import SecretaryEngine
from secretary.inference.schema import Action, InferenceEvent, InferenceResult, SecretaryAssessment
from secretary.perception.extractor import ExtractedEvent
from secretary.policy.hard_rules import HardRules
from secretary.policy.proactive import PolicyThresholds, ProactivePolicy
from secretary.policy.watch import WatchManager
from secretary.notifications.mock import MockNotificationProvider


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


def extracted(
    *,
    action: Action = Action.IGNORE,
    confidence: float = 0.9,
    importance: float = 0.8,
    interrupt: float = 0.8,
    event_type: str = "research",
    topic: str | None = "pytest",
    failure: str | None = None,
) -> ExtractedEvent:
    return ExtractedEvent(
        timestamp=NOW,
        event_type=event_type,
        activity="research",
        app="Chrome.exe",
        summary="A bounded semantic summary",
        importance=importance,
        novelty=0.5,
        confidence=confidence,
        failure_signature=failure,
        topic=topic,
        candidate_action=action,
        interrupt_score=interrupt,
    )


def policy(*, cooldown: float = 0.0, max_hypotheses: int = 3) -> tuple[ProactivePolicy, WatchManager, HardRules]:
    watch = WatchManager(max_active_hypotheses=max_hypotheses)
    hard = HardRules(max_notifications_per_hour=10, cooldown_seconds=cooldown)
    return ProactivePolicy(watch, hard, PolicyThresholds()), watch, hard


class PolicyFusionTests(unittest.TestCase):
    def test_a_high_confidence_model_watch_creates_hypothesis(self) -> None:
        policy_obj, watch, _ = policy()
        decision = policy_obj.decide(extracted(action=Action.WATCH), WorkingState(), 0, NOW)
        self.assertEqual(decision.action, Action.WATCH)
        self.assertEqual(decision.candidate_action, Action.WATCH)
        self.assertEqual(len(watch.snapshot()), 1)
        self.assertEqual(watch.snapshot()[0]["evidence"], 1)

    def test_b_low_confidence_model_watch_does_not_create_watch(self) -> None:
        policy_obj, watch, _ = policy()
        decision = policy_obj.decide(extracted(action=Action.WATCH, confidence=0.4), WorkingState(), 0, NOW)
        self.assertEqual(decision.action, Action.IGNORE)
        self.assertEqual(watch.snapshot(), [])

    def test_c_model_notify_without_watch_evidence_is_suppressed(self) -> None:
        policy_obj, _, _ = policy()
        decision = policy_obj.decide(extracted(action=Action.NOTIFY), WorkingState(), 0, NOW)
        self.assertNotEqual(decision.action, Action.NOTIFY)
        self.assertEqual(decision.suppression_reason, "insufficient_watch_evidence")

    def test_d_model_notify_with_watch_evidence_reaches_hard_rules(self) -> None:
        policy_obj, watch, _ = policy()
        watch_event = extracted(action=Action.WATCH)
        policy_obj.decide(watch_event, WorkingState(), 0, NOW)
        watch.observe_model(watch_event, NOW + timedelta(seconds=1), "second related observation")
        decision = policy_obj.decide(extracted(action=Action.NOTIFY), WorkingState(current_objective="review pytest"), 0, NOW + timedelta(seconds=2))
        self.assertEqual(decision.action, Action.NOTIFY)
        self.assertEqual(decision.watch_evidence, 2)

    def test_e_notification_cooldown_suppresses_otherwise_valid_notify(self) -> None:
        policy_obj, watch, hard = policy(cooldown=300)
        first = extracted(action=Action.WATCH)
        policy_obj.decide(first, WorkingState(), 0, NOW)
        watch.observe_model(first, NOW + timedelta(seconds=1), "more evidence")
        valid = policy_obj.decide(extracted(action=Action.NOTIFY), WorkingState(current_objective="review pytest"), 0, NOW + timedelta(seconds=2))
        self.assertEqual(valid.action, Action.NOTIFY)
        hard.mark_notified(extracted(action=Action.NOTIFY), NOW + timedelta(seconds=2))
        second = extracted(action=Action.NOTIFY, topic="other-topic")
        watch.observe_model(second, NOW + timedelta(seconds=3), "other evidence")
        watch.observe_model(second, NOW + timedelta(seconds=4), "other evidence")
        suppressed = policy_obj.decide(second, WorkingState(current_objective="review other topic"), 0, NOW + timedelta(seconds=10))
        self.assertEqual(suppressed.action, Action.INVESTIGATE)
        self.assertEqual(suppressed.suppression_reason, "notification_cooldown")

    def test_f_deterministic_repeated_failure_overrides_model_ignore(self) -> None:
        policy_obj, _, _ = policy()
        event = extracted(action=Action.IGNORE, event_type="failure", topic=None, failure="test-failure:python")
        decision = policy_obj.decide(event, WorkingState(), 2, NOW)
        self.assertEqual(decision.action, Action.WATCH)

    def test_model_hypotheses_are_deduplicated_expiring_and_bounded(self) -> None:
        policy_obj, watch, _ = policy(max_hypotheses=3)
        state = WorkingState(current_objective="research")
        for topic in ("one", "two", "three", "four"):
            decision = policy_obj.decide(extracted(action=Action.WATCH, topic=topic), state, 0, NOW)
            self.assertIn(decision.action, (Action.WATCH, Action.REMEMBER))
        self.assertEqual(len(watch.snapshot()), 3)
        first = extracted(action=Action.WATCH, topic="one")
        policy_obj.decide(first, state, 0, NOW + timedelta(seconds=1))
        self.assertEqual(len(watch.snapshot()), 3)
        self.assertEqual(watch.active.evidence, 2)
        watch.expire(NOW + timedelta(minutes=21))
        self.assertEqual(watch.snapshot(), [])

    def test_g_privacy_blocks_inference_and_watch(self) -> None:
        calls = []

        class Provider:
            name = "fake"
            model = None

            def analyze(self, request):
                calls.append(request)
                return InferenceResult(
                    event=InferenceEvent(summary="should not be seen"),
                    secretary=SecretaryAssessment(),
                    provider=self.name,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SecretaryConfig(project_root=root, database_path=root / "state.db", log_directory=root / "logs")
            engine = SecretaryEngine(config, inference_provider=Provider(), notifier=MockNotificationProvider())
            try:
                result = engine.process({
                    "timestamp": NOW.isoformat(),
                    "foreground_app": "KeePass.exe",
                    "text": "secret",
                })
                self.assertTrue(result.privacy_suppressed)
                self.assertEqual(calls, [])
                self.assertEqual(engine.watch.snapshot(), [])
            finally:
                engine.close()
