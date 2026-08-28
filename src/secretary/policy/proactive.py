from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..context.working_state import WorkingState
from ..inference.schema import Action
from ..perception.extractor import ExtractedEvent
from .hard_rules import HardRules
from .watch import WatchHypothesis, WatchManager


@dataclass(frozen=True)
class PolicyThresholds:
    remember_min_confidence: float = 0.75
    remember_min_importance: float = 0.50
    watch_min_confidence: float = 0.65
    watch_min_importance: float = 0.50
    investigate_min_confidence: float = 0.70
    investigate_min_importance: float = 0.60
    notify_min_confidence: float = 0.80
    notify_min_importance: float = 0.75
    notify_min_interrupt_score: float = 0.70
    notify_min_watch_evidence: int = 2


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    evidence: int = 0
    notification_title: str | None = None
    notification_body: str | None = None
    candidate_action: Action = Action.IGNORE
    candidate_confidence: float = 0.0
    candidate_importance: float = 0.0
    interrupt_score: float = 0.0
    deterministic_evidence: int = 0
    watch_id: str | None = None
    watch_evidence: int = 0
    suppression_reason: str | None = None
    cloud_escalation_candidate: bool = False


class ProactivePolicy:
    """Fuse model suggestions with deterministic evidence and final hard gates."""

    def __init__(
        self,
        watch: WatchManager,
        hard_rules: HardRules,
        thresholds: PolicyThresholds | None = None,
    ) -> None:
        self.watch = watch
        self.hard_rules = hard_rules
        self.thresholds = thresholds or PolicyThresholds()

    def decide(self, event: ExtractedEvent, state: WorkingState, failure_count: int, now: datetime) -> Decision:
        self.watch.expire(now)
        candidate = event.candidate_action
        deterministic_evidence = failure_count if event.failure_signature else 0
        active_before = self.watch.active
        working_context = self._has_working_context(state)

        # This branch is intentionally retained as the reliable baseline. The
        # model can add context, but cannot turn repeated failures into silence.
        if event.event_type == "failure" and event.failure_signature:
            if failure_count <= 1:
                return self._decision(event, Action.REMEMBER, "first failure is worth remembering", deterministic_evidence=1)
            if failure_count == 2:
                self.watch.observe_failure(event, now, "second similar failure")
                return self._decision(event, Action.WATCH, "second similar failure; continue observing", deterministic_evidence=2)
            self.watch.observe_failure(event, now, f"failure repetition #{failure_count}")
            active = self.watch.active
            evidence = active.evidence if active else 0
            strong_deterministic = failure_count >= 4 and evidence >= 4
            if strong_deterministic:
                allowed, reason = self.hard_rules.can_notify(event, now, require_interrupt=False)
                if allowed:
                    return self._decision(
                        event,
                        Action.NOTIFY,
                        "repeated failure has high deterministic evidence",
                        evidence=evidence,
                        deterministic_evidence=failure_count,
                        notification_title="Ambient Secretary",
                        notification_body="The same failure pattern keeps recurring. Pause for a deliberate check of the failing command and the latest change.",
                    )
                return self._decision(
                    event,
                    Action.INVESTIGATE,
                    f"notification suppressed: {reason}",
                    evidence=evidence,
                    deterministic_evidence=failure_count,
                    suppression_reason=self._suppression_code(reason),
                )
            return self._decision(
                event,
                Action.INVESTIGATE,
                "repeated failure warrants investigation",
                evidence=evidence,
                deterministic_evidence=failure_count,
            )

        if event.event_type == "documentation" and active_before:
            delta = self.watch.observe_related(event, now)
            if delta:
                return self._decision(event, Action.WATCH, "related documentation adds evidence")

        # A model WATCH is allowed to create only a bounded, expiring
        # observation hypothesis. It never produces a notification by itself.
        if candidate == Action.WATCH:
            if self._meets_watch_threshold(event):
                hypothesis = self.watch.observe_model(event, now, event_reason(event))
                if hypothesis is not None:
                    return self._decision(
                        event,
                        Action.WATCH,
                        "model watch candidate accepted into bounded hypothesis",
                        evidence=hypothesis.evidence,
                        hypothesis=hypothesis,
                    )
            if self._meaningful(event):
                return self._decision(event, Action.REMEMBER, "low-confidence model WATCH retained as memory only")
            return self._decision(event, Action.IGNORE, "model WATCH below watch thresholds", suppression_reason="low_watch_confidence_or_importance")

        if candidate == Action.REMEMBER and self._meets_remember_threshold(event):
            return self._decision(event, Action.REMEMBER, "meaningful model REMEMBER candidate accepted")

        if candidate == Action.ASK_CLOUD:
            # Cloud remains a mock boundary. Do not call it here; retain only a
            # safe diagnostic marker and use local policy if it has evidence.
            if self._meets_investigate_threshold(event) and self.watch.active and working_context:
                return self._decision(
                    event,
                    Action.INVESTIGATE,
                    "cloud escalation candidate recorded; local investigation selected",
                    evidence=self.watch.active.evidence,
                    suppression_reason="cloud_provider_mock",
                    cloud_escalation_candidate=True,
                )
            return self._decision(
                event,
                Action.IGNORE,
                "cloud escalation candidate recorded; cloud provider is mock",
                suppression_reason="cloud_provider_mock",
                cloud_escalation_candidate=True,
            )

        if candidate == Action.INVESTIGATE:
            if self._meets_investigate_threshold(event) and self.watch.active and working_context:
                return self._decision(
                    event,
                    Action.INVESTIGATE,
                    "model investigation candidate supported by active WATCH evidence",
                    evidence=self.watch.active.evidence,
                )
            if self._meaningful(event):
                return self._decision(event, Action.REMEMBER, "model INVESTIGATE lacks active evidence; retained as memory")
            return self._decision(event, Action.IGNORE, "model INVESTIGATE lacks active evidence", suppression_reason="insufficient_watch_evidence")

        if candidate == Action.NOTIFY:
            return self._model_notify_decision(event, now, working_context)

        return self._decision(event, Action.IGNORE, "no high-value intervention is indicated")

    def _model_notify_decision(self, event: ExtractedEvent, now: datetime, working_context: bool) -> Decision:
        active = self.watch.active
        watch_evidence = active.evidence if active else 0
        scores_ok = (
            event.confidence >= self.thresholds.notify_min_confidence
            and event.importance >= self.thresholds.notify_min_importance
            and event.interrupt_score >= self.thresholds.notify_min_interrupt_score
        )
        evidence_ok = active is not None and watch_evidence >= self.thresholds.notify_min_watch_evidence and working_context
        if not scores_ok:
            reason = "model NOTIFY suppressed by notification score thresholds"
            suppression = "low_interrupt_score" if event.interrupt_score < self.thresholds.notify_min_interrupt_score else "low_confidence_or_importance"
            return self._decision(event, Action.INVESTIGATE if self._meets_investigate_threshold(event) else Action.IGNORE, reason, evidence=watch_evidence, watch=active, suppression_reason=suppression)
        if not evidence_ok:
            return self._decision(
                event,
                Action.INVESTIGATE,
                "model NOTIFY suppressed until active WATCH evidence is sufficient",
                evidence=watch_evidence,
                watch=active,
                suppression_reason="insufficient_watch_evidence",
            )
        allowed, reason = self.hard_rules.can_notify(
            event,
            now,
            interrupt_score=event.interrupt_score,
            min_interrupt_score=self.thresholds.notify_min_interrupt_score,
            require_interrupt=True,
        )
        if not allowed:
            return self._decision(
                event,
                Action.INVESTIGATE,
                f"notification suppressed: {reason}",
                evidence=watch_evidence,
                watch=active,
                suppression_reason=self._suppression_code(reason),
            )
        return self._decision(
            event,
            Action.NOTIFY,
            "model NOTIFY passed evidence, score, and hard gates",
            evidence=watch_evidence,
            watch=active,
            notification_title="Ambient Secretary",
            notification_body="A watched work pattern has enough evidence for a deliberate check.",
        )

    def _decision(
        self,
        event: ExtractedEvent,
        action: Action,
        reason: str,
        *,
        evidence: int = 0,
        deterministic_evidence: int = 0,
        hypothesis: WatchHypothesis | None = None,
        watch: WatchHypothesis | None = None,
        notification_title: str | None = None,
        notification_body: str | None = None,
        suppression_reason: str | None = None,
        cloud_escalation_candidate: bool = False,
    ) -> Decision:
        watched = hypothesis or watch or self.watch.active
        return Decision(
            action=action,
            reason=reason,
            evidence=evidence,
            notification_title=notification_title,
            notification_body=notification_body,
            candidate_action=event.candidate_action,
            candidate_confidence=event.confidence,
            candidate_importance=event.importance,
            interrupt_score=event.interrupt_score,
            deterministic_evidence=deterministic_evidence,
            watch_id=watched.watch_id if watched else None,
            watch_evidence=watched.evidence if watched else 0,
            suppression_reason=suppression_reason,
            cloud_escalation_candidate=cloud_escalation_candidate,
        )

    def _meaningful(self, event: ExtractedEvent) -> bool:
        return event.importance >= self.thresholds.remember_min_importance and event.confidence >= self.thresholds.remember_min_confidence

    def _meets_remember_threshold(self, event: ExtractedEvent) -> bool:
        return self._meaningful(event)

    def _meets_watch_threshold(self, event: ExtractedEvent) -> bool:
        return event.confidence >= self.thresholds.watch_min_confidence and event.importance >= self.thresholds.watch_min_importance

    def _meets_investigate_threshold(self, event: ExtractedEvent) -> bool:
        return event.confidence >= self.thresholds.investigate_min_confidence and event.importance >= self.thresholds.investigate_min_importance

    @staticmethod
    def _has_working_context(state: WorkingState) -> bool:
        return bool(
            state.current_project
            or state.current_objective
            or state.current_subgoal
            or state.active_apps
            or state.recent_events
            or state.recent_failures
            or state.hypotheses
        )

    @staticmethod
    def _suppression_code(reason: str) -> str:
        text = reason.casefold()
        if "cooldown" in text:
            return "notification_cooldown"
        if "rate" in text:
            return "notification_rate_limit"
        if "interrupt" in text:
            return "low_interrupt_score"
        if "importance" in text or "confidence" in text:
            return "low_confidence_or_importance"
        if "repeated notification" in text:
            return "duplicate_notification"
        return "hard_rule_suppression"


def event_reason(event: ExtractedEvent) -> str:
    return event.candidate_reason[:300] if event.candidate_reason else "model proposed observing this work pattern"
