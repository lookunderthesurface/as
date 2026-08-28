from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import math

from ..context.working_state import WorkingState
from ..inference.schema import Action
from ..memory.intervention import PreferenceKind, PreferenceSource
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
    reason_code: str = "POLICY_IGNORE"
    preference_ids: tuple[int, ...] = ()
    preference_effect: str | None = None
    similar_episode_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class PreferenceContext:
    """The small, already-ranked preference slice used for one decision."""

    ids: tuple[int, ...] = ()
    kinds: tuple[str, ...] = ()
    similar_episode_ids: tuple[int, ...] = ()

    @property
    def prefers_early_warning(self) -> bool:
        return any(kind in {PreferenceKind.EARLIER_WARNING.value, PreferenceKind.MORE_PROACTIVE.value} for kind in self.kinds)

    @property
    def avoids_isolated_interruptions(self) -> bool:
        return PreferenceKind.AVOID_ISOLATED.value in self.kinds

    @property
    def timing_sensitive(self) -> bool:
        return PreferenceKind.TIMING_SENSITIVE.value in self.kinds

    @property
    def effect(self) -> str | None:
        if self.avoids_isolated_interruptions:
            return "USER_PREFERS_SILENCE_FOR_ISOLATED_ERROR"
        if self.timing_sensitive:
            return "USER_PREFERS_BETTER_TIMING"
        if self.prefers_early_warning:
            return "USER_PREFERS_EARLY_WARNING"
        return None


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

    def decide(
        self,
        event: ExtractedEvent,
        state: WorkingState,
        failure_count: int,
        now: datetime,
        *,
        preferences: Sequence[Mapping[str, object]] = (),
        similar_episodes: Sequence[Mapping[str, object]] = (),
    ) -> Decision:
        self.watch.expire(now)
        candidate = event.candidate_action
        deterministic_evidence = failure_count if event.failure_signature else 0
        active_before = self.watch.active
        working_context = self._has_working_context(state)
        preference_context = self._preference_context(preferences, similar_episodes)

        if event.event_type in {"recovery", "success"} and self.watch.resolve(event, now):
            return self._decision(
                event,
                Action.IGNORE,
                "watched problem resolved",
                watch=active_before,
                reason_code="WATCH_RESOLVED",
                preference_context=preference_context,
            )

        # This branch is intentionally retained as the reliable baseline. The
        # model can add context, but cannot turn repeated failures into silence.
        if event.event_type == "failure" and event.failure_signature:
            if failure_count <= 1:
                if preference_context.prefers_early_warning and self._meets_watch_threshold(event):
                    self.watch.observe_failure(event, now, "explicit preference requests earlier warning")
                    return self._decision(
                        event,
                        Action.WATCH,
                        "explicit preference requests an earlier silent watch",
                        deterministic_evidence=1,
                        reason_code="USER_PREFERS_EARLY_WARNING",
                        preference_context=preference_context,
                    )
                return self._decision(
                    event,
                    Action.REMEMBER,
                    "first failure is worth remembering",
                    deterministic_evidence=1,
                    reason_code="REPEATED_FAILURE_FIRST",
                    preference_context=preference_context,
                )
            if failure_count == 2:
                self.watch.observe_failure(event, now, "second similar failure")
                return self._decision(
                    event,
                    Action.WATCH,
                    "second similar failure; continue observing",
                    deterministic_evidence=2,
                    reason_code="REPEATED_FAILURE_WATCH",
                    preference_context=preference_context,
                )
            self.watch.observe_failure(event, now, f"failure repetition #{failure_count}")
            active = self.watch.active
            evidence = active.evidence if active else 0
            strong_deterministic = failure_count >= 4 and evidence >= 4
            if strong_deterministic:
                if preference_context.timing_sensitive and evidence < self._timing_required_watch_evidence():
                    return self._decision(
                        event,
                        Action.INVESTIGATE,
                        "notification delayed because the user prefers stronger timing evidence",
                        evidence=evidence,
                        deterministic_evidence=failure_count,
                        suppression_reason="user_preference_timing_sensitive",
                        reason_code="USER_PREFERS_BETTER_TIMING",
                        preference_context=preference_context,
                    )
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
                        reason_code="REPEATED_FAILURE_NOTIFY",
                        preference_context=preference_context,
                    )
                return self._decision(
                    event,
                    Action.INVESTIGATE,
                    f"notification suppressed: {reason}",
                    evidence=evidence,
                    deterministic_evidence=failure_count,
                    suppression_reason=self._suppression_code(reason),
                    reason_code="REPEATED_FAILURE_NOTIFY_SUPPRESSED",
                    preference_context=preference_context,
                )
            return self._decision(
                event,
                Action.INVESTIGATE,
                "repeated failure warrants investigation",
                evidence=evidence,
                deterministic_evidence=failure_count,
                reason_code="REPEATED_FAILURE_INVESTIGATE",
                preference_context=preference_context,
            )

        if event.event_type == "documentation" and active_before:
            delta = self.watch.observe_related(event, now)
            if delta:
                return self._decision(
                    event,
                    Action.WATCH,
                    "related documentation adds evidence",
                    reason_code="WATCH_RELATED_DOCUMENTATION",
                    preference_context=preference_context,
                )

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
                        reason_code="MODEL_WATCH_ACCEPTED",
                        preference_context=preference_context,
                    )
            if self._meaningful(event):
                return self._decision(event, Action.REMEMBER, "low-confidence model WATCH retained as memory only", reason_code="MODEL_WATCH_REMEMBERED", preference_context=preference_context)
            return self._decision(event, Action.IGNORE, "model WATCH below watch thresholds", suppression_reason="low_watch_confidence_or_importance", reason_code="MODEL_WATCH_SUPPRESSED", preference_context=preference_context)

        if candidate == Action.REMEMBER and self._meets_remember_threshold(event):
            return self._decision(event, Action.REMEMBER, "meaningful model REMEMBER candidate accepted", reason_code="MODEL_REMEMBER_ACCEPTED", preference_context=preference_context)

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
                    reason_code="CLOUD_CANDIDATE_MOCK",
                    preference_context=preference_context,
                )
            return self._decision(
                event,
                Action.IGNORE,
                "cloud escalation candidate recorded; cloud provider is mock",
                suppression_reason="cloud_provider_mock",
                cloud_escalation_candidate=True,
                reason_code="CLOUD_CANDIDATE_MOCK",
                preference_context=preference_context,
            )

        if candidate == Action.INVESTIGATE:
            if self._meets_investigate_threshold(event) and self.watch.active and working_context:
                return self._decision(
                    event,
                    Action.INVESTIGATE,
                    "model investigation candidate supported by active WATCH evidence",
                    evidence=self.watch.active.evidence,
                    reason_code="MODEL_INVESTIGATE_ACCEPTED",
                    preference_context=preference_context,
                )
            if self._meaningful(event):
                return self._decision(event, Action.REMEMBER, "model INVESTIGATE lacks active evidence; retained as memory", reason_code="MODEL_INVESTIGATE_REMEMBERED", preference_context=preference_context)
            return self._decision(event, Action.IGNORE, "model INVESTIGATE lacks active evidence", suppression_reason="insufficient_watch_evidence", reason_code="MODEL_INVESTIGATE_SUPPRESSED", preference_context=preference_context)

        if candidate == Action.NOTIFY:
            return self._model_notify_decision(event, now, working_context, deterministic_evidence, preference_context)

        return self._decision(event, Action.IGNORE, "no high-value intervention is indicated", preference_context=preference_context)

    def _model_notify_decision(
        self,
        event: ExtractedEvent,
        now: datetime,
        working_context: bool,
        deterministic_evidence: int,
        preference_context: PreferenceContext,
    ) -> Decision:
        active = self.watch.active
        watch_evidence = active.evidence if active else 0
        if preference_context.avoids_isolated_interruptions and (not event.failure_signature or deterministic_evidence <= 1):
            return self._decision(
                event,
                Action.INVESTIGATE,
                "model NOTIFY suppressed by the user's isolated-error silence preference",
                evidence=watch_evidence,
                watch=active,
                suppression_reason="user_preference_avoid_isolated",
                reason_code="USER_PREFERS_SILENCE_FOR_ISOLATED_ERROR",
                preference_context=preference_context,
            )
        if preference_context.timing_sensitive and watch_evidence < self._timing_required_watch_evidence():
            return self._decision(
                event,
                Action.INVESTIGATE,
                "model NOTIFY delayed until stronger evidence arrives",
                evidence=watch_evidence,
                watch=active,
                suppression_reason="user_preference_timing_sensitive",
                reason_code="USER_PREFERS_BETTER_TIMING",
                preference_context=preference_context,
            )
        scores_ok = (
            event.confidence >= self.thresholds.notify_min_confidence
            and event.importance >= self.thresholds.notify_min_importance
            and event.interrupt_score >= self.thresholds.notify_min_interrupt_score
        )
        evidence_ok = active is not None and watch_evidence >= self.thresholds.notify_min_watch_evidence and working_context
        if not scores_ok:
            reason = "model NOTIFY suppressed by notification score thresholds"
            suppression = "low_interrupt_score" if event.interrupt_score < self.thresholds.notify_min_interrupt_score else "low_confidence_or_importance"
            return self._decision(event, Action.INVESTIGATE if self._meets_investigate_threshold(event) else Action.IGNORE, reason, evidence=watch_evidence, watch=active, suppression_reason=suppression, reason_code="MODEL_NOTIFY_SCORE_SUPPRESSED", preference_context=preference_context)
        if not evidence_ok:
            return self._decision(
                event,
                Action.INVESTIGATE,
                "model NOTIFY suppressed until active WATCH evidence is sufficient",
                evidence=watch_evidence,
                watch=active,
                suppression_reason="insufficient_watch_evidence",
                reason_code="MODEL_NOTIFY_EVIDENCE_SUPPRESSED",
                preference_context=preference_context,
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
                reason_code="MODEL_NOTIFY_HARD_RULE_SUPPRESSED",
                preference_context=preference_context,
            )
        return self._decision(
            event,
            Action.NOTIFY,
            "model NOTIFY passed evidence, score, and hard gates",
            evidence=watch_evidence,
            watch=active,
            notification_title="Ambient Secretary",
            notification_body="A watched work pattern has enough evidence for a deliberate check.",
            reason_code="MODEL_NOTIFY_ACCEPTED",
            preference_context=preference_context,
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
        reason_code: str = "POLICY_IGNORE",
        preference_context: PreferenceContext | None = None,
    ) -> Decision:
        watched = hypothesis or watch or self.watch.active
        context = preference_context or PreferenceContext()
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
            reason_code=reason_code,
            preference_ids=context.ids,
            preference_effect=context.effect,
            similar_episode_ids=context.similar_episode_ids,
        )

    @staticmethod
    def _preference_context(
        preferences: Sequence[Mapping[str, object]],
        similar_episodes: Sequence[Mapping[str, object]],
    ) -> PreferenceContext:
        ids: list[int] = []
        kinds: list[str] = []
        for preference in preferences:
            if str(preference.get("status", "ACTIVE")).upper() != "ACTIVE":
                continue
            source = str(preference.get("source") or "").upper()
            if not source:
                # Direct callers from older integrations did not include a
                # source; treat those records as explicit rather than silently
                # discarding them. Persisted records always have a source.
                source = PreferenceSource.EXPLICIT_USER.value
            try:
                confidence = float(preference.get("confidence", 0.0))
            except (TypeError, ValueError, OverflowError):
                confidence = 0.0
            if source not in {item.value for item in PreferenceSource}:
                continue
            try:
                evidence_count = int(preference.get("evidence_count", 1))
            except (TypeError, ValueError, OverflowError):
                evidence_count = 0
            if source != PreferenceSource.EXPLICIT_USER.value and (
                not math.isfinite(confidence) or confidence < 0.8 or evidence_count < 1
            ):
                continue
            try:
                preference_id = int(preference.get("id"))
            except (TypeError, ValueError):
                preference_id = 0
            if preference_id > 0 and preference_id not in ids:
                ids.append(preference_id)
            kind = str(preference.get("preference") or "").upper()
            if kind in {item.value for item in PreferenceKind} and kind not in kinds:
                kinds.append(kind)
        episode_ids: list[int] = []
        for episode in similar_episodes:
            try:
                episode_id = int(episode.get("id"))
            except (TypeError, ValueError):
                episode_id = 0
            if episode_id > 0 and episode_id not in episode_ids:
                episode_ids.append(episode_id)
        return PreferenceContext(tuple(ids[:8]), tuple(kinds[:8]), tuple(episode_ids[:8]))

    def _meaningful(self, event: ExtractedEvent) -> bool:
        return event.importance >= self.thresholds.remember_min_importance and event.confidence >= self.thresholds.remember_min_confidence

    def _meets_remember_threshold(self, event: ExtractedEvent) -> bool:
        return self._meaningful(event)

    def _meets_watch_threshold(self, event: ExtractedEvent) -> bool:
        return event.confidence >= self.thresholds.watch_min_confidence and event.importance >= self.thresholds.watch_min_importance

    def _meets_investigate_threshold(self, event: ExtractedEvent) -> bool:
        return event.confidence >= self.thresholds.investigate_min_confidence and event.importance >= self.thresholds.investigate_min_importance

    def _timing_required_watch_evidence(self) -> int:
        """Make a timing-sensitive preference require materially more evidence."""
        return max(self.thresholds.notify_min_watch_evidence + 3, 5)

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
