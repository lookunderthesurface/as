from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class UserReaction(str, Enum):
    """Observed or explicitly reported reaction to an intervention."""

    UNKNOWN = "UNKNOWN"
    OPENED = "OPENED"
    DISMISSED = "DISMISSED"
    IGNORED = "IGNORED"
    FOLLOWED = "FOLLOWED"
    EXPLICIT_POSITIVE = "EXPLICIT_POSITIVE"
    EXPLICIT_NEGATIVE = "EXPLICIT_NEGATIVE"


class InterventionOutcome(str, Enum):
    UNKNOWN = "UNKNOWN"
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    DEFERRED = "DEFERRED"
    EXPIRED = "EXPIRED"


class InterventionStatus(str, Enum):
    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"
    EXPIRED = "EXPIRED"
    RECORDED = "RECORDED"


class PreferenceSource(str, Enum):
    EXPLICIT_USER = "EXPLICIT_USER"
    OBSERVED_OUTCOME = "OBSERVED_OUTCOME"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    SYSTEM_DEFAULT = "SYSTEM_DEFAULT"


class PreferenceKind(str, Enum):
    AVOID_ISOLATED = "AVOID_ISOLATED"
    EARLIER_WARNING = "EARLIER_WARNING"
    MORE_PROACTIVE = "MORE_PROACTIVE"
    TIMING_SENSITIVE = "TIMING_SENSITIVE"


@dataclass(frozen=True)
class InterventionEpisode:
    """A bounded record of one meaningful intervention opportunity."""

    event_timestamp: str
    situation_type: str
    activity: str
    event_type: str
    candidate_action: str
    final_action: str
    reason_codes: tuple[str, ...] = ()
    model_confidence: float = 0.0
    importance: float = 0.0
    interrupt_score: float = 0.0
    was_notified: bool = False
    status: InterventionStatus = InterventionStatus.RECORDED
    user_reaction: UserReaction = UserReaction.UNKNOWN
    outcome: InterventionOutcome = InterventionOutcome.UNKNOWN
    topic: str | None = None
    failure_signature: str | None = None
    summary: str = ""
    watch_id: str | None = None
    watch_context: Mapping[str, object] = field(default_factory=dict)
    notification_id: int | None = None
    explicit_feedback: str | None = None
    preference_ids: tuple[int, ...] = ()
    learned_preference_id: int | None = None
    session_id: int | None = None
    id: int | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "event_timestamp": self.event_timestamp,
            "situation_type": self.situation_type,
            "activity": self.activity,
            "event_type": self.event_type,
            "topic": self.topic,
            "failure_signature": self.failure_signature,
            "summary": self.summary,
            "watch_id": self.watch_id,
            "watch_context": dict(self.watch_context) if isinstance(self.watch_context, Mapping) else {},
            "candidate_action": self.candidate_action,
            "final_action": self.final_action,
            "reason_codes": list(self.reason_codes),
            "model_confidence": self.model_confidence,
            "importance": self.importance,
            "interrupt_score": self.interrupt_score,
            "was_notified": self.was_notified,
            "notification_id": self.notification_id,
            "status": _enum_value(self.status),
            "user_reaction": _enum_value(self.user_reaction),
            "outcome": _enum_value(self.outcome),
            "explicit_feedback": self.explicit_feedback,
            "preference_ids": list(self.preference_ids),
            "learned_preference_id": self.learned_preference_id,
        }


@dataclass(frozen=True)
class InterventionPreference:
    """An explainable, scoped preference that can be superseded or disabled."""

    scope_key: str
    situation_type: str
    activity: str
    event_type: str
    preference: PreferenceKind
    content: str
    source: PreferenceSource
    confidence: float = 1.0
    evidence_count: int = 1
    topic: str | None = None
    failure_signature: str | None = None
    status: str = "ACTIVE"
    id: int | None = None
    created_at: str | None = None
    updated_at: str | None = None
    last_episode_id: int | None = None
    supersedes_id: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "scope_key": self.scope_key,
            "situation_type": self.situation_type,
            "activity": self.activity,
            "event_type": self.event_type,
            "topic": self.topic,
            "failure_signature": self.failure_signature,
            "preference": _enum_value(self.preference),
            "content": self.content,
            "source": _enum_value(self.source),
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "last_episode_id": self.last_episode_id,
            "supersedes_id": self.supersedes_id,
        }


@dataclass(frozen=True)
class FeedbackInstruction:
    value: str
    reaction: UserReaction = UserReaction.UNKNOWN
    preference: PreferenceKind | None = None


_FEEDBACK_ALIASES: dict[str, FeedbackInstruction] = {
    "USEFUL": FeedbackInstruction("USEFUL", UserReaction.EXPLICIT_POSITIVE, PreferenceKind.MORE_PROACTIVE),
    "POSITIVE": FeedbackInstruction("USEFUL", UserReaction.EXPLICIT_POSITIVE, PreferenceKind.MORE_PROACTIVE),
    "GOOD": FeedbackInstruction("USEFUL", UserReaction.EXPLICIT_POSITIVE, PreferenceKind.MORE_PROACTIVE),
    "MORE_PROACTIVE": FeedbackInstruction("MORE_PROACTIVE", UserReaction.EXPLICIT_POSITIVE, PreferenceKind.MORE_PROACTIVE),
    "EARLIER": FeedbackInstruction("MORE_PROACTIVE", UserReaction.EXPLICIT_POSITIVE, PreferenceKind.EARLIER_WARNING),
    "DONT_REMIND": FeedbackInstruction("DONT_REMIND", UserReaction.EXPLICIT_NEGATIVE, PreferenceKind.AVOID_ISOLATED),
    "DO_NOT_REMIND": FeedbackInstruction("DONT_REMIND", UserReaction.EXPLICIT_NEGATIVE, PreferenceKind.AVOID_ISOLATED),
    "NO": FeedbackInstruction("DONT_REMIND", UserReaction.EXPLICIT_NEGATIVE, PreferenceKind.AVOID_ISOLATED),
    "TIMING_BAD": FeedbackInstruction("TIMING_BAD", UserReaction.EXPLICIT_NEGATIVE, PreferenceKind.TIMING_SENSITIVE),
    "BAD_TIMING": FeedbackInstruction("TIMING_BAD", UserReaction.EXPLICIT_NEGATIVE, PreferenceKind.TIMING_SENSITIVE),
    "TIMING": FeedbackInstruction("TIMING_BAD", UserReaction.EXPLICIT_NEGATIVE, PreferenceKind.TIMING_SENSITIVE),
    "OBSERVED": FeedbackInstruction("OBSERVED"),
    "REACTION": FeedbackInstruction("REACTION"),
    "FORGET": FeedbackInstruction("FORGET", UserReaction.EXPLICIT_NEGATIVE),
    "DISABLE": FeedbackInstruction("FORGET", UserReaction.EXPLICIT_NEGATIVE),
}


def normalize_feedback(value: str) -> FeedbackInstruction:
    key = value.strip().upper().replace("-", "_").replace(" ", "_")
    instruction = _FEEDBACK_ALIASES.get(key)
    if instruction is None:
        allowed = ", ".join(sorted({item.value for item in _FEEDBACK_ALIASES.values()}))
        raise ValueError(f"unsupported feedback value {value!r}; use one of: {allowed}")
    return instruction


def parse_reaction(value: str | UserReaction | None) -> UserReaction:
    if value is None:
        return UserReaction.UNKNOWN
    if isinstance(value, UserReaction):
        return value
    try:
        return UserReaction(value.strip().upper().replace("-", "_"))
    except ValueError as exc:
        raise ValueError(f"unsupported user reaction: {value}") from exc


def parse_outcome(value: str | InterventionOutcome | None) -> InterventionOutcome:
    if value is None:
        return InterventionOutcome.UNKNOWN
    if isinstance(value, InterventionOutcome):
        return value
    try:
        return InterventionOutcome(value.strip().upper().replace("-", "_"))
    except ValueError as exc:
        raise ValueError(f"unsupported intervention outcome: {value}") from exc


def classify_situation(event_type: str, activity: str, failure_signature: str | None = None) -> str:
    """Use stable semantic buckets for deterministic preference matching."""
    event = event_type.casefold()
    current_activity = activity.casefold()
    if failure_signature or event == "failure":
        return "debugging"
    if event == "documentation" or current_activity == "research":
        return "research"
    if event == "coding" or current_activity == "editor":
        return "coding"
    if event == "terminal" or current_activity == "terminal":
        return "terminal"
    if event in {"app_switch", "navigation"} or current_activity == "navigation":
        return "navigation"
    return event or current_activity or "desktop"


def build_scope_key(
    *,
    situation_type: str,
    activity: str,
    event_type: str,
    topic: str | None,
    failure_signature: str | None,
) -> str:
    values = (situation_type, activity, event_type, topic or "*", failure_signature or "*")
    return "|".join(_scope_part(value) for value in values)


def _scope_part(value: str) -> str:
    return "-".join(value.casefold().strip().split())[:160] or "*"


def _enum_value(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw)
