from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from ..events.schema import NormalizedEvent
from .metrics import InferenceMetrics


class Action(str, Enum):
    """The only actions the inference boundary may suggest."""

    IGNORE = "IGNORE"
    REMEMBER = "REMEMBER"
    WATCH = "WATCH"
    INVESTIGATE = "INVESTIGATE"
    ASK_CLOUD = "ASK_CLOUD"
    NOTIFY = "NOTIFY"


ACTION_VALUES = tuple(action.value for action in Action)


@dataclass(frozen=True)
class InferenceRequest:
    """Transient, provider-neutral input for one semantic inference request."""

    current_event: NormalizedEvent
    recent_events: tuple[Mapping[str, object], ...] = ()
    working_state: Mapping[str, object] | None = None
    active_hypotheses: tuple[Mapping[str, object], ...] = ()
    recent_failures: tuple[str, ...] = ()
    recent_assistant_decisions: tuple[str, ...] = ()
    image_path: str | None = None
    use_vision: bool = False
    context_text: str = ""


@dataclass(frozen=True)
class InferenceEvent:
    event_type: str = "activity"
    activity: str = "desktop"
    summary: str = "Work activity was observed"
    topic: str | None = None
    failure_signature: str | None = None
    importance: float = 0.1
    novelty: float = 0.1
    confidence: float = 0.5


@dataclass(frozen=True)
class SecretaryAssessment:
    candidate_action: Action = Action.IGNORE
    interrupt_score: float = 0.0
    reason: str = "No high-value intervention is indicated"


@dataclass(frozen=True)
class InferenceResult:
    """Validated structured result. Candidate actions never bypass policy."""

    event: InferenceEvent
    secretary: SecretaryAssessment
    provider: str = "unknown"
    model: str | None = None
    error_type: str | None = None
    metrics: InferenceMetrics | None = None

    @classmethod
    def safe(
        cls,
        error_type: str = "inference_failure",
        provider: str = "unknown",
        model: str | None = None,
    ) -> "InferenceResult":
        return cls(
            event=InferenceEvent(summary="Inference result was unavailable", confidence=0.0),
            secretary=SecretaryAssessment(reason="Inference result unavailable"),
            provider=provider,
            model=model,
            error_type=error_type,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "event": {
                "event_type": self.event.event_type,
                "activity": self.event.activity,
                "summary": self.event.summary,
                "topic": self.event.topic,
                "failure_signature": self.event.failure_signature,
                "importance": self.event.importance,
                "novelty": self.event.novelty,
                "confidence": self.event.confidence,
            },
            "secretary": {
                "candidate_action": self.secretary.candidate_action.value,
                "interrupt_score": self.secretary.interrupt_score,
                "reason": self.secretary.reason,
            },
            "provider": self.provider,
            "model": self.model,
            "error_type": self.error_type,
            "metrics": self.metrics.as_dict() if self.metrics else None,
        }


def _bounded_text(value: object, default: str, limit: int) -> str:
    if not isinstance(value, str):
        return default
    text = value.strip()
    return text[:limit] if text else default


def _probability(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number):
        return default
    return min(1.0, max(0.0, number))


def _action(value: object) -> Action:
    try:
        return Action(str(value).upper())
    except (ValueError, TypeError):
        return Action.IGNORE


def validate_inference_result(
    value: object,
    *,
    provider: str = "unknown",
    model: str | None = None,
) -> InferenceResult:
    """Convert untrusted provider output to a conservative result."""
    if isinstance(value, InferenceResult):
        return value
    if not isinstance(value, Mapping):
        return InferenceResult.safe("malformed_result", provider=provider, model=model)

    raw_event = value.get("event")
    event_data = raw_event if isinstance(raw_event, Mapping) else value
    raw_secretary = value.get("secretary")
    secretary_data = raw_secretary if isinstance(raw_secretary, Mapping) else value
    event = InferenceEvent(
        event_type=_bounded_text(event_data.get("event_type"), "activity", 80),
        activity=_bounded_text(event_data.get("activity"), "desktop", 80),
        summary=_bounded_text(event_data.get("summary"), "Work activity was observed", 500),
        topic=_bounded_text(event_data.get("topic"), "", 160) or None,
        failure_signature=_bounded_text(event_data.get("failure_signature"), "", 160) or None,
        importance=_probability(event_data.get("importance"), 0.1),
        novelty=_probability(event_data.get("novelty"), 0.1),
        confidence=_probability(event_data.get("confidence"), 0.5),
    )
    secretary = SecretaryAssessment(
        candidate_action=_action(secretary_data.get("candidate_action", secretary_data.get("action"))),
        interrupt_score=_probability(secretary_data.get("interrupt_score"), 0.0),
        reason=_bounded_text(secretary_data.get("reason"), "No high-value intervention is indicated", 500),
    )
    return InferenceResult(
        event=event,
        secretary=secretary,
        provider=_bounded_text(value.get("provider"), provider, 80),
        model=_bounded_text(value.get("model"), model or "", 160) or None,
        error_type=_bounded_text(value.get("error_type"), "", 80) or None,
    )


RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "event": {
            "type": "object",
            "properties": {
                "event_type": {"type": "string"},
                "activity": {"type": "string"},
                "summary": {"type": "string"},
                "topic": {"type": ["string", "null"]},
                "failure_signature": {"type": ["string", "null"]},
                "importance": {"type": "number", "minimum": 0, "maximum": 1},
                "novelty": {"type": "number", "minimum": 0, "maximum": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["event_type", "activity", "summary", "importance", "novelty", "confidence"],
        },
        "secretary": {
            "type": "object",
            "properties": {
                "candidate_action": {"enum": list(ACTION_VALUES)},
                "interrupt_score": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
            "required": ["candidate_action", "interrupt_score", "reason"],
        },
    },
    "required": ["event", "secretary"],
}
