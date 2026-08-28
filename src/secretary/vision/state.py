"""Semantic GUI state and deterministic GUI state deltas.

The system deliberately does not hand the rest of the runtime raw
images/OCR. A ``SemanticGUIState`` is the compressed, stable, verifiable
interpretation of one desktop moment; a ``GUIStateDelta`` is the bounded
difference between two moments. Both are kept small enough to persist and
to send back into the next perception so the model does not re-understand
the world from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SemanticGUIState:
    """One compressed interpretation of what the GUI currently expresses."""

    timestamp: datetime
    application: str = "unknown"
    window: str = ""
    activity: str = "desktop"
    topic: str | None = None
    task_hint: str | None = None
    regions: Mapping[str, object] = field(default_factory=dict)
    primary_content: str = ""
    progress: str = "unknown"
    interaction_state: str = "active"
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    visual_entities: Mapping[str, object] = field(default_factory=dict)
    relevant_text: str = ""
    confidence: float = 0.5
    perception_mode: str = "structured"
    keyframe_reason: str = "none"

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "application": self.application,
            "window": self.window,
            "activity": self.activity,
            "topic": self.topic,
            "task_hint": self.task_hint,
            "regions": dict(self.regions),
            "primary_content": self.primary_content,
            "progress": self.progress,
            "interaction_state": self.interaction_state,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "visual_entities": dict(self.visual_entities),
            "relevant_text": self.relevant_text,
            "confidence": self.confidence,
            "perception_mode": self.perception_mode,
            "keyframe_reason": self.keyframe_reason,
        }

    @classmethod
    def from_mapping(cls, value: object, timestamp: datetime) -> "SemanticGUIState":
        """Conservative conversion of an untrusted perception mapping."""
        if not isinstance(value, Mapping):
            return cls(timestamp=timestamp)
        return cls(
            timestamp=timestamp,
            application=str(value.get("application") or value.get("app") or "unknown")[:120],
            window=str(value.get("window") or "")[:160],
            activity=str(value.get("activity") or "desktop")[:80],
            topic=_optional_text(value.get("topic"), 160),
            task_hint=_optional_text(value.get("task_hint") or value.get("user_goal") or value.get("current_task_hint"), 240),
            regions=_bounded_mapping(value.get("regions"), 40, 160),
            primary_content=_bounded_text(value.get("primary_content"), 500),
            progress=_progress(str(value.get("progress") or "")),
            interaction_state=str(value.get("interaction_state") or "active")[:40],
            errors=tuple(_bounded_text(item, 200) for item in _string_list(value.get("errors"))[:6]),
            warnings=tuple(_bounded_text(item, 200) for item in _string_list(value.get("warnings"))[:4]),
            visual_entities=_bounded_mapping(value.get("visual_entities") or value.get("visual_components"), 20, 160),
            relevant_text=_bounded_text(value.get("relevant_text") or value.get("interpretation_text"), 800),
            confidence=_probability(value.get("confidence")),
        )

    @property
    def semantic_signature(self) -> str:
        parts = (
            self.application.casefold(),
            self.window.casefold(),
            self.activity.casefold(),
            self.topic or "",
            self.progress,
            "|".join(self.errors).casefold()[:300],
        )
        return ":".join(parts)

    def short_trajectory_label(self) -> str:
        return f"{self.activity} {self.topic or self.window or self.application}".strip().replace("  ", " ")


def _progress(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized in {"stalled", "stuck", "blocked", "failed", "failing"}:
        return "stalled"
    if normalized in {"recovered", "resolved", "complete", "completed", "passed", "done"}:
        return "recovered"
    return "running" if normalized in {"running", "active", "working"} else "unknown"


PROGRESS_ORDER = ("unknown", "running", "stalled", "recovered")


def compute_gui_state_delta(previous: SemanticGUIState | None, current: SemanticGUIState) -> "GUIStateDelta":
    """Deterministic bounded difference that never introduces new content."""
    changed: list[str] = []
    if previous is None:
        changed.append("first_state")
    else:
        if previous.application.casefold() != current.application.casefold():
            changed.append("application")
        if previous.window.casefold() != current.window.casefold():
            changed.append("window")
        if previous.activity.casefold() != current.activity.casefold():
            changed.append("activity")
        if (previous.topic or "") != (current.topic or "") and "topic" not in changed:
            changed.append("topic")
        if previous.progress != current.progress:
            changed.append("progress")
        if previous.interaction_state.casefold() != current.interaction_state.casefold():
            changed.append("interaction")
        previous_errors = set(previous.errors)
        current_errors = set(current.errors)
        if previous_errors != current_errors:
            changed.append("errors")
        if bool(previous.regions.get("modal")) != bool(current.regions.get("modal")):
            changed.append("modal")
        if previous.semantic_signature != current.semantic_signature and "first_state" not in changed:
            if len([c for c in changed if c in {"application", "window", "activity", "topic", "progress", "errors", "modal"}]) == 0:
                changed.append("layout")
    recovered = bool(previous is not None and previous.progress in {"stalled", "running"} and current.progress == "recovered")
    regressed = bool(previous is not None and previous.progress in {"running", "recovered"} and current.progress == "stalled")
    normalized = tuple(dict.fromkeys(changed))
    return GUIStateDelta(
        changed_fields=normalized,
        previous_progress=previous.progress if previous else "unknown",
        current_progress=current.progress,
        recovery=recovered,
        regression=regressed,
    )


@dataclass(frozen=True)
class GUIStateDelta:
    """Bounded change between consecutive GUI states."""

    changed_fields: tuple[str, ...] = ()
    previous_progress: str = "unknown"
    current_progress: str = "unknown"
    recovery: bool = False
    regression: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "changed_fields": list(self.changed_fields),
            "previous_progress": self.previous_progress,
            "current_progress": self.current_progress,
            "recovery": self.recovery,
            "regression": self.regression,
        }


def _optional_text(value: object, limit: int) -> str | None:
    text = _bounded_text(value, limit)
    return text or None


def _bounded_text(value: object, limit: int) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split()).strip()
    return text[: max(1, limit)] if text else ""


def _bounded_mapping(value: object, max_keys: int, value_limit: int) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return MappingProxyType({})
    cleaned: dict[str, object] = {}
    for key in list(value)[: max_keys]:
        item = value[key]
        safe_key = _bounded_text(key, 80)
        if not safe_key:
            continue
        if isinstance(item, Mapping):
            inner: dict[str, object] = {}
            for inner_key in list(item)[: 12]:
                inner[_bounded_text(inner_key, 60)] = _bounded_text(item[inner_key], value_limit)
            cleaned[safe_key] = inner
        elif isinstance(item, (bool, int, float)):
            cleaned[safe_key] = item
        else:
            cleaned[safe_key] = _bounded_text(item, value_limit)
    return MappingProxyType(cleaned)


def _string_list(value: object) -> Sequence[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(str(item) for item in value)


def _probability(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.5
    import math

    if not math.isfinite(number):
        return 0.5
    return min(1.0, max(0.0, number))
