"""Persistent desktop world state and semantic trajectory.

World state is the system's ongoing compressed understanding of the user's
desktop.  Trajectory is the higher-level sequence of ``SemanticEvent`` items
that the brain and diagnostics consume, rather than the raw per-frame
observation stream.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Mapping, Sequence

from .perception import GUIPerceptionOutput
from .state import GUIStateDelta, SemanticGUIState, compute_gui_state_delta
@dataclass(frozen=True)
class SemanticEvent:
    """One compressed, human-readable step of the desktop history."""

    timestamp: datetime
    label: str
    activity: str = "desktop"
    application: str = "unknown"
    topic: str | None = None
    importance: float = 0.1

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "label": self.label,
            "activity": self.activity,
            "application": self.application,
            "topic": self.topic,
            "importance": self.importance,
        }


class SemanticTrajectory:
    """Bounded recent trajectory; adjacent similar steps are merged."""

    def __init__(self, max_events: int = 40, merge_after_seconds: int = 150) -> None:
        self.max_events = max(8, max_events)
        self.merge_after_seconds = max(30, merge_after_seconds)
        self._events: deque[SemanticEvent] = deque(maxlen=max_events)

    def append(self, event: SemanticEvent) -> None:
        """Merge a new event into the previous step when they express the same meaning."""
        if self._events:
            previous = self._events[-1]
            same_work = (
                event.application.casefold() == previous.application.casefold()
                and event.activity == previous.activity
                and (event.topic or "") == (previous.topic or "")
            )
            close_in_time = _seconds_between(previous.timestamp, event.timestamp) <= self.merge_after_seconds
            if same_work and close_in_time and event.timestamp >= previous.timestamp:
                self._events[-1] = dataclasses.replace(
                    previous,
                    timestamp=event.timestamp,
                    label=event.label,
                    importance=max(previous.importance, event.importance),
                )
                return
        self._events.append(event)

    def extend_delta(self, delta: GUIStateDelta, state: SemanticGUIState) -> None:
        label = _delta_label(delta, state)
        if label:
            self.append(
                SemanticEvent(
                    timestamp=state.timestamp,
                    label=label,
                    activity=state.activity,
                    application=state.application,
                    topic=state.topic,
                    importance=1.0 if delta.recovery or delta.regression else 0.35,
                )
            )

    def snapshot(self) -> tuple[SemanticEvent, ...]:
        return tuple(self._events)

    def to_text(self, limit_chars: int = 900) -> str:
        lines = [f"{_fmt(event.timestamp)} {event.label}" for event in list(self._events)[-20:]]
        text = "\n".join(lines)
        return text[: max(120, limit_chars)]

    def for_since(self, minutes: int) -> list[SemanticEvent]:
        cutoff = _aware(datetime.now(timezone.utc)) - timedelta(minutes=max(1, minutes))
        return [event for event in self._events if event.timestamp >= cutoff]

    def clear(self) -> None:
        self._events.clear()


@dataclass
class DesktopWorldState:
    """Ongoing maintained understanding of the user's desktop."""

    current_gui: SemanticGUIState | None = None
    last_delta: GUIStateDelta | None = None
    current_activity: str = "desktop"
    current_topic: str | None = None
    current_task_hint: str | None = None
    active_failures: list[str] = field(default_factory=list)
    trajectory: SemanticTrajectory = field(default_factory=SemanticTrajectory)
    last_output: GUIPerceptionOutput | None = None

    def update(self, output: GUIPerceptionOutput) -> GUIStateDelta | None:
        """Fold one perception result into world state; returns its delta, if any."""
        if output.failed:
            return None
        state = output.state
        if state.perception_mode in {"", "none"}:
            state = _copy_with_mode(state, output)
        delta = compute_gui_state_delta(self.current_gui, state)
        self.current_gui = state
        self.last_delta = delta
        self.last_output = output
        self._apply_state(state, delta)
        return delta

    def update_structured(self, state: SemanticGUIState) -> GUIStateDelta | None:
        """Fold a cheap structured-only state into the world."""
        delta = compute_gui_state_delta(self.current_gui, state)
        self.current_gui = state
        self.last_delta = delta
        self._apply_state(state, delta)
        return delta

    def _apply_state(self, state: SemanticGUIState, delta: GUIStateDelta) -> None:
        self.current_activity = state.activity
        self.current_topic = state.topic or self.current_topic
        self.current_task_hint = state.task_hint or self.current_task_hint
        if state.errors:
            for error in state.errors:
                if error not in self.active_failures:
                    self.active_failures.append(error)
                self.active_failures = self.active_failures[-8:]
        elif "application" in delta.changed_fields or delta.recovery:
            # A different app or a recovery signal closes the previous error set.
            self.active_failures.clear()
        self.trajectory.extend_delta(delta, state)

    def snapshot(self) -> dict[str, object]:
        return {
            "current_gui": self.current_gui.as_dict() if self.current_gui else None,
            "last_delta": self.last_delta.as_dict() if self.last_delta else None,
            "current_activity": self.current_activity,
            "current_topic": self.current_topic,
            "current_task_hint": self.current_task_hint,
            "active_failures": list(self.active_failures),
            "trajectory": [event.as_dict() for event in self.trajectory.snapshot()],
        }


def _copy_with_mode(state: SemanticGUIState, output: GUIPerceptionOutput) -> SemanticGUIState:
    import dataclasses

    return dataclasses.replace(state, perception_mode=output.provider or "perception")


def _delta_label(delta: GUIStateDelta, state: SemanticGUIState) -> str:
    if delta.recovery:
        return f"{state.activity}: recovered"
    if delta.regression:
        return f"{state.activity}: regressed"
    fields = set(delta.changed_fields)
    if fields == {"first_state"}:
        if state.progress != "unknown":
            return f"{state.activity}: {state.progress}"
        return f"{state.activity}: observed" if state.activity != "desktop" else f"{state.application}: observed"
    if not fields:
        return ""
    # Semantic signals beat identity changes in the trajectory label.
    if "progress" in fields:
        return f"{state.activity}: {state.progress}"
    if "errors" in fields:
        topic = f" on {state.topic}" if state.topic else ""
        return f"{state.activity}{topic}: errors"
    if "application" in fields:
        return f"{state.application}: changed"
    topic = f" on {state.topic}" if state.topic else ""
    return f"{state.activity}{topic}"


def _fmt(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%H:%M")


def _seconds_between(start: datetime, end: datetime) -> int:
    return max(0, int((end - start).total_seconds()))


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
