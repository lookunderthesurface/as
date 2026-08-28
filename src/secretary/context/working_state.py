from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..perception.extractor import ExtractedEvent
from ..state.reducer import Observation, WorldStateReducer


@dataclass
class WorkingState:
    current_project: str | None = None
    current_objective: str | None = None
    current_subgoal: str | None = None
    current_topic: str | None = None
    active_apps: list[str] = field(default_factory=list)
    recent_events: deque[dict[str, object]] = field(default_factory=lambda: deque(maxlen=20))
    recent_failures: deque[str] = field(default_factory=lambda: deque(maxlen=12))
    decisions: deque[str] = field(default_factory=lambda: deque(maxlen=20))
    unresolved_questions: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    hypotheses: list[dict[str, object]] = field(default_factory=list)
    pending_reminders: deque[str] = field(default_factory=lambda: deque(maxlen=10))

    def observe(self, event: ExtractedEvent, *, allow_objective_update: bool = True) -> None:
        # The reducer is the single deterministic state-transition owner.
        observation = Observation(
            event_type=event.event_type,
            activity=event.activity,
            app=event.app,
            summary=event.summary,
            confidence=event.confidence,
            topic=event.topic,
            failure_signature=event.failure_signature,
            timestamp=getattr(event.timestamp, "isoformat", lambda: str(event.timestamp))(),
        )
        WorldStateReducer.apply(self, observation, allow_objective_update=allow_objective_update)

    def add_decision(self, decision: str) -> None:
        self.decisions.append(decision)

    def set_hypotheses(self, hypotheses: list[dict[str, object]]) -> None:
        self.hypotheses = hypotheses[:8]

    def snapshot(self) -> dict[str, object]:
        return {
            "current_project": self.current_project,
            "current_objective": self.current_objective,
            "current_subgoal": self.current_subgoal,
            "current_topic": self.current_topic,
            "active_apps": list(self.active_apps),
            "recent_events": list(self.recent_events),
            "recent_failures": list(self.recent_failures),
            "decisions": list(self.decisions),
            "unresolved_questions": list(self.unresolved_questions),
            "hypotheses": list(self.hypotheses),
            "pending_reminders": list(self.pending_reminders),
        }
