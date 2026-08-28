from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..perception.extractor import ExtractedEvent


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
        if event.app in self.active_apps:
            self.active_apps.remove(event.app)
        self.active_apps.append(event.app)
        self.active_apps = self.active_apps[-8:]
        self.recent_events.append({
            "timestamp": getattr(event.timestamp, "isoformat", lambda: str(event.timestamp))(),
            "event_type": event.event_type,
            "activity": event.activity,
            "app": event.app,
            "summary": event.summary,
        })
        if event.event_type in {"recovery", "success"}:
            self.current_objective = None
            self.current_subgoal = None
            self.current_topic = None
        elif event.failure_signature:
            known_failure = event.failure_signature in self.recent_failures
            self.recent_failures.append(event.failure_signature)
            # Do not let a low-confidence, late result hijack the current
            # objective. A strong model signal or an already-known failure may
            # update the state; stale results use allow_objective_update=False.
            if allow_objective_update and (event.confidence >= 0.8 or known_failure):
                self.current_objective = f"resolve {event.failure_signature}"
                self.current_subgoal = "determine whether the failure is repeating"
            if event.topic and (event.confidence >= 0.8 or self.current_topic is None):
                self.current_topic = event.topic
        elif event.event_type == "documentation" and self.current_objective:
            self.current_subgoal = "compare documentation with the current failure"
            if event.topic:
                self.current_topic = event.topic

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
