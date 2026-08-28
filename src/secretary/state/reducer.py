"""Pure, deterministic world-state transition reducer.

``apply`` is the single place that defines how a semantic observation folds
into the working state.  The engine calls it once per applied result; tests
call it directly and must observe identical transitions for identical inputs.

The reducer is deliberately a pure function: no LLM calls, no I/O, no wall
clock.  The engine owns when to run it and what to persist afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..perception.extractor import ExtractedEvent


@dataclass(frozen=True)
class Observation:
    """The semantic observation the reducer consumes (provider-neutral)."""

    event_type: str
    activity: str
    app: str
    summary: str
    confidence: float
    topic: str | None = None
    failure_signature: str | None = None
    timestamp: str | None = None


class WorldStateReducer:
    """Deterministic working-state transition rules."""

    @staticmethod
    def apply(state, observation: Observation, *, allow_objective_update: bool = True) -> None:
        """Fold one observation into a WorkingState (in place, deterministic)."""
        if observation.app in state.active_apps:
            state.active_apps.remove(observation.app)
        state.active_apps.append(observation.app)
        state.active_apps = state.active_apps[-8:]
        state.recent_events.append({
            "timestamp": observation.timestamp,
            "event_type": observation.event_type,
            "activity": observation.activity,
            "app": observation.app,
            "summary": observation.summary,
        })
        if observation.event_type in {"recovery", "success"}:
            state.current_objective = None
            state.current_subgoal = None
            state.current_topic = None
        elif observation.failure_signature:
            known_failure = observation.failure_signature in state.recent_failures
            state.recent_failures.append(observation.failure_signature)
            if allow_objective_update and (observation.confidence >= 0.8 or known_failure):
                state.current_objective = f"resolve {observation.failure_signature}"
                state.current_subgoal = "determine whether the failure is repeating"
            if observation.topic and (observation.confidence >= 0.8 or state.current_topic is None):
                state.current_topic = observation.topic
        elif observation.event_type == "documentation" and state.current_objective:
            state.current_subgoal = "compare documentation with the current failure"
            if observation.topic:
                state.current_topic = observation.topic


def to_observation(event: ExtractedEvent) -> Observation:
    """Project an extracted event onto the reducer contract."""
    return Observation(
        event_type=event.event_type,
        activity=event.activity,
        app=event.app,
        summary=event.summary,
        confidence=event.confidence,
        topic=event.topic,
        failure_signature=event.failure_signature,
    )
