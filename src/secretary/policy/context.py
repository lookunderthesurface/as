"""DecisionContext: the bounded policy input container.

Contains exactly what ``ProactivePolicy`` needs to decide, so the policy
does not reach into engines, stores, or vision internals.  The engine is
the only constructor; tests construct small contexts directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Sequence

from ..context.working_state import WorkingState
from ..perception.extractor import ExtractedEvent
from ..vision.world import DesktopWorldState
from .watch import WatchHypothesis


@dataclass(frozen=True)
class DecisionContext:
    """One deterministic slice of the system, ready for one policy decision."""

    event: ExtractedEvent
    working_state: WorkingState
    now: datetime
    failure_count: int = 0
    activation_failure_count: int = 0
    active_watch: WatchHypothesis | None = None
    watch_snapshot: tuple[Mapping[str, object], ...] = ()
    world_state: DesktopWorldState | None = None
    preferences: tuple[Mapping[str, object], ...] = ()
    similar_episodes: tuple[Mapping[str, object], ...] = ()
    relevant_memories: tuple[Mapping[str, object], ...] = ()

    @property
    def trajectory_text(self) -> str:
        if self.world_state is None:
            return ""
        return self.world_state.trajectory.to_text(900)

    @property
    def gui_state_text(self) -> str:
        if self.world_state is None or self.world_state.current_gui is None:
            return ""
        state = self.world_state.current_gui
        return (
            f"app={state.application}; window={state.window}; activity={state.activity}; "
            f"topic={state.topic or 'none'}; progress={state.progress}"
        )

    @property
    def memory_context(self) -> Sequence[Mapping[str, object]]:
        return self.relevant_memories
