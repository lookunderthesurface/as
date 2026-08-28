"""Visual cognition orchestrator.

Per engine: keys one frame at a time, spends pixels only when the keyframe
gate asks, and propagates results into the world state.  This module owns the
bridge between the deterministic gate and the VLM provider; the engine's job
is only to call ``on_accepted_event`` before building an inference request and
``on_inference_recorded`` after the decision pipeline persists its trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .keyframe import KeyframeDecision, VisualKeyframeScheduler
from .perception import GUIPerceptionOutput, GUIPerceptionProvider
from .state import SemanticGUIState
from .world import DesktopWorldState

REQUEST_STAGE = ("visual_keyframe", "structured_update", "same_state")


@dataclass(frozen=True)
class CognitionUpdate:
    stage: str
    state: SemanticGUIState | None = None
    decision: KeyframeDecision | None = None
    used_vision: bool = False
    perception_provider: str | None = None

    @property
    def is_same(self) -> bool:
        return self.stage == "same_state"

    @property
    def perception_failed(self) -> bool:
        return self.stage == "perception_failed"


class VisualCognition:
    """Owns the keyframe gate, the perception provider, and the world state."""

    def __init__(
        self,
        perception: GUIPerceptionProvider,
        world: DesktopWorldState | None = None,
        keyframes: VisualKeyframeScheduler | None = None,
        excluded_apps: tuple[str, ...] = (),
    ) -> None:
        self.perception = perception
        self.world = world or DesktopWorldState()
        self.keyframes = keyframes or VisualKeyframeScheduler()
        self.excluded_apps = tuple(item.casefold() for item in excluded_apps if item.strip())
        self._last_update: CognitionUpdate | None = None

    def is_excluded(self, event) -> bool:
        """Defense in depth: never send excluded-app pixels or text to a model."""
        app = getattr(event, "foreground_app", "") or ""
        return any(excluded in app.casefold() for excluded in self.excluded_apps)

    @property
    def last_update(self) -> CognitionUpdate | None:
        return self._last_update

    @property
    def current_gui_state(self) -> SemanticGUIState | None:
        return self.world.current_gui

    def on_accepted_event(
        self,
        event,
        now: datetime,
        *,
        skip_vision: bool = False,
    ) -> CognitionUpdate:
        """Called for an already privacy-cleared event before inference.

        ``skip_vision`` forces the structured-only path (used by tests and
        shadow sanity runs when no pixels are available).
        """
        if self.is_excluded(event):
            return CognitionUpdate(stage="same_state", decision=None)
        decision = self.keyframes.evaluate(event, now=now)
        if decision.is_visual and not skip_vision and event.image_path:
            output = self.perception.perceive(
                _build_perception_request(self, event, now)
            )
            if output.failed:
                self._last_update = CognitionUpdate(
                    stage="perception_failed",
                    state=self.world.current_gui,
                    decision=decision,
                    used_vision=True,
                    perception_provider=output.provider,
                )
                return self._last_update
            self.world.update(output)
            self._last_update = CognitionUpdate(
                stage="visual_keyframe",
                state=self.world.current_gui,
                decision=decision,
                used_vision=True,
                perception_provider=output.provider,
            )
        elif decision.is_visual and (skip_vision or not event.image_path):
            # The gate wants pixels but none are available this frame
            # (privacy-cleared, no screenshot); fall back to a structured
            # update rather than silently fabricating a visual state.
            from .structured import structured_gui_state

            state = structured_gui_state(
                event,
                decision,
                now=now,
                previous=self.world.current_gui,
            )
            self.world.update_structured(state)
            self._last_update = CognitionUpdate(
                stage="structured_update",
                state=state,
                decision=decision,
                used_vision=False,
            )
        else:
            from .structured import structured_gui_state

            state = structured_gui_state(
                event,
                decision,
                now=now,
                previous=self.world.current_gui,
            )
            if self.world.current_gui is not None and state.semantic_signature == self.world.current_gui.semantic_signature:
                self._last_update = CognitionUpdate(
                    stage="same_state",
                    state=self.world.current_gui,
                    decision=decision,
                    used_vision=False,
                )
            else:
                self.world.update_structured(state)
                self._last_update = CognitionUpdate(
                    stage="structured_update",
                    state=state,
                    decision=decision,
                    used_vision=False,
                )
        return self._last_update

    def recent_trajectory_text(self, limit: int = 900) -> str:
        return self.world.trajectory.to_text(limit)

    def snapshot(self) -> dict[str, object]:
        return self.world.snapshot()


def _build_perception_request(cognition: VisualCognition, event, now: datetime):
    from .perception import GUIPerceptionRequest

    world = cognition.world
    previous = world.current_gui.as_dict() if world.current_gui else None
    trajectory_text = world.trajectory.to_text(900)
    return GUIPerceptionRequest(
        event=event,
        previous_state=previous,
        trajectory_text=trajectory_text,
        generation_id=0,
    )
