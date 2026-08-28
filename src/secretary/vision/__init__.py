"""Adaptive GUI-first persistent visual cognition.

Screenpipe remains the raw observation source. This package converts selected
desktop moments into bounded ``SemanticGUIState`` objects, computes
``GUIStateDelta`` transitions, and folds everything into a
``DesktopWorldState`` with a compact ``SemanticTrajectory``.
"""

from .keyframe import KeyframeDecision, VisualKeyframeScheduler
from .perception import (
    GUIPerceptionOutput,
    GUIPerceptionProvider,
    GUIPerceptionRequest,
    render_gui_perception_prompt,
)
from .state import SemanticGUIState, GUIStateDelta, compute_gui_state_delta
from .world import DesktopWorldState, SemanticEvent, SemanticTrajectory

__all__ = [
    "KeyframeDecision",
    "VisualKeyframeScheduler",
    "GUIPerceptionOutput",
    "GUIPerceptionProvider",
    "GUIPerceptionRequest",
    "render_gui_perception_prompt",
    "SemanticGUIState",
    "GUIStateDelta",
    "compute_gui_state_delta",
    "DesktopWorldState",
    "SemanticEvent",
    "SemanticTrajectory",
]
