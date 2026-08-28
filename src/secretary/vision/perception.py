"""GUI-first multimodal perception.

This is the interface between the deterministic keyframe gate and the local
VLM.  It never receives an excluded-app event (privacy filtering happens in
the engine before ``analyze`` is called) and never returns or persists raw
screenshots: the provider returns a bounded ``GUIPerceptionOutput`` that the
engine folds into world state and eventually into SQLite.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Protocol

from ..events.schema import NormalizedEvent
from .state import SemanticGUIState

GROUNDING_MAX_CHARS = 1200
TRAJECTORY_MAX_CHARS = 900


@dataclass(frozen=True)
class GUIPerceptionRequest:
    """Bounded input for one local VLM GUI perception call."""

    event: NormalizedEvent
    previous_state: Mapping[str, object] | None = None
    trajectory_text: str = ""
    generation_id: int = 0


@dataclass(frozen=True)
class GUIPerceptionOutput:
    state: SemanticGUIState
    provider: str = "unknown"
    model: str | None = None
    error_type: str | None = None

    @property
    def failed(self) -> bool:
        return self.error_type is not None

    @classmethod
    def safe(cls, provider: str = "unknown", model: str | None = None, error_type: str = "inference_failure") -> "GUIPerceptionOutput":
        return cls(
            state=SemanticGUIState(timestamp=_aware(datetime.now(timezone.utc))),
            provider=provider,
            model=model,
            error_type=error_type,
        )


class GUIPerceptionProvider(Protocol):
    """Provider-neutral boundary; the mock and Ollama adapter satisfy it."""

    def perceive(self, request: GUIPerceptionRequest) -> GUIPerceptionOutput: ...


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def render_gui_perception_prompt(request: GUIPerceptionRequest, grounding: str = "") -> str:
    """Render the deterministic prompt; the VLM must answer structured questions."""
    event = request.event
    sections: list[str] = [
        "TASK: understand ONE desktop GUI moment. Do not write a general image description.",
        "PRIORITY: trust the exact OCR/accessibility text above your pixel reading for wording.",
        "STAY BOUNDED: no secrets, no raw code, one-file summaries only.",
        "",
        "CURRENT MOMENT",
        f"app={event.foreground_app}; window={event.window_title}; source={event.event_source}",
    ]
    if event.browser_url:
        sections.append(f"browser_url={event.browser_url[:160]}")
    if event.text.strip():
        sections.append(f"visible_text={event.text[:400]}")
    if grounding:
        sections.append("")
        sections.append("STRUCTURED GROUNDING (exact text from OCR/UIA)")
        sections.append(grounding[:GROUNDING_MAX_CHARS])
    if request.previous_state:
        sections.append("")
        sections.append("PREVIOUS GUI STATE (what this moment should be compared to)")
        sections.append(_render_previous_state(request.previous_state))
    if request.trajectory_text:
        sections.append("")
        sections.append("SHORT SEMANTIC TRAJECTORY")
        sections.append(request.trajectory_text[:TRAJECTORY_MAX_CHARS])
    sections.append("")
    sections.append(
        "ANSWER ONLY THIS STRUCTURE: application, window, activity (one of "
        "coding|debugging|research|documentation|testing|maintenance). topic, "
        "regions (dict of labeled GUI areas), primary_content, progress (one of "
        "running|stalled|recovered|unknown), interaction_state, errors (list of "
        "short exact error strings), warnings, visual_entities, relevant_text, "
        "confidence in 0..1, current_task_hint."
    )
    return "\n".join(sections)


def _render_previous_state(state: Mapping[str, object]) -> str:
    lines: list[str] = []
    for key in ("application", "window", "activity", "topic", "progress", "errors"):
        value = state.get(key)
        if value:
            lines.append(f"{key}={str(value)[:160]}")
    if not lines:
        return "(none)"
    return "\n".join(lines[:10])


# The structured JSON this VLM request must return. A single object, no prose.
GUI_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "application": {"type": "string"},
        "window": {"type": "string"},
        "activity": {"type": "string"},
        "topic": {"type": ["string", "null"]},
        "current_task_hint": {"type": ["string", "null"]},
        "regions": {"type": "object"},
        "primary_content": {"type": "string"},
        "progress": {"type": "string"},
        "interaction_state": {"type": "string"},
        "errors": {"type": "array", "items": {"type": "string"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "visual_entities": {"type": "object"},
        "relevant_text": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["application", "activity", "progress", "confidence"],
}
