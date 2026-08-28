"""Cheap structured-only GUI updates for moments that need no pixels.

The keyframe scheduler already decided a moment changes in a text-level way
(terminal appends, same-window refresh).  This builds a bounded
``SemanticGUIState`` from structured fields only — no screenshot, no VLM.
"""

from __future__ import annotations

from datetime import datetime

from ..events.schema import NormalizedEvent
from .keyframe import KeyframeDecision
from .state import SemanticGUIState

_ERROR_FIELDS = (
    "traceback",
    "exception",
    "assertionerror",
    "syntaxerror",
    "valueerror",
    "except ",
    "failed",
    "failure",
    "fatal",
    "exited with code",
    "command not found",
    "npm err",
)


def structured_gui_state(event: NormalizedEvent, decision: KeyframeDecision, *, now: datetime | None = None, previous: SemanticGUIState | None = None) -> SemanticGUIState:
    """Build a state without pixels; never fabricate visual claims."""
    errors = _extract_errors(event.text)
    progress = "stalled" if errors else _progress_hint(event.text)
    activity = _activity_hint(previous, event)
    return SemanticGUIState(
        timestamp=now or event.timestamp,
        application=event.foreground_app or (previous.application if previous else "unknown"),
        window=event.window_title or (previous.window if previous else ""),
        activity=activity,
        topic=previous.topic if previous else None,
        task_hint="fix failing work" if errors else _task_hint(event.text),
        regions={"visible": True, "modal": False},
        primary_content=_first_line(event.text),
        progress=progress,
        interaction_state=_interaction_hint(event),
        errors=tuple(errors[:6]),
        warnings=(),
        visual_entities={"app": event.foreground_app, "window": event.window_title},
        relevant_text=event.text[:800],
        confidence=0.95,
        perception_mode="structured",
        keyframe_reason=decision.reason,
    )


def _first_line(value: str) -> str:
    line = value.strip().splitlines()
    return line[0][:200] if line else ""


def _extract_errors(text: str) -> list[str]:
    """One bounded error per error-bearing line; no marker-level duplication."""
    errors: list[str] = []
    for line in text.splitlines():
        compact = " ".join(line.split()).strip()
        if not compact:
            continue
        lowered = compact.casefold()
        if any(marker in lowered for marker in _ERROR_FIELDS):
            # "no errors" / "0 errors" is a clean state, not an error line.
            if "no errors" in lowered or "0 errors" in lowered or "no error" in lowered:
                continue
            errors.append(compact[:180])
    return errors[:6]


def _progress_hint(text: str) -> str:
    lowered = f" {text.casefold()} "
    # Recovery synonyms that can be prefixed by a negation (unsuccessful,
    # not resolved) must not flip a running task to recovered.
    if any(marker in lowered for marker in ("unsuccessful", "not resolved", "not passed", "never resolved")):
        return "running"
    if any(marker in lowered for marker in (" all tests passed", "all tests passed", "successfully", "succeeded", "passed", "resolved", "fixed", "0 errors", "0 failures", "green")):
        return "recovered"
    if any(marker in lowered for marker in ("running", "started", "downloading", "compiling", "building", "retrying", "processing", "reading")):
        return "running"
    return "unknown"


def _activity_hint(previous: SemanticGUIState | None, event: NormalizedEvent) -> str:
    app = f"{event.foreground_app} {event.window_title}".casefold()
    if "terminal" in app or "powershell" in app or "cmd" in app or "bash" in app:
        return "terminal"
    if "chrome" in app or "edge" in app or "firefox" in app or "browser" in app:
        return "research"
    if "vs code" in app or "visual studio" in app or "code.exe" in app or "jetbrains" in app:
        return "coding"
    if "unknown" in event.foreground_app.casefold() and previous is not None:
        return previous.activity
    return "desktop"


def _interaction_hint(event: NormalizedEvent) -> str:
    lowered = f"{event.foreground_app} {event.window_title} {event.text}".casefold()
    if "modal" in lowered or "dialog" in lowered:
        return "modal"
    if not event.text.strip():
        return "idle"
    return "active"


def _task_hint(text: str) -> str:
    lowered = text.casefold()
    if "pytest" in lowered or "npm test" in lowered or "cargo test" in lowered:
        return "run tests"
    if "git " in lowered:
        return "version control operation"
    if "install" in lowered or "update" in lowered:
        return "dependency operation"
    return "continue current task"
