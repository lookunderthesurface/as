"""Deterministic mock for GUI perception (no vision model required in tests)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from .perception import GUIPerceptionOutput, GUIPerceptionProvider, GUIPerceptionRequest, render_gui_perception_prompt
from .state import SemanticGUIState


class MockGUIPerceptionProvider:
    """Structured, reproducible substitute for the local VLM GUI perception step."""

    name = "mock-gui"
    model = None

    def perceive(self, request: GUIPerceptionRequest) -> GUIPerceptionOutput:
        event = request.event
        prompt = render_gui_perception_prompt(request)
        app = event.foreground_app.casefold()
        window = event.window_title.casefold()
        text = event.text.casefold()
        browserish = any(token in (app, window) for token in ("chrome", "edge", "firefox", "browser"))
        terminalish = any(token in (app, window) for token in ("terminal", "powershell", "cmd.exe", "bash"))

        errors: list[str] = []
        warnings: list[str] = []
        progress = "running"
        activity = "desktop"
        topic = None
        if _has_error(text):
            errors.append(_first_line_of(event.text))
            progress = "stalled"
            topic = "terminal failure"
        elif _has_success(text) or _has_recovery(text):
            progress = "recovered"
            topic = "recovery"
        elif _has_progress(text):
            progress = "running"
        if browserish:
            activity = "research"
            if not topic:
                topic = "documentation research"
                warnings.append("searching for solution")
        elif terminalish:
            activity = "testing"
            if not topic:
                topic = "command run"
        elif "code" in app or "visual studio" in app:
            activity = "coding"
            if not topic:
                topic = "source editing"
        if "modal" in window or ("modal" in text and "dialog" in window):
            warnings.append("modal dialog visible")
        if "chart" in window or "graph" in window:
            activity = "reviewing"
            if not topic:
                topic = "data visualization"

        state = SemanticGUIState(
            timestamp=_aware(datetime.now(timezone.utc)),
            application=event.foreground_app or "unknown",
            window=event.window_title[:160],
            activity=activity,
            topic=topic,
            task_hint=("fix failing work" if progress == "stalled" else "continue current task"),
            regions={"visible": True, "modal": False},
            primary_content=_first_line_of(event.text)[:200],
            progress=progress,
            interaction_state="active",
            errors=tuple(errors[:6]),
            warnings=tuple(warnings[:4]),
            visual_entities={"app": event.foreground_app},
            relevant_text=event.text[:800],
            confidence=0.9 if errors else 0.85,
            perception_mode="vision",
            keyframe_reason="mock",
        )
        return GUIPerceptionOutput(state=state, provider=self.name)


def _first_line_of(value: str) -> str:
    line = value.strip().splitlines()
    return line[0][:200] if line else ""


_NEGATION_CONTEXT = ("un", "not ", "never ", "no ", "unsuccessful", "failure", "failed", "error")


def _has_error(text: str) -> bool:
    lowered = f" {text.casefold()} "
    for marker in (" traceback", "exception", "assertionerror", "syntaxerror", "valueerror", "fatal", "exited with code", "command not found", "npm err"):
        if marker.strip() in lowered or marker in lowered:
            return True
    for marker in ("failed", "failure", "error"):
        if marker in lowered:
            # A false positive like "no errors" or "download unsuccessful"
            # is a progress signal, not an error.
            if any(negated in lowered for negated in _NEGATION_CONTEXT):
                continue
            return True
    return False


def _has_success(text: str) -> bool:
    lowered = f" {text.casefold()} "
    markers = ("successfully", "succeeded", "all tests passed", "passed", "fixed", "resolved", "recovered")
    return any(marker in lowered for marker in markers)


def _has_recovery(text: str) -> bool:
    lowered = f" {text.casefold()} "
    return any(marker in lowered for marker in ("recovered", "resolved", "passed", "green", "0 errors", "0 failures"))


def _has_progress(text: str) -> bool:
    lowered = f" {text.casefold()} "
    return any(marker in lowered for marker in ("running", "started", "downloading", "compiling", "building", "retrying", "processing"))


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
