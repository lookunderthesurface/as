from __future__ import annotations

from time import monotonic
from collections.abc import Callable

from ..events.schema import NormalizedEvent


def should_use_vision(event: NormalizedEvent) -> bool:
    """Deterministic first-pass vision gate; no model or network is consulted."""
    if event.visual_required:
        return True
    text = event.text.strip()
    app_context = f"{event.foreground_app} {event.window_title}".casefold()
    if not text or len(text) < 24:
        return True
    if any(token in app_context for token in ("image", "photo", "video", "figma", "photoshop", "canvas", "design")):
        return True
    # Text-rich terminals and editors are deliberately text-first.
    if any(token in app_context for token in ("terminal", "powershell", "cmd.exe", "vscode", "visual studio", "code.exe", "jetbrains")):
        return False
    return False


class VisionGate:
    """Apply an optional cooldown after the pure deterministic vision rule."""

    def __init__(self, cooldown_seconds: float = 0.0, clock: Callable[[], float] = monotonic) -> None:
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.clock = clock
        self._last_used: float | None = None

    def allow(self, event: NormalizedEvent, now: float | None = None) -> bool:
        if not should_use_vision(event):
            return False
        if event.visual_required:
            self._last_used = self.clock() if now is None else now
            return True
        current = self.clock() if now is None else now
        if self._last_used is not None and current - self._last_used < self.cooldown_seconds:
            return False
        self._last_used = current
        return True
