"""Deterministic adaptive visual perception gating.

GPUs are for the work that genuinely needs pixels.  The scheduler decides,
from cheap structured signals only, whether the current desktop moment
deserves a full VLM perception (``VISUAL``), a structured-only update
(``STRUCTURED``), or nothing new (``SAME``).  No second model is consulted
and this must be cheap per call, so the capture worker can run it inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Sequence

from ..events.schema import NormalizedEvent

KEYFRAME_SAME = "SAME"
KEYFRAME_STRUCTURED = "STRUCTURED"
KEYFRAME_VISUAL = "VISUAL"

# App/window/title heuristics that indicate the pixels carry layout meaning
# that OCR text alone cannot express.
_VISUAL_CONTENT_TOKENS = (
    "vs code",
    "visual studio",
    "code.exe",
    "figma",
    "photoshop",
    "design",
    "canvas",
    "blender",
    "unity",
    "image",
    "photo",
    "diagram",
    "chart",
    "graph",
    "modal",
    "dialog",
    "browser",
    "chrome",
    "edge",
    "firefox",
)


@dataclass(frozen=True)
class KeyframeDecision:
    """One answer to 'should we spend pixels on this moment?'."""

    level: str = KEYFRAME_SAME
    reason: str = "no_visual_change"
    structured_signature: str | None = None

    @property
    def is_visual(self) -> bool:
        return self.level == KEYFRAME_VISUAL

    @property
    def is_structured(self) -> bool:
        return self.level == KEYFRAME_STRUCTURED

    @property
    def needs_perception(self) -> bool:
        return self.is_visual


class VisualKeyframeScheduler:
    """Greedy, stateless-in-persistence adaptive gate for VLM perception."""

    def __init__(
        self,
        *,
        min_visual_interval_seconds: float = 45.0,
        forced_visual_interval_seconds: float = 90.0,
        same_app_lookback: int = 3,
    ) -> None:
        self.min_visual_interval_seconds = max(0.0, float(min_visual_interval_seconds))
        self.forced_visual_interval_seconds = max(self.min_visual_interval_seconds, float(forced_visual_interval_seconds))
        self.same_app_lookback = max(1, same_app_lookback)
        self._min_interval = timedelta(seconds=self.min_visual_interval_seconds)
        self._forced_interval = timedelta(seconds=self.forced_visual_interval_seconds)
        # In-memory transient tracker. One engine owns one scheduler. A
        # capture restart can lose this memory; the cost is one extra VLM call.
        self._last_event: NormalizedEvent | None = None
        self._last_visual_at: datetime | None = None
        self._last_structured_at: datetime | None = None
        self._recent_apps: list[str] = []
        self._recent_error_signatures: list[str] = []

    @property
    def recent_apps(self) -> Sequence[str]:
        return tuple(self._recent_apps)

    def evaluate(self, event: NormalizedEvent, now: datetime | None = None) -> KeyframeDecision:
        """Classify one accepted, already-privacy-filtered event."""
        current = _aware(now or datetime.now(timezone.utc))
        visual_signals: list[str] = []
        structured_signals: list[str] = []

        if event.visual_required:
            visual_signals.append("visual_required")
        if event.screen_changed and not self._same_identity(event):
            visual_signals.append("screen_changed")
        if self._application_changed(event):
            visual_signals.append("application_changed")
        elif self._window_changed(event):
            visual_signals.append("window_changed")
        if self._new_error_or_failure_appeared(event):
            visual_signals.append("new_error_observed")
        if self._looks_visual_content(event):
            visual_signals.append("visual_content_app")
        if self._text_shifted(event):
            structured_signals.append("text_update")

        forced_refresh = self._last_visual_at is not None and (current - self._last_visual_at) >= self._forced_interval if self._last_visual_at is not None else False
        if forced_refresh:
            visual_signals.append("periodic_refresh")

        # Visual cooldown: low-priority visual signals (screen_changed,
        # visual_content_app) must not burn a VLM call every frame. High
        # priority signals (app switch, fresh error, visual_required, forced
        # refresh) bypass the cooldown because they carry real novelty.
        if visual_signals and self._last_visual_at is not None:
            priority = any(
                signal in {"application_changed", "new_error_observed", "visual_required", "periodic_refresh"}
                for signal in visual_signals
            )
            if not priority and (current - self._last_visual_at) < self._min_interval:
                structured_signals.append("visual_cooldown_throttled")
                visual_signals = []

        if visual_signals:
            self._last_visual_at = current
            self._remember(event)
            return KeyframeDecision(KEYFRAME_VISUAL, "|".join(sorted(set(visual_signals))[:6]))
        if structured_signals:
            should_structured = self._last_structured_at is None or (current - self._last_structured_at) >= self._min_interval / 2
            if should_structured:
                self._last_structured_at = current
                self._remember(event)
                return KeyframeDecision(
                    KEYFRAME_STRUCTURED,
                    "|".join(sorted(set(structured_signals))[:4]),
                    structured_signature=self._structured_signature(event),
                )
        self._remember(event)
        return KeyframeDecision(KEYFRAME_SAME, self._same_reason(), structured_signature=self._structured_signature(event))

    # --- helpers -----------------------------------------------------------
    def _remember(self, event: NormalizedEvent) -> None:
        if self._last_event is None or event.foreground_app.casefold() != self._last_event.foreground_app.casefold():
            self._recent_apps.append(event.foreground_app)
            self._recent_apps = self._recent_apps[-8:]
        if _error_token(event.text):
            fingerprint = f"{event.foreground_app.casefold()}:{sha256(event.text[:160].encode('utf-8', errors='replace')).hexdigest()[:12]}"
            if fingerprint not in self._recent_error_signatures[-12:]:
                self._recent_error_signatures.append(fingerprint)
            self._recent_error_signatures = self._recent_error_signatures[-24:]
        self._last_event = event

    def _same_identity(self, event: NormalizedEvent) -> bool:
        if self._last_event is None:
            return False
        return (
            self._last_event.foreground_app.casefold() == event.foreground_app.casefold()
            and self._last_event.window_title.casefold() == event.window_title.casefold()
        )

    def _application_changed(self, event: NormalizedEvent) -> bool:
        return self._last_event is not None and self._last_event.foreground_app.casefold() != event.foreground_app.casefold()

    def _window_changed(self, event: NormalizedEvent) -> bool:
        return self._last_event is not None and self._last_event.window_title.casefold() != event.window_title.casefold()

    def _new_error_or_failure_appeared(self, event: NormalizedEvent) -> bool:
        if not _error_token(event.text):
            return False
        fingerprint = f"{event.foreground_app.casefold()}:{sha256(event.text[:160].encode('utf-8', errors='replace')).hexdigest()[:12]}"
        return fingerprint not in self._recent_error_signatures

    def _text_shifted(self, event: NormalizedEvent) -> bool:
        if self._last_event is None:
            return False
        return bool(event.text) and event.text[:160] != self._last_event.text[:160]

    @staticmethod
    def _looks_visual_content(event: NormalizedEvent) -> bool:
        context = f"{event.foreground_app} {event.window_title}".casefold()
        return any(token in context for token in _VISUAL_CONTENT_TOKENS) and "terminal" not in context and "powershell" not in context

    @staticmethod
    def _same_reason() -> str:
        return "desktop_unchanged"

    def _structured_signature(self, event: NormalizedEvent) -> str:
        compact = " ".join(event.text.split())[:120]
        return sha256(f"{event.foreground_app.casefold()}:{compact}".encode("utf-8", errors="replace")).hexdigest()[:16]


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


_ERROR_TOKEN_TEST = (
    "error",
    "exception",
    "traceback",
    "failed",
    "failure",
    "assertionerror",
    "fatal",
    "exited with code",
    "command not found",
    "npm err",
    "syntaxerror",
    "valueerror",
    "exception occurred",
)


def _error_token(text: str) -> bool:
    lowered = text.casefold()
    return any(token in lowered for token in _ERROR_TOKEN_TEST)
