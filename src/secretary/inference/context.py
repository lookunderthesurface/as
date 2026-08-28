from __future__ import annotations

from collections.abc import Iterable, Mapping
from time import monotonic
from collections.abc import Callable

from ..events.schema import NormalizedEvent
from .schema import InferenceRequest
from .vision import VisionGate


class InferenceContextBuilder:
    """Build bounded, priority-ordered context for a local inference provider."""

    def __init__(self, max_text_chars: int = 6000, vision_cooldown_seconds: float = 0.0, clock: Callable[[], float] = monotonic) -> None:
        self.max_text_chars = max(500, max_text_chars)
        self.vision_gate = VisionGate(vision_cooldown_seconds, clock=clock)

    def build(
        self,
        current_event: NormalizedEvent,
        *,
        recent_events: Iterable[NormalizedEvent | Mapping[str, object]] = (),
        working_state: Mapping[str, object] | None = None,
        active_hypotheses: Iterable[Mapping[str, object]] = (),
        recent_failures: Iterable[str] = (),
        recent_assistant_decisions: Iterable[str] = (),
        image_path: str | None = None,
        use_vision: bool | None = None,
    ) -> InferenceRequest:
        vision = self.vision_gate.allow(current_event) if use_vision is None else bool(use_vision)
        recent_events_tuple = tuple(recent_events)
        hypotheses_tuple = tuple(active_hypotheses)
        failures_tuple = tuple(recent_failures)
        decisions_tuple = tuple(recent_assistant_decisions)
        text = self._render(
            current_event,
            recent_events=recent_events_tuple,
            working_state=working_state or {},
            active_hypotheses=hypotheses_tuple,
            recent_failures=failures_tuple,
            recent_assistant_decisions=decisions_tuple,
        )
        return InferenceRequest(
            current_event=current_event,
            recent_events=tuple(self._event_mapping(item) for item in recent_events_tuple),
            working_state=dict(working_state or {}),
            active_hypotheses=tuple(dict(item) for item in hypotheses_tuple),
            recent_failures=tuple(str(item)[:160] for item in failures_tuple),
            recent_assistant_decisions=tuple(str(item)[:80] for item in decisions_tuple),
            image_path=image_path if vision else None,
            use_vision=vision,
            context_text=text,
        )

    def _render(
        self,
        current_event: NormalizedEvent,
        *,
        recent_events: tuple[NormalizedEvent | Mapping[str, object], ...],
        working_state: Mapping[str, object],
        active_hypotheses: tuple[Mapping[str, object], ...],
        recent_failures: tuple[str, ...],
        recent_assistant_decisions: tuple[str, ...],
    ) -> str:
        current_text_budget = max(120, min(2000, self.max_text_chars // 3))
        sections = [
            "CURRENT EVENT\n" + self._current_line(current_event, current_text_budget),
            "CURRENT OBJECTIVE\n" + self._text(working_state.get("current_objective"), "none"),
            "CURRENT SUBGOAL\n" + self._text(working_state.get("current_subgoal"), "none"),
            "ACTIVE WATCH HYPOTHESIS\n" + self._render_mappings(active_hypotheses, "none"),
            "RECENT FAILURES\n" + self._render_values(recent_failures, "none"),
            "RECENT TRAJECTORY\n" + self._render_events(recent_events, "none"),
            "RECENT ASSISTANT DECISIONS\n" + self._render_values(recent_assistant_decisions, "none"),
        ]
        # Current event and active hypotheses are highest priority. Add sections
        # in priority order, dropping older trajectory text at the budget edge.
        result = sections[0]
        for section in sections[1:5]:
            result = self._append(result, section)
        trajectory = sections[5]
        decisions = sections[6]
        result = self._append(result, trajectory)
        result = self._append(result, decisions)
        return result[: self.max_text_chars]

    def _append(self, current: str, section: str) -> str:
        candidate = current + "\n\n" + section
        if len(candidate) <= self.max_text_chars:
            return candidate
        remaining = self.max_text_chars - len(current) - 2
        if remaining <= 0:
            return current
        return current + "\n\n" + section[:remaining]

    @staticmethod
    def _current_line(event: NormalizedEvent, text_limit: int = 2000) -> str:
        return f"app={event.foreground_app}; window={event.window_title}; source={event.event_source}; text={event.text[:text_limit]}"

    @staticmethod
    def _event_mapping(item: NormalizedEvent | Mapping[str, object]) -> Mapping[str, object]:
        if isinstance(item, NormalizedEvent):
            return {
                "timestamp": item.timestamp.isoformat(),
                "app": item.foreground_app,
                "window": item.window_title,
                "event_source": item.event_source,
                "text": item.text[:800],
            }
        return {str(key): value for key, value in item.items()}

    @classmethod
    def _render_events(cls, events: tuple[NormalizedEvent | Mapping[str, object], ...], default: str) -> str:
        if not events:
            return default
        lines: list[str] = []
        for item in events[-20:]:
            mapping = cls._event_mapping(item)
            lines.append(
                f"{mapping.get('timestamp', '')} {mapping.get('app', mapping.get('foreground_app', 'unknown'))}: "
                f"{mapping.get('text', mapping.get('summary', ''))}"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_mappings(values: tuple[Mapping[str, object], ...], default: str) -> str:
        if not values:
            return default
        return "\n".join(
            "; ".join(f"{key}={str(value)[:200]}" for key, value in item.items()) for item in values[-8:]
        )

    @staticmethod
    def _render_values(values: tuple[object, ...], default: str) -> str:
        return "\n".join(str(value)[:200] for value in values[-12:]) if values else default

    @staticmethod
    def _text(value: object, default: str) -> str:
        return str(value)[:500] if value else default
