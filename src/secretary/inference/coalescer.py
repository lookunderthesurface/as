from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from collections.abc import Callable, Iterable

from ..events.schema import NormalizedEvent


@dataclass(frozen=True)
class EventBatch:
    """A short, bounded activity context for one inference request."""

    events: tuple[NormalizedEvent, ...]

    @property
    def current_event(self) -> NormalizedEvent:
        return self.events[-1]


class EventCoalescer:
    """Group nearby events and keep only a bounded recent context."""

    def __init__(
        self,
        window_seconds: float = 2.0,
        max_events: int = 20,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.window_seconds = max(0.0, window_seconds)
        self.max_events = max(1, max_events)
        self.clock = clock
        self._events: list[NormalizedEvent] = []
        self._last_arrival: float | None = None

    def add(self, event: NormalizedEvent, now: float | None = None) -> EventBatch | None:
        """Add an event; return a completed batch if the previous window ended."""
        arrival = self.clock() if now is None else now
        completed: EventBatch | None = None
        if self._events and self._last_arrival is not None and arrival - self._last_arrival > self.window_seconds:
            completed = self.flush()
        self._events.append(event)
        self._events.sort(key=lambda item: (item.timestamp, item.stable_id))
        self._events = self._events[-self.max_events :]
        self._last_arrival = arrival
        return completed

    def extend(self, events: Iterable[NormalizedEvent], now: float | None = None) -> EventBatch | None:
        completed: EventBatch | None = None
        for event in events:
            completed = self.add(event, now=now)
        return completed

    def flush(self) -> EventBatch | None:
        if not self._events:
            return None
        result = EventBatch(tuple(self._events))
        self._events.clear()
        self._last_arrival = None
        return result

    def snapshot(self) -> tuple[NormalizedEvent, ...]:
        return tuple(self._events)

    @property
    def pending_count(self) -> int:
        return len(self._events)
