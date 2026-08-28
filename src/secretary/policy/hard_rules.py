from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone

from ..perception.extractor import ExtractedEvent


class HardRules:
    def __init__(
        self,
        max_notifications_per_hour: int = 2,
        min_importance: float = 0.60,
        min_confidence: float = 0.75,
        cooldown_seconds: float = 0.0,
    ) -> None:
        self.max_notifications_per_hour = max(1, max_notifications_per_hour)
        self.min_importance = min(0.99, max(0.0, min_importance))
        self.min_confidence = min(0.99, max(0.0, min_confidence))
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self._notification_times: deque[datetime] = deque()
        self._notified_signatures: set[str] = set()
        self._last_notification_at: datetime | None = None

    def can_notify(
        self,
        event: ExtractedEvent,
        now: datetime,
        *,
        interrupt_score: float = 1.0,
        min_interrupt_score: float | None = None,
        require_interrupt: bool = False,
    ) -> tuple[bool, str]:
        if event.importance < self.min_importance:
            return False, "importance below notification threshold"
        if event.confidence < self.min_confidence:
            return False, "confidence below notification threshold"
        if require_interrupt and min_interrupt_score is not None and interrupt_score < min_interrupt_score:
            return False, "interrupt score below notification threshold"
        cutoff = now - timedelta(hours=1)
        while self._notification_times and self._notification_times[0] < cutoff:
            self._notification_times.popleft()
        if self._last_notification_at is not None and self.cooldown_seconds:
            elapsed = (now - self._last_notification_at).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False, "notification cooldown active"
        if len(self._notification_times) >= self.max_notifications_per_hour:
            return False, "notification rate limited"
        if event.failure_signature and event.failure_signature in self._notified_signatures:
            return False, "repeated notification suppressed"
        return True, "final gate passed"

    def mark_notified(self, event: ExtractedEvent, now: datetime) -> None:
        self._notification_times.append(now)
        self._last_notification_at = now
        if event.failure_signature:
            self._notified_signatures.add(event.failure_signature)
