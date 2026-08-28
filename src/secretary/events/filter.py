from __future__ import annotations

from collections import deque

from .schema import NormalizedEvent


class MeaningfulEventFilter:
    """Drop duplicate capture frames while retaining useful app/activity changes."""

    def __init__(self, max_seen: int = 500) -> None:
        self._seen: deque[str] = deque(maxlen=max_seen)
        self._seen_set: set[str] = set()

    def accept(self, event: NormalizedEvent) -> bool:
        fingerprint = event.stable_id
        if event.frame_id is None:
            # Replay fixtures and some capture rows have no frame id. Their timestamp
            # is the only provider-neutral boundary between repeated observations.
            fingerprint = ":".join((event.timestamp.isoformat(), event.foreground_app, event.window_title, event.event_source, event.text[:160]))
        if fingerprint in self._seen_set:
            return False
        if len(self._seen) == self._seen.maxlen:
            self._seen_set.discard(self._seen[0])
        self._seen.append(fingerprint)
        self._seen_set.add(fingerprint)
        return bool(event.foreground_app and (event.text.strip() or event.screen_changed or event.event_source))
