from __future__ import annotations

from .store import MemoryStore


def compact_store(store: MemoryStore, max_events: int = 5000) -> None:
    store.trim(max_events=max_events)

