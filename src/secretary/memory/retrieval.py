from __future__ import annotations

from .store import MemoryStore


def retrieve_relevant_memories(store: MemoryStore, query: str, limit: int = 5) -> list[dict[str, object]]:
    """FTS-backed retrieval kept deliberately small for the first version."""
    safe_query = " ".join(part for part in query.split() if part.isalnum() or part.replace("-", "").isalnum())
    return store.search_memories(safe_query, limit) if safe_query else []

