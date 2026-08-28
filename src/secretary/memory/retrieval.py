from __future__ import annotations

from collections.abc import Mapping

from ..perception.extractor import ExtractedEvent
from .intervention import classify_situation
from .store import MemoryStore


def retrieve_relevant_memories(store: MemoryStore, query: str, limit: int = 5) -> list[dict[str, object]]:
    """FTS-backed retrieval kept deliberately small for the first version."""
    safe_query = " ".join(part for part in query.split() if part.isalnum() or part.replace("-", "").isalnum())
    return store.search_memories(safe_query, limit) if safe_query else []


def retrieve_relevant_intervention_preferences(
    store: MemoryStore,
    event: ExtractedEvent,
    *,
    limit: int = 5,
) -> list[dict[str, object]]:
    """Return a small deterministic set of active preferences for this event."""
    situation = classify_situation(event.event_type, event.activity, event.failure_signature)
    ranked: list[tuple[int, dict[str, object]]] = []
    for preference in store.active_intervention_preferences(limit=200):
        score = _context_score(
            preference,
            situation=situation,
            event_type=event.event_type,
            activity=event.activity,
            topic=event.topic,
            failure_signature=event.failure_signature,
        )
        if score > 0:
            item = dict(preference)
            item["match_score"] = score
            ranked.append((score, item))
    # Store queries already return newest records first; a stable score-only
    # sort preserves that recency order for ties.
    ranked.sort(key=lambda item: -item[0])
    return [item[1] for item in ranked[: max(1, min(20, limit))]]


def retrieve_similar_intervention_episodes(
    store: MemoryStore,
    event: ExtractedEvent,
    *,
    limit: int = 3,
) -> list[dict[str, object]]:
    """Retrieve a few bounded, semantically similar intervention examples."""
    situation = classify_situation(event.event_type, event.activity, event.failure_signature)
    ranked: list[tuple[int, dict[str, object]]] = []
    for episode in store.recent_intervention_episodes(limit=100):
        score = _context_score(
            episode,
            situation=situation,
            event_type=event.event_type,
            activity=event.activity,
            topic=event.topic,
            failure_signature=event.failure_signature,
        )
        if score > 0:
            item = dict(episode)
            item["match_score"] = score
            ranked.append((score, item))
    ranked.sort(key=lambda item: -item[0])
    return [item[1] for item in ranked[: max(1, min(20, limit))]]


def _context_score(
    value: Mapping[str, object],
    *,
    situation: str,
    event_type: str,
    activity: str,
    topic: str | None,
    failure_signature: str | None,
) -> int:
    stored_failure = str(value.get("failure_signature") or "")
    current_failure = str(failure_signature or "")
    # A preference learned for one concrete failure must never leak into a
    # different failure merely because both happened in the terminal.
    if stored_failure and stored_failure.casefold() != current_failure.casefold():
        return 0
    if current_failure and not stored_failure:
        return 0
    stored_situation = str(value.get("situation_type") or "")
    stored_event_type = str(value.get("event_type") or "")
    stored_activity = str(value.get("activity") or "")
    if stored_situation and stored_situation.casefold() != situation.casefold():
        return 0
    if stored_event_type and stored_event_type.casefold() != event_type.casefold():
        return 0
    if stored_activity and stored_activity.casefold() != activity.casefold():
        return 0
    score = 0
    if stored_situation:
        score += 6
    if stored_event_type:
        score += 5
    if stored_activity:
        score += 2
    if current_failure and stored_failure:
        score += 12
    stored_topic = str(value.get("topic") or "")
    if topic and stored_topic:
        current_tokens = set(_tokens(topic))
        stored_tokens = set(_tokens(stored_topic))
        if current_tokens and stored_tokens:
            if not current_tokens & stored_tokens and not (current_failure and stored_failure):
                return 0
            if current_tokens & stored_tokens:
                score += 4 if current_tokens == stored_tokens else 2
    return score


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.casefold().replace("_", " ").replace("-", " ").split() if part)

