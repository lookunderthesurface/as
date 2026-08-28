"""Background memory consolidation (Letta dreaming, done safely).

Consolidation is *not* real-time perception. It runs only when explicitly
triggered (session end, idle hour, or ``secretary consolidate``), folds a
bounded set of recent semantic signals into a small number of durable
memories, and never deletes source episodes. Every produced durable memory
carries ``source_episode_ids`` + provider + confidence so the system remains
auditable (why does it remember this?).

The consolidator is deterministic for the current version: it uses
structured signals (outcomes, feedback, trajectory labels) with no LLM call.
That keeps tests stable and avoids silent model-driven persona mutation.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from .hierarchy import MemorySource, MemoryTier, SOURCE_WEIGHT
from .store import MemoryStore

DEFAULT_BATCH_LIMIT = 60
CONSOLIDATION_MARKER_PREFIX = "__consolidated__"


@dataclass(frozen=True)
class ConsolidationResult:
    episodes_considered: int
    memories_produced: int
    superseded: int
    skipped_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "episodes_considered": self.episodes_considered,
            "memories_produced": self.memories_produced,
            "superseded": self.superseded,
            "skipped_reason": self.skipped_reason,
        }


class MemoryConsolidator:
    """Fold recent intervention evidence into durable semantic memories."""

    def __init__(self, store: MemoryStore, *, min_episodes: int = 3, batch_limit: int = DEFAULT_BATCH_LIMIT) -> None:
        self.store = store
        self.min_episodes = max(1, min_episodes)
        self.batch_limit = max(1, batch_limit)

    def consolidate(self, *, session_id: int | None = None, now: datetime | None = None) -> ConsolidationResult:
        marker = self._previous_consolidation(session_id)
        episodes = self.store.recent_intervention_episodes(limit=self.batch_limit)
        candidates = [episode for episode in episodes if _episode_id(episode) not in marker]
        if len(candidates) < self.min_episodes:
            return ConsolidationResult(len(candidates), 0, 0, skipped_reason="episodes_below_threshold")
        evaluated = self._evaluate(candidates, now=now)
        produced = 0
        superseded = 0
        for fact in evaluated:
            memory_id = self._upsert_durable_fact(fact, now=now)
            if memory_id is not None:
                produced += 1
            memory_id = self._supersede_older_equivalent(fact, newer_id=memory_id)
            if memory_id:
                superseded += 1
        self._remember_consolidation(session_id, candidates, now=now)
        return ConsolidationResult(len(candidates), produced, superseded)

    # --- internals ---------------------------------------------------------
    def _evaluate(self, episodes: Sequence[Mapping[str, object]], *, now: datetime | None = None) -> list[dict[str, object]]:
        """Extract durable facts from bounded episode evidence (no LLM)."""
        facts: list[dict[str, object]] = []
        by_signature: dict[str, dict[str, object]] = {}
        for episode in episodes:
            signature = str(episode.get("failure_signature") or "")[:120]
            reaction = str(episode.get("user_reaction") or "UNKNOWN")
            outcome = str(episode.get("outcome") or "UNKNOWN")
            if not signature and not str(episode.get("topic") or ""):
                continue
            key = signature or _topic_key(str(episode.get("topic") or ""))
            bucket = by_signature.setdefault(key, {
                "signature": signature,
                "topic": str(episode.get("topic") or ""),
                "episodes": [],
                "resolved": 0,
                "positive_feedback": 0,
                "explicit_negative": 0,
            })
            bucket["episodes"].append(episode)
            if outcome in {"RESOLVED", "EXPIRED"}:
                bucket["resolved"] += 1
            if reaction in {"EXPLICIT_POSITIVE", "ACCEPTED", "FOLLOWED"}:
                bucket["positive_feedback"] += 1
            if reaction in {"EXPLICIT_NEGATIVE", "REJECTED"}:
                bucket["explicit_negative"] += 1
        for bucket in by_signature.values():
            episodes = bucket["episodes"]
            seen = int(bucket["resolved"]) + int(bucket["positive_feedback"])
            evidence = self._evidence_weight(episodes)
            if evidence < 0.5:
                continue
            if len(episodes) < 2:
                continue
            topic_label = str(bucket["topic"] or "")[:160] or str(bucket["signature"])[:160] or "desktop"
            if bucket["explicit_negative"] > bucket["positive_feedback"]:
                conclusion = f"User dislikes reminders for {topic_label} situations."
                confidence = min(0.85, evidence)
            elif seen >= 2:
                conclusion = f"Repeated {topic_label} problem; prior attempts resolved with user attention or shared knowledge."
                confidence = min(0.8, evidence)
            else:
                conclusion = f"Recurring {topic_label} work pattern, resolution inconclusive."
                confidence = min(0.55, evidence)
            facts.append({
                "conclusion": conclusion,
                "topic": topic_label,
                "source_episode_ids": [_episode_id(ep) for ep in episodes[:8]],
                "confidence": confidence,
                "provider": "consolidator",
                "model": "deterministic-v1",
            })
        return facts[:8]

    def _evidence_weight(self, episodes: Sequence[Mapping[str, object]]) -> float:
        total = 0.0
        for episode in episodes[:10]:
            reaction = str(episode.get("user_reaction") or "UNKNOWN")
            outcome = str(episode.get("outcome") or "UNKNOWN")
            if reaction in {"EXPLICIT_POSITIVE", "ACCEPTED"}:
                total += 1.0
            elif reaction in {"EXPLICIT_NEGATIVE", "REJECTED"}:
                total += 0.8
            elif outcome == "RESOLVED":
                total += 0.4
            else:
                total += 0.1
        return total

    def _upsert_durable_fact(self, fact: Mapping[str, object], *, now: datetime | None = None) -> int | None:
        conclusion = str(fact.get("conclusion") or "")
        if not conclusion:
            return None
        existing = self._find_equivalent(conclusion)
        if existing is not None:
            return None
        return self.store.record_memory(
            conclusion,
            source=MemorySource.CONSOLIDATED,
            importance=min(0.9, float(fact.get("confidence") or 0.5) + 0.2),
            tier=MemoryTier.SEMANTIC,
            confidence=float(fact.get("confidence") or 0.5),
            source_episode_ids=[int(item) for item in (fact.get("source_episode_ids") or []) if item],
            provider=str(fact.get("provider") or "consolidator"),
            model=str(fact.get("model") or "deterministic-v1"),
        )

    def _find_equivalent(self, conclusion: str) -> int | None:
        rows = self.store.active_memories(tier=MemoryTier.SEMANTIC, limit=200)
        normalized = _normalize_text(conclusion)
        for row in rows:
            if row.get("source") != MemorySource.CONSOLIDATED.value:
                continue
            if _normalize_text(str(row.get("content") or "")) == normalized:
                return int(row["id"])
        for row in rows:
            topic = str(row.get("tags") or "")
            if conclusion[:80] in str(row.get("content") or ""):
                return int(row["id"])
        return None

    def _supersede_older_equivalent(self, fact: Mapping[str, object], *, newer_id: int | None = None) -> int | None:
        if newer_id is None:
            return None
        rows = self.store.active_memories(tier=MemoryTier.SEMANTIC, limit=200)
        normalized = _normalize_text(str(fact.get("conclusion") or ""))
        superseded = 0
        for row in rows:
            if int(row["id"]) == newer_id:
                continue
            if row.get("source") != MemorySource.CONSOLIDATED.value:
                continue
            if _normalize_text(str(row.get("content") or "")) == normalized:
                if self.store.supersede_memory(int(row["id"]), replacing_with=newer_id):
                    superseded += 1
        return superseded or None

    def _previous_consolidation(self, session_id: int | None) -> set[int]:
        marker = self.store.get_meta(CONSOLIDATION_MARKER_PREFIX + str(session_id if session_id is not None else "all"))
        if not marker:
            return set()
        try:
            parsed = json.loads(marker)
            if isinstance(parsed, list):
                return {int(item) for item in parsed}
        except (TypeError, ValueError, json.JSONDecodeError):
            return set()
        return set()

    def _remember_consolidation(self, session_id: int | None, episodes: Sequence[Mapping[str, object]], *, now: datetime | None = None) -> None:
        existing = self._previous_consolidation(session_id)
        for episode in episodes:
            episode_id = _episode_id(episode)
            if episode_id:
                existing.add(episode_id)
        self.store.set_meta(
            CONSOLIDATION_MARKER_PREFIX + str(session_id if session_id is not None else "all"),
            json.dumps(sorted(existing)[-200:], ensure_ascii=True),
        )


def _episode_id(episode: Mapping[str, object]) -> int:
    try:
        return int(episode.get("id"))
    except (TypeError, ValueError):
        return 0


def _topic_key(topic: str) -> str:
    return "topic-" + _normalize_text(topic)[:60]


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().casefold().split())[:400]


# LLM consolidation output must stay below durable-truth confidence.
LLM_CONSOLIDATED_MAX_CONFIDENCE = 0.75
LLM_KNOWLEDGE_MAX_ITEMS = 5


class ConsolidationValidator:
    """Gate between an LLM's memory proposal and durable storage.

    A model suggestion never becomes durable truth directly: sources must
    exist, content must be bounded and clean, confidence is capped, and
    duplicates/contradictions are demoted to low-confidence candidates.
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def validate(self, candidate: Mapping[str, object]) -> tuple[bool, str]:
        statement = str(candidate.get("statement") or "").strip()
        if not statement:
            return False, "empty_statement"
        if "[redacted]" in statement.casefold():
            return False, "redacted_content"
        if len(statement) > 500:
            return False, "statement_too_long"
        episode_ids = [int(item) for item in (candidate.get("source_episode_ids") or [])]
        if not episode_ids:
            return False, "missing_source_episodes"
        for episode_id in episode_ids:
            if self.store.get_intervention_episode(episode_id) is None:
                return False, f"unknown_source_episode:{episode_id}"
        try:
            confidence = float(candidate.get("confidence") or 0.0)
        except (TypeError, ValueError):
            return False, "bad_confidence"
        if confidence > LLM_CONSOLIDATED_MAX_CONFIDENCE:
            confidence = LLM_CONSOLIDATED_MAX_CONFIDENCE
        if confidence <= 0.0:
            return False, "non_positive_confidence"
        return True, "ok"


class LLMConsolidator:
    """Optional model-assisted dreaming on top of the deterministic baseline.

    The model receives ONLY privacy-filtered, already-semantic inputs
    (trajectory labels, bounded episode metadata). Its proposals pass a
    validator before persistence; any failure falls back to the
    deterministic consolidator. Never blocks real-time perception.
    """

    def __init__(self, store: MemoryStore, complete: Callable[[str], str], *, batch_limit: int = DEFAULT_BATCH_LIMIT) -> None:
        self.store = store
        self.complete = complete
        self.batch_limit = max(1, batch_limit)
        self.validator = ConsolidationValidator(store)

    def consolidate(self, *, session_id: int | None = None) -> ConsolidationResult:
        episodes = self.store.recent_intervention_episodes(limit=self.batch_limit)
        if len(episodes) < 2:
            return ConsolidationResult(len(episodes), 0, 0, skipped_reason="episodes_below_threshold")
        bounded = [
            {
                "id": _episode_id(episode),
                "situation": str(episode.get("situation_type") or ""),
                "activity": str(episode.get("activity") or ""),
                "topic": str(episode.get("topic") or ""),
                "outcome": str(episode.get("outcome") or ""),
                "reaction": str(episode.get("user_reaction") or ""),
                "summary": str(episode.get("summary") or "")[:200],
            }
            for episode in episodes
        ]
        prompt = (
            "Extract durable knowledge from these desktop-work episodes. "
            "Return ONLY compact JSON: {\"knowledge\": [{\"statement\": str<=300 chars, "
            "\"confidence\": 0..0.75, \"source_episode_ids\": [ids]}], \"episode_summary\": str<=300 chars}. "
            "Rules: generalizable lessons only (never transient state); every item must cite episode ids; "
            "no secrets, no raw screen text.\nEPISODES:\n"
            + json.dumps(bounded, ensure_ascii=True)
        )
        try:
            raw = self.complete(prompt)
            parsed = json.loads(raw)
        except Exception:
            return MemoryConsolidator(self.store, batch_limit=self.batch_limit).consolidate(session_id=session_id)
        knowledge_items = parsed.get("knowledge") if isinstance(parsed, Mapping) else None
        if not isinstance(knowledge_items, list):
            return MemoryConsolidator(self.store, batch_limit=self.batch_limit).consolidate(session_id=session_id)
        produced = 0
        accepted_items = []
        for item in knowledge_items[:LLM_KNOWLEDGE_MAX_ITEMS]:
            if not isinstance(item, Mapping):
                continue
            ok, _reason = self.validator.validate(item)
            if not ok:
                continue
            accepted_items.append(item)
            episode_ids = [int(e) for e in (item.get("source_episode_ids") or [])]
            memory_id = self.store.record_memory(
                str(item.get("statement")),
                source=MemorySource.CONSOLIDATED,
                importance=min(0.85, float(item.get("confidence") or 0.5) + 0.2),
                tier=MemoryTier.SEMANTIC,
                confidence=float(item.get("confidence") or 0.5),
                source_episode_ids=episode_ids,
                provider="ollama",
                model="consolidator-llm",
            )
            if memory_id:
                produced += 1
        self.store.set_meta(
            CONSOLIDATION_MARKER_PREFIX + str(session_id if session_id is not None else "all"),
            json.dumps(sorted({_episode_id(episode) for episode in episodes})[-200:], ensure_ascii=True),
        )
        return ConsolidationResult(len(episodes), produced, 0)
