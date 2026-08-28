from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from secretary.memory.consolidation import MemoryConsolidator
from secretary.memory.hierarchy import MemorySource, MemoryStatus, MemoryTier
from secretary.memory.store import MemoryStore


NOW = datetime(2026, 8, 28, 12, 0)


def make_episode(store: MemoryStore, *, minutes: int, outcome: str = "UNKNOWN", reaction: str = "UNKNOWN", signature: str = "test-failure:python", topic: str = "test-failure:python") -> int:
    return store.record_intervention_episode(
        session_id=None,
        event_timestamp=NOW,
        situation_type="debugging",
        activity="terminal",
        event_type="failure",
        topic=topic,
        failure_signature=signature,
        candidate_action="NOTIFY",
        final_action="WOULD_NOTIFY",
        outcome=outcome,
        user_reaction=reaction,
        status="RECORDED",
    )


class MemoryConsolidationTests(unittest.TestCase):
    def test_consolidation_produces_durable_memory_from_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            for minute in range(5):
                make_episode(store, minutes=minute, outcome="RESOLVED", reaction="EXPLICIT_POSITIVE")
            consolidator = MemoryConsolidator(store, min_episodes=3)
            result = consolidator.consolidate()
            self.assertEqual(result.memories_produced, 1)
            self.assertGreaterEqual(result.episodes_considered, 5)
            memories = store.active_memories(tier=MemoryTier.SEMANTIC)
            self.assertTrue(memories)
            memory = memories[0]
            self.assertEqual(memory["source"], MemorySource.CONSOLIDATED.value)
            self.assertEqual(memory["provider"], "consolidator")
            self.assertEqual(memory["model"], "deterministic-v1")
            self.assertEqual(len(memory["source_episode_ids"]), 5)
            store.close()

    def test_consolidation_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            for minute in range(5):
                make_episode(store, minutes=minute, outcome="RESOLVED", reaction="EXPLICIT_POSITIVE")
            consolidator = MemoryConsolidator(store, min_episodes=3)
            first = consolidator.consolidate()
            second = consolidator.consolidate()
            self.assertEqual(first.memories_produced, 1)
            self.assertEqual(second.memories_produced, 0)
            self.assertEqual(len(store.active_memories(tier=MemoryTier.SEMANTIC)), 1)
            store.close()

    def test_consolidation_does_not_destroy_source_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            episode_ids = [make_episode(store, minutes=m, outcome="RESOLVED", reaction="EXPLICIT_POSITIVE") for m in range(5)]
            MemoryConsolidator(store, min_episodes=3).consolidate()
            remaining = store.recent_intervention_episodes(limit=10)
            self.assertEqual({int(item["id"]) for item in remaining}, set(episode_ids))
            store.close()

    def test_consolidation_skips_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            make_episode(store, minutes=0, outcome="RESOLVED", reaction="EXPLICIT_POSITIVE")
            result = MemoryConsolidator(store, min_episodes=3).consolidate()
            self.assertEqual(result.skipped_reason, "episodes_below_threshold")
            self.assertEqual(store.active_memories(), [])
            store.close()

    def test_explicit_negative_feedback_supersedes_positive_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            for minute in range(4):
                make_episode(store, minutes=minute, outcome="RESOLVED", reaction="EXPLICIT_NEGATIVE")
            MemoryConsolidator(store, min_episodes=3).consolidate()
            memories = store.active_memories(tier=MemoryTier.SEMANTIC)
            self.assertTrue(memories)
            self.assertIn("dislikes", memories[0]["content"])
            store.close()

    def test_superseded_memory_not_retrieved_as_current_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            old = store.record_memory("Timing is root cause.", source=MemorySource.OBSERVED_EVENT, importance=0.7)
            new = store.record_memory("KV layout is root cause.", source=MemorySource.CONSOLIDATED, importance=0.95)
            store.supersede_memory(old, replacing_with=new)
            contents = [row["content"] for row in store.active_memories()]
            self.assertNotIn("Timing is root cause.", contents)
            self.assertIn("KV layout is root cause.", contents)
            store.close()


if __name__ == "__main__":
    unittest.main()
