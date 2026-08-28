from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from secretary.memory.hierarchy import (
    CORE_MEMORY_CHARS_LIMIT,
    MemorySource,
    MemoryStatus,
    MemoryTier,
    MODEL_INFERENCE_MAX_CONFIDENCE,
)
from secretary.memory.store import MemoryStore


NOW = datetime(2026, 8, 28, 12, 0).isoformat()


class MemoryHierarchyTests(unittest.TestCase):
    def make_store(self, directory: str) -> MemoryStore:
        return MemoryStore(Path(directory) / "state.db")

    def test_core_memory_roundtrips_and_is_always_retrieved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            core_id = store.record_memory(
                "User strongly prefers low interruption.",
                source=MemorySource.EXPLICIT_USER,
                importance=1.0,
                tier=MemoryTier.CORE,
                confidence=1.0,
            )
            store.record_memory(
                "Saw a pytest failure for attention.py today.",
                source=MemorySource.OBSERVED_EVENT,
                importance=0.4,
                tier=MemoryTier.EPISODIC,
                confidence=0.5,
            )
            context = store.get_memory_context(max_chars=2000)
            self.assertIn("low interruption", context["core_memory"])
            self.assertGreater(context["core_memory_chars"], 0)
            self.assertLessEqual(
                context["core_memory_chars"] + context["retrieved_memory_chars"],
                CORE_MEMORY_CHARS_LIMIT + 2000,
            )
            core = store.core_memory()
            self.assertEqual(len(core), 1)
            self.assertEqual(core[0]["id"], core_id)
            store.close()

    def test_superseded_memory_is_not_retrieved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            old = store.record_memory("Timing is the root cause.", source=MemorySource.OBSERVED_EVENT, importance=0.7, confidence=0.6)
            new = store.record_memory("KV layout is the confirmed root cause.", source=MemorySource.CONSOLIDATED, importance=0.95, confidence=0.9)
            self.assertTrue(store.supersede_memory(old, replacing_with=new))
            active = store.active_memories()
            contents = [row["content"] for row in active]
            self.assertNotIn("Timing is the root cause.", contents)
            self.assertIn("KV layout is the confirmed root cause.", contents)
            old_row = store.get_memory(old)
            self.assertEqual(old_row["status"], MemoryStatus.SUPERSEDED.value)
            self.assertEqual(old_row["supersedes_id"], new)
            store.close()

    def test_model_inference_memory_confidence_is_bounded(self) -> None:
        """A single model guess cannot permanently shape the profile."""
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            store.record_memory(
                "User probably dislikes notifications.",
                source=MemorySource.MODEL_INFERENCE,
                importance=0.9,
                confidence=0.95,
                tier=MemoryTier.CORE,
            )
            memories = store.core_memory()
            self.assertEqual(len(memories), 1)
            self.assertGreaterEqual(float(memories[0]["confidence"]), 0.0)
            self.assertLessEqual(float(memories[0]["confidence"]), MODEL_INFERENCE_MAX_CONFIDENCE)
            store.close()

    def test_memory_source_aliases_normalize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            store.record_memory("legacy secretary memory", source="secretary")
            store.record_memory("model said something", source="model")
            row = store.search_memories("legacy*", limit=5)
            self.assertTrue(row)
            self.assertEqual(row[0]["source"], MemorySource.SYSTEM_DEFAULT.value)
            model_row = store.search_memories("model*", limit=5)
            self.assertTrue(model_row)
            self.assertEqual(model_row[0]["source"], MemorySource.MODEL_INFERENCE.value)
            store.close()

    def test_privacy_redaction_preserved_for_new_memory_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            store.record_memory("token=sk-1234567890 super secret", source="user")
            rows = store.active_memories(limit=10)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["content"], "[redacted]")
            store.close()

    def test_memory_status_expire(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            memory_id = store.record_memory("temporary research note", source="user", confidence=0.5)
            self.assertTrue(store.expire_memory(memory_id))
            self.assertEqual(store.active_memories(), [])
            row = store.get_memory(memory_id)
            self.assertEqual(row["status"], MemoryStatus.EXPIRED.value)
            store.close()


if __name__ == "__main__":
    unittest.main()
