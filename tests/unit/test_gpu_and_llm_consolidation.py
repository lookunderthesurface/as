from __future__ import annotations

import io
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from secretary.config import SecretaryConfig
from secretary.main import run_consolidate, run_feedback
from secretary.memory.consolidation import ConsolidationValidator, LLMConsolidator
from secretary.memory.hierarchy import MemorySource, MemoryTier
from secretary.memory.store import MemoryStore
from secretary.runtime_gpu import GPUStatus, GPUStatusProvider, _classify


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


def make_episode(store: MemoryStore, minutes: int = 0) -> int:
    return store.record_intervention_episode(
        session_id=None,
        event_timestamp=NOW,
        situation_type="debugging",
        activity="terminal",
        event_type="failure",
        topic="test-failure:python",
        failure_signature="test-failure:python",
        candidate_action="NOTIFY",
        final_action="WOULD_NOTIFY",
        outcome="RESOLVED",
        user_reaction="EXPLICIT_POSITIVE",
    )


class GPUStatusProviderTests(unittest.TestCase):
    def test_classification_boundaries(self) -> None:
        self.assertEqual(_classify(3.0, 1000, 12282), GPUStatus.IDLE)
        self.assertEqual(_classify(30.0, 4000, 12282), GPUStatus.NORMAL)
        self.assertEqual(_classify(70.0, 4000, 12282), GPUStatus.BUSY)
        self.assertEqual(_classify(95.0, 11000, 12282), GPUStatus.CRITICAL)
        self.assertEqual(_classify(None, None, None), GPUStatus.UNKNOWN)

    def test_missing_nvidia_smi_is_unknown_not_error(self) -> None:
        provider = GPUStatusProvider(runner=lambda: (_ for _ in ()).throw(RuntimeError("missing")))
        snapshot = provider.snapshot(now=100.0)
        self.assertEqual(snapshot.status, GPUStatus.UNKNOWN)

    def test_interval_widening_and_dreaming_gate(self) -> None:
        self.assertEqual(GPUStatusProvider.min_visual_interval_for(GPUStatus.IDLE, 90.0), 90.0)
        self.assertEqual(GPUStatusProvider.min_visual_interval_for(GPUStatus.BUSY, 90.0), 180.0)
        self.assertEqual(GPUStatusProvider.min_visual_interval_for(GPUStatus.CRITICAL, 90.0), 360.0)
        self.assertFalse(GPUStatusProvider.dreaming_allowed(GPUStatus.CRITICAL))
        self.assertTrue(GPUStatusProvider.dreaming_allowed(GPUStatus.IDLE))


class LLMConsolidatorTests(unittest.TestCase):
    def test_valid_llm_knowledge_is_persisted_with_audit_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            episode_one = make_episode(store)
            episode_two = make_episode(store)
            payload = '{"knowledge": [{"statement": "KV placement resolved the test-failure:python class before.", "confidence": 0.7, "source_episode_ids": [' + str(episode_one) + ',' + str(episode_two) + ']}], "episode_summary": "debugging session"}'

            consolidator = LLMConsolidator(store, lambda _prompt: payload)
            result = consolidator.consolidate()
            self.assertEqual(result.memories_produced, 1)
            memory = store.active_memories(tier=MemoryTier.SEMANTIC)[0]
            self.assertEqual(memory["source"], MemorySource.CONSOLIDATED.value)
            self.assertEqual(memory["provider"], "ollama")
            self.assertEqual(memory["source_episode_ids"], [episode_one, episode_two])
            self.assertLessEqual(float(memory["confidence"]), 0.75)
            store.close()

    def test_llm_output_without_sources_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            make_episode(store)
            payload = '{"knowledge": [{"statement": "Unsupported claim.", "confidence": 0.9, "source_episode_ids": []}]}'
            result = LLMConsolidator(store, lambda _prompt: payload).consolidate()
            self.assertEqual(result.memories_produced, 0)
            self.assertEqual(store.active_memories(), [])
            store.close()

    def test_llm_failure_falls_back_to_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            for _ in range(4):
                make_episode(store)
            def broken(_prompt: str) -> str:
                raise RuntimeError("model offline")
            result = LLMConsolidator(store, broken).consolidate()
            self.assertEqual(result.skipped_reason, "episodes_below_threshold" if result.episodes_considered < 2 else None)
            store.close()


class ConsolidateCLITests(unittest.TestCase):
    def test_consolidate_cli_deterministic_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SecretaryConfig(project_root=root, database_path=root / "state.db", log_directory=root / "logs")
            store = MemoryStore(config.database_path)
            for _ in range(4):
                make_episode(store)
            store.close()
            output = io.StringIO()
            self.assertEqual(run_consolidate(config, output), 0)
            self.assertIn("deterministic-v1", output.getvalue())
            self.assertIn("Durable memories produced", output.getvalue())


class DimensionalFeedbackCLITests(unittest.TestCase):
    def test_feedback_cli_timing_content_learns_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SecretaryConfig(project_root=root, database_path=root / "data" / "state.db", log_directory=root / "logs")
            store = MemoryStore(config.database_path)
            episode_id = store.record_intervention_episode(
                session_id=None,
                event_timestamp=NOW,
                situation_type="debugging",
                activity="terminal",
                event_type="failure",
                topic="test-failure:python",
                failure_signature="test-failure:python",
                candidate_action="NOTIFY",
                final_action="WOULD_NOTIFY",
            )
            store.close()
            output = io.StringIO()
            self.assertEqual(run_feedback(config, episode_id, None, output, timing="too-early", content="too-generic"), 0)
            text = output.getvalue()
            self.assertIn("timing: TOO_EARLY", text)
            self.assertIn("content: TOO_GENERIC", text)
            self.assertIn("learned memory #", text)
            reopened = MemoryStore(config.database_path)
            timing_rows = [row for row in reopened.active_memories() if "timing-knowledge" in str(row["tags"])]
            self.assertEqual(len(timing_rows), 1)
            reopened.close()


if __name__ == "__main__":
    unittest.main()
