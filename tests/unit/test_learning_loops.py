from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from secretary.config import SecretaryConfig
from secretary.engine import SecretaryEngine
from secretary.memory.hierarchy import MemorySource, MemoryTier
from secretary.memory.retrieval import retrieve_relevant_knowledge
from secretary.memory.store import MemoryStore
from secretary.policy.critic import InterventionCritic
from secretary.policy.context import DecisionContext
from secretary.policy.watch import WatchManager
from secretary.perception.extractor import ExtractedEvent
from secretary.inference.schema import Action


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


def extracted(*, failure: str | None = "test-failure:python", action: Action = Action.NOTIFY, confidence: float = 0.9) -> ExtractedEvent:
    return ExtractedEvent(
        timestamp=NOW,
        event_type="failure" if failure else "coding",
        activity="terminal" if failure else "editor",
        app="WindowsTerminal.exe",
        summary="bounded summary",
        importance=0.9,
        novelty=0.5,
        confidence=confidence,
        failure_signature=failure,
        topic=failure or "pytest",
        candidate_action=action,
        interrupt_score=0.9,
    )


class KnowledgeRetrievalTests(unittest.TestCase):
    def test_same_issue_retrieves_prior_knowledge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            store.record_memory(
                "KV placement previously resolved this class of test-failure:python attention inconsistency.",
                source=MemorySource.CONSOLIDATED,
                importance=0.95,
                tier=MemoryTier.SEMANTIC,
                confidence=0.9,
                tags="knowledge|test-failure:python",
            )
            results = retrieve_relevant_knowledge(store, extracted())
            self.assertTrue(results)
            self.assertIn("KV placement", results[0]["content"])
            store.close()

    def test_unrelated_issue_does_not_retrieve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            store.record_memory(
                "KV placement previously resolved the attention inconsistency.",
                source=MemorySource.CONSOLIDATED,
                importance=0.95,
                tier=MemoryTier.SEMANTIC,
                confidence=0.9,
            )
            other = extracted(failure="npm-failure:node")
            other = ExtractedEvent(
                timestamp=NOW,
                event_type="failure",
                activity="terminal",
                app="WindowsTerminal.exe",
                summary="bounded summary",
                importance=0.9,
                novelty=0.5,
                confidence=0.9,
                failure_signature="npm-failure:node",
                topic="unrelated packaging",
                candidate_action=Action.NOTIFY,
                interrupt_score=0.9,
            )
            self.assertEqual(retrieve_relevant_knowledge(store, other), [])
            store.close()

    def test_superseded_knowledge_is_not_retrieved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            old = store.record_memory(
                "Timing changes fix test-failure:python.",
                source=MemorySource.CONSOLIDATED,
                tier=MemoryTier.SEMANTIC,
                confidence=0.6,
            )
            store.supersede_memory(old)
            self.assertEqual(retrieve_relevant_knowledge(store, extracted()), [])
            store.close()


class MemoryChangesSuggestionTests(unittest.TestCase):
    """Acceptance: retrieved memory must make the suggestion concrete."""

    def make_engine(self, root: Path) -> SecretaryEngine:
        return SecretaryEngine(SecretaryConfig(
            project_root=root,
            database_path=root / "data" / "state.db",
            log_directory=root / "logs",
            inference_provider="mock",
            cloud_provider="mock",
        ))

    def test_retrieved_knowledge_enriches_repeated_failure_suggestion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = self.make_engine(root)
            try:
                engine.store.record_memory(
                    "KV placement previously resolved this class of test-failure:python attention inconsistency.",
                    source=MemorySource.CONSOLIDATED,
                    importance=0.95,
                    tier=MemoryTier.SEMANTIC,
                    confidence=0.9,
                )
                for minute in range(5):
                    engine.process({
                        "timestamp": f"2026-08-28T12:0{minute}:00Z",
                        "foreground_app": "WindowsTerminal.exe",
                        "window_title": "PowerShell",
                        "event_source": "command",
                        "text": "pytest FAILED",
                    })
                notified = [
                    trace for trace in engine.store.recent_decision_traces(20)
                    if trace["final_action"] in {"NOTIFY", "WOULD_NOTIFY"}
                ]
                self.assertTrue(notified, "repeated failures should reach notification")
                episode = engine.store.recent_intervention_episodes(5)[0]
                self.assertIn("RELATED_PAST_SOLUTION_AVAILABLE", episode["reason_codes"])
            finally:
                engine.close()

    def test_recent_rejection_lets_critic_silence_model_notify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = self.make_engine(root)
            try:
                # Two similar past interventions the user explicitly disliked.
                for minute in (0, 1):
                    episode_id = engine.store.record_intervention_episode(
                        session_id=engine.session_id,
                        event_timestamp=NOW - timedelta(minutes=10 - minute),
                        situation_type="debugging",
                        activity="terminal",
                        event_type="failure",
                        topic="test-failure:python",
                        failure_signature="test-failure:python",
                        candidate_action="NOTIFY",
                        final_action="WOULD_NOTIFY",
                    )
                    engine.store.record_intervention_feedback(episode_id, "dont-remind")
                # Fresh watch evidence so the model candidate has support.
                for minute in range(2, 5):
                    engine.process({
                        "timestamp": f"2026-08-28T12:0{minute}:00Z",
                        "foreground_app": "WindowsTerminal.exe",
                        "window_title": "PowerShell",
                        "event_source": "command",
                        "text": "pytest FAILED",
                    })
                decisions = engine.store.recent_decision_traces(10)
                self.assertFalse(
                    any(trace["final_action"] in {"NOTIFY", "WOULD_NOTIFY"} for trace in decisions),
                    "explicitly rejected similar interventions must not re-notify",
                )
            finally:
                engine.close()


class DimensionalFeedbackTests(unittest.TestCase):
    def test_timing_and_content_learn_into_separate_memories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
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
            result = store.record_dimensional_feedback(
                episode_id,
                timing="too-early",
                content="too-generic",
            )
            self.assertEqual(result["timing"], "TOO_EARLY")
            self.assertEqual(result["content"], "TOO_GENERIC")
            self.assertEqual(len(result["knowledge_memory_ids"]), 2)
            memories = store.active_memories(tier=MemoryTier.SEMANTIC)
            timing_rows = [row for row in memories if "timing-knowledge" in str(row["tags"])]
            content_rows = [row for row in memories if "content-knowledge" in str(row["tags"])]
            self.assertEqual(len(timing_rows), 1)
            self.assertEqual(len(content_rows), 1)
            self.assertIn("wait for stronger", timing_rows[0]["content"])
            self.assertIn("generic advice", content_rows[0]["content"])
            row = store.get_intervention_episode(episode_id)
            self.assertEqual(row["timing_feedback"], "TOO_EARLY")
            self.assertEqual(row["content_feedback"], "TOO_GENERIC")
            store.close()


class WatchReadinessTests(unittest.TestCase):
    def test_readiness_grows_with_evidence_research_and_memory(self) -> None:
        watch = WatchManager()
        base = extracted(action=Action.WATCH)
        watch.observe_failure(base, NOW, "first failure")
        hypothesis = watch.active
        self.assertIsNotNone(hypothesis)
        low = hypothesis.intervention_readiness()
        watch.observe_failure(base, NOW + timedelta(minutes=5), "second")
        watch.observe_failure(base, NOW + timedelta(minutes=10), "third")
        docs = ExtractedEvent(
            timestamp=NOW + timedelta(minutes=12),
            event_type="documentation",
            activity="research",
            app="Chrome.exe",
            summary="reading docs",
            importance=0.4,
            novelty=0.4,
            confidence=0.9,
            topic="test-failure:python",
            candidate_action=Action.WATCH,
        )
        watch.observe_related(docs, NOW + timedelta(minutes=12))
        mid = hypothesis.intervention_readiness()
        high = hypothesis.intervention_readiness(memory_support=1.0)
        self.assertGreater(mid, low)
        self.assertGreater(high, mid)
        self.assertLessEqual(high, 1.0)
        snapshot = watch.snapshot()[0]
        self.assertIn("readiness", snapshot)
        self.assertIn("research_seen", snapshot)

    def test_critic_silent_on_strong_negative_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            for minute in range(3):
                episode_id = store.record_intervention_episode(
                    session_id=None,
                    event_timestamp=NOW - timedelta(minutes=30 - minute),
                    situation_type="debugging",
                    activity="terminal",
                    event_type="failure",
                    topic="test-failure:python",
                    failure_signature="test-failure:python",
                    candidate_action="NOTIFY",
                    final_action="WOULD_NOTIFY",
                )
                store.record_intervention_feedback(episode_id, "dont-remind")
            from secretary.memory.retrieval import retrieve_similar_intervention_episodes

            context = DecisionContext(
                event=extracted(),
                working_state=__import__("secretary.context.working_state", fromlist=["WorkingState"]).WorkingState(),
                now=NOW,
                failure_count=4,
                similar_episodes=tuple(retrieve_similar_intervention_episodes(store, extracted(), limit=3)),
            )
            critique = InterventionCritic().critique(context, watch_evidence=4)
            self.assertEqual(critique.recommendation, "SILENT")
            self.assertIn("RECENT_SIMILAR_REJECTION", critique.reasons)
            store.close()


if __name__ == "__main__":
    unittest.main()
