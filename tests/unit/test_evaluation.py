from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from secretary.evaluation.matrix import evaluate_labels, evaluate_scenario
from secretary.evaluation.proactive_bench import ProactiveBenchItem, run_proactive_bench
from secretary.memory.doctor import diagnose
from secretary.memory.hierarchy import MemorySource, MemoryTier
from secretary.memory.store import MemoryStore


NOW = datetime(2026, 8, 28, 12, 0)


class EvaluationMatrixTests(unittest.TestCase):
    def test_labels_make_matrix(self) -> None:
        matrix = evaluate_labels(["useful", "not-useful", "not-needed", "needed-but-bad-timing"])
        self.assertEqual(matrix.labeled_opportunities, 4)
        self.assertEqual(matrix.tp, 2)
        self.assertEqual(matrix.fp, 2)
        self.assertEqual(matrix.fn, 0)
        self.assertAlmostEqual(matrix.precision, 0.5)
        self.assertIsNotNone(matrix.precision)

    def test_no_labels_returns_zeroes_with_safe_none(self) -> None:
        matrix = evaluate_labels([])
        self.assertEqual(matrix.labeled_opportunities, 0)
        self.assertFalse(matrix.has_ground_truth)
        self.assertIsNone(matrix.precision)
        self.assertIsNone(matrix.recall)

    def test_scenario_matrix_uses_ground_truth(self) -> None:
        interventions = [
            {"final_action": "NOTIFY"},
            {"final_action": "IGNORE"},
            {"final_action": "IGNORE"},
            {"final_action": "WOULD_NOTIFY"},
        ]
        ground_truth = [
            {"index": 0, "needed": True},
            {"index": 1, "needed": True},
            {"index": 2, "needed": False},
            {"index": 3, "needed": False},
        ]
        result = evaluate_scenario(interventions=interventions, ground_truth=ground_truth)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["tn"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)


class ProactiveBenchAdapterTests(unittest.TestCase):
    def test_adapter_is_runnable_without_engine(self) -> None:
        items = [
            ProactiveBenchItem.from_mapping({"timestamp": "t0", "event_type": "failure", "activity": "debugging", "app": "terminal", "text": "pytest failed", "ground_truth": {"needed": True}}),
            ProactiveBenchItem.from_mapping({"timestamp": "t1", "event_type": "coding", "activity": "editing", "app": "vscode", "text": "typing", "ground_truth": {"needed": False}}),
        ]
        result = run_proactive_bench(items)
        self.assertEqual(result["labeled_opportunities"], 2)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 0)


class MemoryDoctorTests(unittest.TestCase):
    def test_doctor_dry_run_does_not_mutate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            # Large core memory should be flagged, not fixed.
            store.record_memory("secretary: " * 200, source=MemorySource.EXPLICIT_USER, tier=MemoryTier.CORE, importance=1.0)
            before = store.connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            report = diagnose(store)
            after = store.connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            self.assertEqual(before, after)
            self.assertGreater(report.core_chars, report.core_budget)
            self.assertEqual(report.issues, 1)
            store.close()

    def test_doctor_flags_preference_flip_flop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            # Same scope: dont-remind -> more-proactive -> dont-remind.
            for index in range(5):
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
                value = "dont-remind" if index % 2 == 0 else "more-proactive"
                store.record_intervention_feedback(episode_id, value)
            report = diagnose(store)
            kinds = [finding.kind for finding in report.findings]
            self.assertIn("preference_flip_flop", kinds)
            store.close()

    def test_doctor_detects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "state.db")
            store.record_memory("same fact twice", source="user")
            store.record_memory("same fact twice", source="user")
            report = diagnose(store)
            self.assertTrue(any(finding.kind == "duplicate_content" for finding in report.findings))
            store.close()


if __name__ == "__main__":
    unittest.main()
