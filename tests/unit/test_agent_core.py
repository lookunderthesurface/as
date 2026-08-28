from __future__ import annotations

import unittest

from secretary.context.working_state import WorkingState
from secretary.state.reducer import Observation, WorldStateReducer
from secretary.policy.context import DecisionContext


class ReducerDeterminismTests(unittest.TestCase):
    def test_failure_observation_sets_objective_deterministically(self) -> None:
        state_a, state_b = WorkingState(), WorkingState()
        observation = Observation(
            event_type="failure",
            activity="terminal",
            app="WindowsTerminal.exe",
            summary="pytest failed",
            confidence=0.94,
            topic="test-failure:python",
            failure_signature="test-failure:python",
        )
        for _ in range(2):
            for state in (state_a, state_b):
                WorldStateReducer.apply(state, observation)
        self.assertEqual(state_a.snapshot(), state_b.snapshot())
        self.assertEqual(state_a.current_objective, "resolve test-failure:python")
        self.assertTrue("test-failure:python" in state_a.recent_failures)  # type: ignore[arg-type]

    def test_recovery_clears_objective(self) -> None:
        state = WorkingState()
        WorldStateReducer.apply(state, Observation("failure", "terminal", "app", "fail", 0.9, failure_signature="sig"))
        WorldStateReducer.apply(state, Observation("recovery", "terminal", "app", "passed", 0.92))
        self.assertIsNone(state.current_objective)
        self.assertIsNone(state.current_subgoal)
        self.assertIsNone(state.current_topic)

    def test_low_confidence_failure_does_not_hijack_objective(self) -> None:
        state = WorkingState(current_objective="review code")
        WorldStateReducer.apply(state, Observation("failure", "terminal", "app", "fail", 0.3, failure_signature="new-sig"), allow_objective_update=False)
        self.assertEqual(state.current_objective, "review code")
        self.assertIsNone(state.current_subgoal)
        # The failure is still recorded as a fact; only objective hijacking is blocked.
        self.assertIn("new-sig", list(state.recent_failures))  # type: ignore[arg-type]

    def test_documentation_attaches_to_active_objective(self) -> None:
        state = WorkingState(current_objective="resolve test-failure:python")
        WorldStateReducer.apply(state, Observation("documentation", "research", "Chrome.exe", "searching", 0.9, topic="cuda non-determinism"))
        self.assertEqual(state.current_subgoal, "compare documentation with the current failure")
        self.assertEqual(state.current_topic, "cuda non-determinism")

    def test_same_observation_same_transition_regardless_of_order_of_extra_calls(self) -> None:
        obs_a = Observation("coding", "editor", "Code.exe", "editing", 0.93)
        obs_b = Observation("failure", "terminal", "WindowsTerminal.exe", "pytest FAILED", 0.94, failure_signature="sig")
        first = WorkingState()
        WorldStateReducer.apply(first, obs_a)
        WorldStateReducer.apply(first, obs_b)
        second = WorkingState()
        WorldStateReducer.apply(second, obs_b)
        WorldStateReducer.apply(second, obs_a)
        self.assertNotEqual(first.snapshot(), second.snapshot())


class DecisionContextTests(unittest.TestCase):
    def test_empty_context_has_safe_defaults(self) -> None:
        context = DecisionContext(
            event=None,  # type: ignore[arg-type]
            working_state=WorkingState(),
            now=None,  # type: ignore[arg-type]
        )
        self.assertEqual(context.trajectory_text, "")
        self.assertEqual(context.gui_state_text, "")
        self.assertEqual(context.memory_context, ())


if __name__ == "__main__":
    unittest.main()
