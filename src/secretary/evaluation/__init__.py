"""Proactive evaluation and memory diagnostics CLI entry points."""

from .matrix import EvaluationMatrix, evaluate_labels, evaluate_scenario
from .proactive_bench import ProactiveBenchItem, load_bench_items, run_proactive_bench

__all__ = [
    "EvaluationMatrix",
    "evaluate_labels",
    "evaluate_scenario",
    "ProactiveBenchItem",
    "load_bench_items",
    "run_proactive_bench",
]
