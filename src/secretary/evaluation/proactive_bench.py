"""ProactiveBench-style adapter (interface + synthetic fixture, no runtime deps).

ProactiveAgent / ProactiveBench defines an evaluation methodology for
*proactive* assistance on activity traces: which moment did the agent need
to act, did it propose, did it time the proposal well, and did the user
accept. This adapter reuses that shape without importing their code,
reward model, or data: Ambient Secretary evaluates its own policy against
our own labelled desktop scenarios via ``evaluate_scenario``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .matrix import evaluate_scenario


@dataclass(frozen=True)
class ProactiveBenchItem:
    timestamp: str
    event_type: str
    activity: str
    app: str
    text: str
    ground_truth: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ProactiveBenchItem":
        return cls(
            timestamp=str(value.get("timestamp") or ""),
            event_type=str(value.get("event_type") or "activity"),
            activity=str(value.get("activity") or "desktop"),
            app=str(value.get("app") or value.get("foreground_app") or "unknown"),
            text=str(value.get("text") or ""),
            ground_truth=dict(value.get("ground_truth") or {}),
        )


def load_bench_items(path: Path) -> list[ProactiveBenchItem]:
    """Load a JSONL bench trace (synthetic or real ProactiveBench-like)."""
    items: list[ProactiveBenchItem] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            items.append(ProactiveBenchItem.from_mapping(value))
    return items


def run_proactive_bench(items: Sequence[ProactiveBenchItem], *, policy=None) -> dict[str, object]:
    """Evaluate a bench trace through the installed policy interface.

    The policy argument is optional for unit tests: a simple acceptance
    profile decides by the ``ground_truth`` label only when no policy is
    passed, so the adapter is runnable without engine wiring.
    """
    decisions: list[dict[str, object]] = []
    ground_truth: list[Mapping[str, object]] = []
    for index, item in enumerate(items):
        if policy is not None:
            result = policy(item)
            decisions.append(result)
        else:
            decisions.append({"final_action": "NOTIFY" if item.ground_truth.get("needed") else "IGNORE"})
        truth = dict(item.ground_truth)
        truth["index"] = index
        ground_truth.append(truth)
    return evaluate_scenario(interventions=decisions, ground_truth=ground_truth)
