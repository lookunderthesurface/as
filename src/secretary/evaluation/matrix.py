"""Proactive evaluation matrix (ProactiveAgent-inspired).

Ground truth is human labels only. This module never fabricates TP/FP:
with no labels it reports plain counts; with labels it computes
precision / recall / false-alarm / missed-need on the labeled subset.

Mapping (documented, deterministic):
- label USEFUL                    => intervention helped (TP)
- label NEEDED_BAD_TIMING         => needed, but timing wrong (FN for timing, not false alarm)
- label NOT_USEFUL / NOT_NEEDED   => not needed (FP)
- label UNSURE                    => excluded from metrics
- baseline silence when user needed help & no intervention happened => FN
  (only measurable where we have a "needed" ground truth, e.g. ProactiveBench
  traces; in live shadow mode we expose numbers but must not claim a rate
  for unlabelled episodes).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..memory.intervention import InterventionLabel


@dataclass(frozen=True)
class EvaluationMatrix:
    labeled_opportunities: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    unlabeled: int = 0

    @property
    def has_ground_truth(self) -> bool:
        return self.labeled_opportunities > 0

    @property
    def precision(self) -> float | None:
        if self.tp + self.fp == 0:
            return None
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        if self.tp + self.fn == 0:
            return None
        return self.tp / (self.tp + self.fn)

    @property
    def false_alarm_rate(self) -> float | None:
        """FP / (FP + TN) in the unaffected class; None without TN ground truth."""
        if self.fp + self.tn == 0:
            return None
        return self.fp / (self.fp + self.tn)

    @property
    def missed_need_rate(self) -> float | None:
        if self.tp + self.fn == 0:
            return None
        return self.fn / (self.tp + self.fn)

    def as_dict(self) -> dict[str, object]:
        return {
            "labeled_opportunities": self.labeled_opportunities,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "unlabeled": self.unlabeled,
            "precision": self.precision,
            "recall": self.recall,
            "false_alarm_rate": self.false_alarm_rate,
            "missed_need_rate": self.missed_need_rate,
            "has_ground_truth": self.has_ground_truth,
        }


def _label_key(value: str) -> str:
    return value.strip().upper().replace("-", "_").replace("_BUT_", "_").replace(" ", "_")


def evaluate_labels(labels: Sequence[str]) -> EvaluationMatrix:
    """Turn a label stream into an evaluation matrix (no need for engine data).

    Label semantics (documented):
    - USEFUL => proposal helped, wanted, needed (TP)
    - NEEDED_BAD_TIMING => needed help but timing was wrong (TP; timing=False)
    - NOT_USEFUL / NOT_NEEDED => proposal was not wanted (FP)
    - TN/FN require a baseline-silence counter-stream; only counted when present.
    """
    counts = Counter(_label_key(item) for item in labels if str(item).strip())
    tp = counts.get(InterventionLabel.USEFUL.value, 0) + counts.get(InterventionLabel.NEEDED_BAD_TIMING.value, 0)
    fp = counts.get(InterventionLabel.NOT_USEFUL.value, 0) + counts.get(InterventionLabel.NOT_NEEDED.value, 0)
    labeled = sum(counts.values())
    return EvaluationMatrix(
        labeled_opportunities=labeled,
        tp=tp,
        fp=fp,
        tn=counts.get("NONE", 0),
        fn=0,
        unlabeled=0,
    )


def evaluate_scenario(
    *,
    interventions: Sequence[Mapping[str, object]],
    ground_truth: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Deterministic adapter for a controlled scenario (AmbientBench).

    ``interventions``: decision rows (final_action etc.).
    ``ground_truth``: {index, needed: bool, useful: bool}.
    """
    tp = fp = tn = fn = 0
    needed_count = 0
    useful_count = 0
    for index, decision in enumerate(interventions):
        marker = next((item for item in (ground_truth or ()) if int(item.get("index", index)) == index), None)
        if marker is None:
            continue
        needed = bool(marker.get("needed", False))
        intervened = str(decision.get("final_action") or "IGNORE") in {"NOTIFY", "WOULD_NOTIFY", "INVESTIGATE"}
        if needed:
            needed_count += 1
        if needed and intervened:
            tp += 1
            useful_count += 1 if not marker.get("timing_bad") else 0
        elif needed and not intervened:
            fn += 1
        elif not needed and intervened:
            fp += 1
        elif not needed and not intervened:
            tn += 1
    matrix = EvaluationMatrix(
        labeled_opportunities=len(ground_truth or ()),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        unlabeled=len(interventions) - len(ground_truth or ()),
    )
    return {
        **matrix.as_dict(),
        "scenario_interventions": len(interventions),
        "needed_count": needed_count,
        "useful_count": useful_count,
        "source": "ground-truth-scenario",
    }
