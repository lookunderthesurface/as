from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from ..events.schema import sanitize_semantic_label


@dataclass(frozen=True)
class SecretaryProfile:
    """A compact, inspectable projection of active intervention preferences."""

    general: str
    rules: tuple[str, ...] = ()
    source_count: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "general": self.general,
            "rules": list(self.rules),
            "source_count": self.source_count,
        }


def build_secretary_profile(preferences: Sequence[Mapping[str, object]]) -> SecretaryProfile:
    rules: list[str] = []
    for preference in preferences:
        if str(preference.get("status", "ACTIVE")).upper() != "ACTIVE":
            continue
        source = str(preference.get("source") or "UNKNOWN").upper()
        try:
            confidence = float(preference.get("confidence", 0.0))
        except (TypeError, ValueError, OverflowError):
            confidence = 0.0
        if not math.isfinite(confidence):
            continue
        if source not in {"EXPLICIT_USER", "SYSTEM_DEFAULT"} and confidence < 0.8:
            continue
        content = sanitize_semantic_label(preference.get("content"), 500)
        if not content:
            continue
        source = str(preference.get("source") or "UNKNOWN")
        rules.append(f"{content} [source={source}, confidence={_confidence(preference.get('confidence')):.2f}]")
    return SecretaryProfile(
        general="User generally prefers low interruption.",
        rules=tuple(rules[:12]),
        source_count=len(rules),
    )


def _confidence(value: object) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError, OverflowError):
        return 0.0
