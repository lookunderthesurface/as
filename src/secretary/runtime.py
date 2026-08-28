from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from threading import Lock
from typing import Mapping


COUNTER_NAMES = (
    "raw_screenpipe_items",
    "normalized_events",
    "privacy_filtered_events",
    "duplicate_events_dropped",
    "coalesced_batches",
    "inference_submitted",
    "inference_replaced",
    "inference_stale_request_dropped",
    "inference_results_received",
    "inference_results_stale_discarded",
    "text_inference_calls",
    "vision_inference_calls",
    "structured_output_failures",
    "provider_failures",
    "policy_ignore",
    "policy_remember",
    "policy_watch",
    "policy_investigate",
    "policy_ask_cloud",
    "policy_notify_candidate",
    "would_notify",
    "real_notify",
)


@dataclass
class RuntimeCounters:
    """Small lock-protected session funnel; no per-second telemetry rows."""

    _values: Counter[str] = field(default_factory=Counter, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def increment(self, name: str, amount: int = 1) -> int:
        if name not in COUNTER_NAMES:
            raise ValueError(f"unknown runtime counter: {name}")
        with self._lock:
            self._values[name] += amount
            return self._values[name]

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {name: int(self._values.get(name, 0)) for name in COUNTER_NAMES}

    def update(self, values: Mapping[str, int]) -> None:
        for name, amount in values.items():
            if name in COUNTER_NAMES and amount:
                self.increment(name, int(amount))

