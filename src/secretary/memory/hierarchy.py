"""Memory hierarchy vocabulary (Letta-inspired, SQLite-backed).

One lexicon, one semantics:
- CORE: always-in-context small block (preferences + durable project facts)
- WORKING_RECALL: transient/in-memory recall (mostly engine state, not rows)
- EPISODIC: what happened (summarized episodes, not raw OCR)
- SEMANTIC: stable durable knowledge (facts/conclusions/workflows)
- INTERVENTION: what was worth reminding, what the user disliked
"""

from __future__ import annotations

from enum import Enum


class MemoryTier(str, Enum):
    CORE = "CORE"
    WORKING_RECALL = "WORKING_RECALL"
    EPISODIC = "EPISODIC"
    SEMANTIC = "SEMANTIC"
    INTERVENTION = "INTERVENTION"


class MemorySource(str, Enum):
    SYSTEM_DEFAULT = "SYSTEM_DEFAULT"
    EXPLICIT_USER = "EXPLICIT_USER"
    OBSERVED_EVENT = "OBSERVED_EVENT"
    OBSERVED_OUTCOME = "OBSERVED_OUTCOME"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    CONSOLIDATED = "CONSOLIDATED"


class MemoryStatus(str, Enum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    EXPIRED = "EXPIRED"


# Core memory must remain tiny. Exceeding this warns (memory-doctor).
CORE_MEMORY_CHARS_LIMIT = 800
# Retrieved memory budget for one prompt.
CONTEXT_MEMORY_CHARS_BUDGET = 1800
CONTEXT_EPISODE_BUDGET = 4

# Relative source weights for supersession and ranking.
SOURCE_WEIGHT = {
    MemorySource.EXPLICIT_USER.value: 1.0,
    MemorySource.CONSOLIDATED.value: 0.9,
    MemorySource.SYSTEM_DEFAULT.value: 0.7,
    MemorySource.OBSERVED_OUTCOME.value: 0.6,
    MemorySource.OBSERVED_EVENT.value: 0.4,
    MemorySource.MODEL_INFERENCE.value: 0.2,
}

# A model guess must not outrank explicitly stated user preferences.
MODEL_INFERENCE_MAX_CONFIDENCE = 0.6
