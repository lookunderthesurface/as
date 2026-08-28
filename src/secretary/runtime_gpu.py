"""GPU status provider: idle-aware compute budget without CUDA dependencies.

Ambient Secretary uses the user's GPU, it never fights the user for it.
This provider queries ``nvidia-smi`` (already installed with the driver —
no CUDA Toolkit, no PyTorch) and maps utilization+VRAM to a coarse budget:

    IDLE      (< 15% util)  -> perception + dreaming allowed
    NORMAL    (< 60% util)  -> normal perception
    BUSY      (< 90% util)  -> reduce background cognition, widen keyframe gaps
    CRITICAL  (>= 90% util) -> only necessary perception, dreaming paused

Failure semantics: if nvidia-smi is missing or errors, the provider reports
UNKNOWN and callers must treat it as NORMAL (never block the agent because
monitoring failed).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import Enum
from time import monotonic
from collections.abc import Callable


class GPUStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    IDLE = "IDLE"
    NORMAL = "NORMAL"
    BUSY = "BUSY"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class GPUSnapshot:
    status: GPUStatus
    utilization_percent: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "utilization_percent": self.utilization_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
        }


def _classify(utilization: float | None, memory_used: float | None, memory_total: float | None) -> GPUStatus:
    if utilization is None:
        return GPUStatus.UNKNOWN
    if utilization >= 90.0:
        return GPUStatus.CRITICAL
    if utilization >= 60.0:
        return GPUStatus.BUSY
    if utilization >= 15.0:
        return GPUStatus.NORMAL
    # Low utilization with a nearly full VRAM still means heavy resident work.
    if memory_total and memory_used and memory_used / memory_total >= 0.92:
        return GPUStatus.BUSY
    return GPUStatus.IDLE


class GPUStatusProvider:
    """Bounded, cached nvidia-smi reader (subprocess, no CUDA dependency)."""

    def __init__(self, cache_seconds: float = 10.0, runner: Callable[..., object] | None = None) -> None:
        self.cache_seconds = max(2.0, cache_seconds)
        self._runner = runner or self._run_nvidia_smi
        self._cached: GPUSnapshot = GPUSnapshot(status=GPUStatus.UNKNOWN)
        self._cached_at: float = 0.0

    @staticmethod
    def _run_nvidia_smi() -> str:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return output.stdout

    def snapshot(self, *, now: float | None = None) -> GPUSnapshot:
        current = monotonic() if now is None else now
        if current - self._cached_at < self.cache_seconds:
            return self._cached
        try:
            raw = str(self._runner()).strip().splitlines()
            line = raw[0] if raw else ""
            parts = [part.strip() for part in line.split(",")]
            utilization = float(parts[0]) if parts and parts[0] else None
            memory_used = float(parts[1]) if len(parts) > 1 and parts[1] else None
            memory_total = float(parts[2]) if len(parts) > 2 and parts[2] else None
            snapshot = GPUSnapshot(
                status=_classify(utilization, memory_used, memory_total),
                utilization_percent=utilization,
                memory_used_mb=memory_used,
                memory_total_mb=memory_total,
            )
        except Exception:
            # Monitoring must never break the agent; treat as UNKNOWN->NORMAL.
            snapshot = GPUSnapshot(status=GPUStatus.UNKNOWN)
        self._cached = snapshot
        self._cached_at = current
        return snapshot

    @staticmethod
    def min_visual_interval_for(status: GPUStatus, base_seconds: float) -> float:
        """Widen perception gaps under compute pressure instead of stopping."""
        if status == GPUStatus.CRITICAL:
            return base_seconds * 4.0
        if status == GPUStatus.BUSY:
            return base_seconds * 2.0
        return base_seconds

    @staticmethod
    def dreaming_allowed(status: GPUStatus) -> bool:
        return status not in {GPUStatus.CRITICAL}
