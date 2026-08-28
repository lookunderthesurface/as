from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .metrics import InferenceMetrics


class InferenceRuntimeState(str, Enum):
    MOCK = "MOCK"
    STARTING = "STARTING"
    READY = "READY"
    BUSY = "BUSY"
    DEGRADED = "DEGRADED"
    NOT_CHECKED = "NOT_CHECKED"


@dataclass(frozen=True)
class LocalInferenceStatus:
    provider: str
    status: InferenceRuntimeState
    model: str | None = None
    last_mode: str | None = None
    last_latency_ms: float | None = None
    last_success_at: datetime | None = None
    last_error_type: str | None = None
    real_model_required: bool = False
    last_metrics: InferenceMetrics | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "status": self.status.value,
            "model": self.model,
            "last_mode": self.last_mode,
            "last_latency_ms": self.last_latency_ms,
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_error_type": self.last_error_type,
            "real_model_required": self.real_model_required,
            "last_metrics": self.last_metrics.as_dict() if self.last_metrics else None,
        }
