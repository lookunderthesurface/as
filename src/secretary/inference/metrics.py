from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InferenceMetrics:
    """Non-sensitive timing and token counters returned by a local provider."""

    wall_latency_ms: float
    ollama_total_duration_ms: float | None = None
    model_load_ms: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    prompt_eval_ms: float | None = None
    generation_ms: float | None = None
    mode: str = "text"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
