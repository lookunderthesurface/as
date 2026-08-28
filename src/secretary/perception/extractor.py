from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..events.schema import NormalizedEvent
from ..inference.base import InferenceProvider
from ..inference.mock import MockInferenceProvider
from ..inference.schema import Action, InferenceRequest, InferenceResult, validate_inference_result


@dataclass(frozen=True)
class ExtractedEvent:
    timestamp: datetime
    event_type: str
    activity: str
    app: str
    summary: str
    importance: float
    novelty: float
    confidence: float
    failure_signature: str | None = None
    topic: str | None = None
    candidate_action: Action = Action.IGNORE
    interrupt_score: float = 0.0
    candidate_reason: str = "No high-value intervention is indicated"


class EventExtractor:
    """Provider-neutral adapter: meaning is supplied by the inference boundary."""

    def __init__(self, inference: InferenceProvider | None = None) -> None:
        self.inference = inference or MockInferenceProvider()
        self.last_result: InferenceResult | None = None

    def extract(self, event: NormalizedEvent, request: InferenceRequest | None = None) -> ExtractedEvent:
        inference_request = request or InferenceRequest(current_event=event)
        provider_name = getattr(self.inference, "name", self.inference.__class__.__name__)
        provider_model = getattr(self.inference, "model", None)
        try:
            raw_result = self.inference.analyze(inference_request)
            result = validate_inference_result(raw_result, provider=provider_name, model=provider_model)
        except Exception:
            result = InferenceResult.safe("provider_error", provider=provider_name, model=provider_model)
        self.last_result = result
        event_result = result.event
        return ExtractedEvent(
            timestamp=event.timestamp,
            event_type=event_result.event_type,
            activity=event_result.activity,
            app=event.foreground_app,
            summary=event_result.summary,
            importance=event_result.importance,
            novelty=event_result.novelty,
            confidence=event_result.confidence,
            failure_signature=event_result.failure_signature,
            topic=event_result.topic,
            candidate_action=result.secretary.candidate_action,
            interrupt_score=result.secretary.interrupt_score,
            candidate_reason=result.secretary.reason,
        )


# Kept as a compatibility import for early callers; the mock belongs to the
# inference provider, not to the capture boundary.
MockEventExtractor = EventExtractor
