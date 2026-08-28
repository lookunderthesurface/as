"""Ollama adapter for GUI perception (screen screenshot + grounding text).

Reuses the already-configured ``OllamaInferenceProvider`` transport and the
same local vision model.  The point of this adapter is a *different* prompt
and *different* structured schema from the event-extraction request: the
model describes the GUI, not the desktop event.  It shares the model, not the
request shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import monotonic
from typing import Mapping

from ..events.schema import NormalizedEvent
from ..inference.ollama import OllamaInferenceProvider
from ..inference.image import ImagePreprocessor
from .perception import (
    GUI_RESPONSE_SCHEMA,
    GUIPerceptionOutput,
    GUIPerceptionProvider,
    GUIPerceptionRequest,
    render_gui_perception_prompt,
)
from .state import SemanticGUIState


@dataclass(frozen=True)
class GUIPerceptionMetrics:
    wall_latency_ms: float | None = None
    mode: str = "vision"

    def as_dict(self) -> dict[str, object]:
        return {"wall_latency_ms": self.wall_latency_ms, "mode": self.mode}


class GUIPerceptionOllamaProvider:
    """Server-side adapter around the local Ollama endpoint."""

    name = "ollama-gui"

    def __init__(self, ollama: OllamaInferenceProvider, image_preprocessor: ImagePreprocessor | None = None) -> None:
        self._ollama = ollama
        self.image_preprocessor = image_preprocessor or ollama.image_preprocessor

    @property
    def model(self) -> str | None:
        return getattr(self._ollama, "vision_model", None) or getattr(self._ollama, "model", None)

    def perceive(self, request: GUIPerceptionRequest) -> GUIPerceptionOutput:
        started = monotonic()
        model = self.model
        if not model:
            return GUIPerceptionOutput.safe(provider=self.name, model=None, error_type="model_not_configured")
        image: str | None = None
        if request.event.image_path:
            prepared = self.image_preprocessor.prepare_image(request.event.image_path)
            if prepared is not None:
                image = prepared.data
        prompt = render_gui_perception_prompt(request)
        payload: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an ambient desktop observer. Return only structured JSON."},
                {"role": "user", "content": prompt, **({"images": [image]} if image else {})},
            ],
            "stream": False,
            "format": GUI_RESPONSE_SCHEMA,
            "keep_alive": getattr(self._ollama, "keep_alive", "30m"),
            "options": {"temperature": getattr(self._ollama, "temperature", 0.0)},
        }
        try:
            raw = self._ollama.http_post(self._ollama.endpoint, payload, self._ollama.timeout_seconds)
        except TypeError:
            raw = self._ollama.http_post(self._ollama.endpoint, payload)
        except Exception as exc:  # transport errors observed by offline tests
            return GUIPerceptionOutput(
                state=SemanticGUIState(timestamp=_aware(datetime.now(timezone.utc))),
                provider=self.name,
                model=model,
                error_type="transport_error",
            )
        if not isinstance(raw, Mapping):
            return GUIPerceptionOutput(
                state=SemanticGUIState(timestamp=_aware(datetime.now(timezone.utc))),
                provider=self.name,
                model=model,
                error_type="malformed_response",
            )
        parsed = _extract_content(raw)
        if parsed is None:
            return GUIPerceptionOutput(
                state=SemanticGUIState(timestamp=_aware(datetime.now(timezone.utc))),
                provider=self.name,
                model=model,
                error_type="malformed_response",
            )
        state = SemanticGUIState.from_mapping(
            parsed,
            timestamp=_aware(datetime.now(timezone.utc)),
        )
        return GUIPerceptionOutput(
            state=state,
            provider=self.name,
            model=model,
            error_type=None,
        )


def _extract_content(response: Mapping[str, object]) -> Mapping[str, object] | None:
    message = response.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except ValueError:
                return None
            return parsed if isinstance(parsed, Mapping) else None
        if isinstance(content, Mapping):
            return content
        return None
    if "application" in response or "activity" in response:
        return response
    return None


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
