"""Tiny direct JSON-schema completer for background dreaming.

Background consolidation needs free-form JSON, not the event/secretary
response schema that ``OllamaInferenceProvider`` enforces. This adapter calls
Ollama ``/api/chat`` with its own response format and returns raw text.
It is transport-minimal, offline-testable, and never used in the realtime
perception path.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

from urllib.error import HTTPError, URLError

from .ollama import _stdlib_post


class CompletionError(RuntimeError):
    pass


CONSOLIDATION_FORMAT: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "knowledge": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "statement": {"type": "string"},
                    "confidence": {"type": "number"},
                    "source_episode_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["statement", "confidence", "source_episode_ids"],
            },
        },
        "episode_summary": {"type": "string"},
    },
    "required": ["knowledge", "episode_summary"],
}


class OllamaTextCompleter:
    """``complete(prompt) -> str`` against a local Ollama chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str,
        timeout_seconds: float = 120.0,
        keep_alive: str = "30m",
        temperature: float = 0.0,
        http_post: Callable[..., object] | None = None,
    ) -> None:
        self.endpoint = f"{base_url.rstrip('/')}/api/chat"
        self.model = model
        self.timeout_seconds = max(1.0, timeout_seconds)
        self.keep_alive = keep_alive
        self.temperature = temperature
        self.http_post = http_post or _stdlib_post

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You extract durable knowledge from desktop work episodes. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": CONSOLIDATION_FORMAT,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature},
        }
        try:
            response = self.http_post(self.endpoint, payload, self.timeout_seconds)
        except TypeError:
            response = self.http_post(self.endpoint, payload)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise CompletionError(f"ollama transport failed: {exc.__class__.__name__}") from exc
        if not isinstance(response, Mapping):
            raise CompletionError("response is not an object")
        message = response.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise CompletionError("empty completion")
        return content
