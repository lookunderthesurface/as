from __future__ import annotations

from typing import Protocol

from .schema import InferenceRequest, InferenceResult


class InferenceProvider(Protocol):
    name: str
    model: str | None

    def analyze(self, request: InferenceRequest) -> InferenceResult:
        ...

    def status(self):
        ...
