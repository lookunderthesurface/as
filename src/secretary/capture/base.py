from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class CaptureProvider(Protocol):
    def poll(self) -> list[Mapping[str, object]]:
        ...

    def health(self) -> bool:
        ...

