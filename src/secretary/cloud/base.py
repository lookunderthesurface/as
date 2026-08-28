from __future__ import annotations

from typing import Protocol


class CloudProvider(Protocol):
    def investigate(self, reason: str) -> str:
        ...

