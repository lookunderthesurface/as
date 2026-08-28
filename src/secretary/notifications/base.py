from __future__ import annotations

from typing import Protocol


class NotificationProvider(Protocol):
    def notify(self, title: str, body: str) -> None:
        ...

