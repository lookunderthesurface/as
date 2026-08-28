from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MockNotificationProvider:
    notifications: list[tuple[str, str]] = field(default_factory=list)

    def notify(self, title: str, body: str) -> None:
        self.notifications.append((title, body))

