from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ShadowNotificationProvider:
    """Record notification attempts without invoking a desktop notification API."""

    shadow: bool = True
    notifications: list[tuple[str, str]] = field(default_factory=list)

    def notify(self, title: str, body: str) -> None:
        self.notifications.append((title[:200], body[:1000]))

