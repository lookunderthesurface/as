from __future__ import annotations

from dataclasses import dataclass

from ..events.schema import NormalizedEvent


@dataclass(frozen=True)
class PrivacyDecision:
    blocked: bool
    reason: str


class PrivacyFilter:
    def __init__(self, excluded_apps: tuple[str, ...] = ("1Password", "KeePass")) -> None:
        self.excluded_apps = tuple(item.casefold() for item in excluded_apps if item.strip())

    def check(self, event: NormalizedEvent) -> PrivacyDecision:
        app = event.foreground_app.casefold()
        blocked = any(excluded in app for excluded in self.excluded_apps)
        return PrivacyDecision(blocked, "excluded app" if blocked else "allowed")

    @staticmethod
    def safe_suppression_metadata() -> dict[str, str]:
        return {"event_type": "privacy-suppressed", "app_category": "excluded", "content": "omitted"}

