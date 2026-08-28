from __future__ import annotations


class MockCloudProvider:
    def investigate(self, reason: str) -> str:
        return "Mock cloud investigation deferred: " + reason[:200]

