from __future__ import annotations

from collections.abc import Iterable, Mapping


class MockCaptureProvider:
    def __init__(self, items: Iterable[Mapping[str, object]] = ()) -> None:
        self.items = [dict(item) for item in items]
        self.index = 0

    def poll(self) -> list[Mapping[str, object]]:
        if self.index >= len(self.items):
            return []
        result = self.items[self.index :]
        self.index = len(self.items)
        return result

    def health(self) -> bool:
        return True

