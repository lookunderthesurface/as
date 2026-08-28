from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SessionState:
    paused: bool = False

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

