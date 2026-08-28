from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..perception.extractor import ExtractedEvent


def _clean(value: str, limit: int) -> str:
    return re.sub(r"\s+", " ", value).strip()[:limit]


@dataclass
class WatchHypothesis:
    """A bounded, expiring observation hypothesis, not an instruction."""

    signature: str
    hypothesis: str
    evidence: int
    created_at: datetime
    expires_at: datetime
    last_reason: str
    updated_at: datetime | None = None

    @property
    def reason(self) -> str:
        return self.last_reason

    @property
    def watch_id(self) -> str:
        return self.signature

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.watch_id,
            "signature": self.signature,
            "hypothesis": self.hypothesis,
            "evidence": self.evidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": (self.updated_at or self.created_at).isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "reason": self.reason,
            "last_reason": self.last_reason,
        }


class WatchManager:
    def __init__(self, expiration_minutes: int = 20, max_active_hypotheses: int = 3) -> None:
        self.expiration_minutes = max(1, expiration_minutes)
        self.max_active_hypotheses = max(1, max_active_hypotheses)
        self._hypotheses: list[WatchHypothesis] = []
        self._active_signature: str | None = None

    @property
    def active(self) -> WatchHypothesis | None:
        if self._active_signature is not None:
            for hypothesis in self._hypotheses:
                if hypothesis.signature == self._active_signature:
                    return hypothesis
        return self._hypotheses[-1] if self._hypotheses else None

    def expire(self, now: datetime) -> bool:
        before = len(self._hypotheses)
        self._hypotheses = [item for item in self._hypotheses if now < item.expires_at]
        expired = len(self._hypotheses) != before
        if self._active_signature and not any(item.signature == self._active_signature for item in self._hypotheses):
            self._active_signature = self._hypotheses[-1].signature if self._hypotheses else None
        return expired

    def _find(self, signature: str) -> WatchHypothesis | None:
        return next((item for item in self._hypotheses if item.signature == signature), None)

    def _touch(self, hypothesis: WatchHypothesis, now: datetime, reason: str) -> None:
        hypothesis.evidence += 1
        hypothesis.last_reason = _clean(reason, 300) or "additional related evidence"
        hypothesis.updated_at = now
        hypothesis.expires_at = now + timedelta(minutes=self.expiration_minutes)
        self._active_signature = hypothesis.signature

    def _create(
        self,
        signature: str,
        hypothesis_text: str,
        now: datetime,
        reason: str,
    ) -> WatchHypothesis | None:
        if len(self._hypotheses) >= self.max_active_hypotheses:
            return None
        item = WatchHypothesis(
            signature=_clean(signature, 180),
            hypothesis=_clean(hypothesis_text, 500) or "A potentially useful work pattern is being observed.",
            evidence=0,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(minutes=self.expiration_minutes),
            last_reason=_clean(reason, 300) or "watch created",
        )
        self._hypotheses.append(item)
        self._active_signature = item.signature
        return item

    def observe_failure(self, event: ExtractedEvent, now: datetime, reason: str = "repeated failure") -> int:
        if not event.failure_signature:
            return 0
        self.expire(now)
        # Preserve the original deterministic failure signature for baseline
        # callers and use prefixed keys only for generic model hypotheses.
        signature = event.failure_signature
        item = self._find(signature)
        if item is None:
            item = self._create(
                signature,
                "User may be stuck debugging the same issue.",
                now,
                reason,
            )
        if item is None:
            return 0
        self._touch(item, now, reason)
        return 1

    def observe_model(
        self,
        event: ExtractedEvent,
        now: datetime,
        reason: str,
        hypothesis_text: str | None = None,
    ) -> WatchHypothesis | None:
        """Create or merge one generic model hypothesis within the active cap."""
        self.expire(now)
        topic = _clean(event.topic or "", 100).casefold()
        if event.failure_signature:
            signature = event.failure_signature
            text = "User may be stuck debugging the same issue."
        elif topic:
            signature = f"topic:{topic}"
            text = "User may be repeatedly investigating the same topic."
        else:
            signature = f"model:{event.event_type}:{_clean(event.app, 80).casefold()}"
            text = hypothesis_text or "User may be switching repeatedly without progress."
        item = self._find(signature)
        if item is None:
            item = self._create(signature, hypothesis_text or text, now, reason)
            if item is None:
                return None
        self._touch(item, now, reason)
        return item

    def observe_related(self, event: ExtractedEvent, now: datetime) -> int:
        self.expire(now)
        item = self.active
        if item is None or event.event_type != "documentation":
            return 0
        self._touch(item, now, "related documentation search")
        return 1

    def snapshot(self) -> list[dict[str, object]]:
        return [item.as_dict() for item in self._hypotheses]
