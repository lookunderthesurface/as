from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import monotonic
from collections.abc import Callable

from .schema import InferenceRequest


@dataclass(frozen=True)
class ScheduledRequest:
    request: InferenceRequest
    submitted_at: float


class InferenceScheduler:
    """Latest-state-wins admission control for potentially slow inference.

    The scheduler is deliberately transport-agnostic. The owner calls
    ``start_next`` before invoking a provider and ``complete`` afterwards, so a
    slow provider can run in a dedicated worker without allowing a stale queue
    to grow behind it.
    """

    def __init__(
        self,
        min_interval_seconds: float = 10.0,
        max_pending_requests: int = 1,
        stale_request_seconds: float = 30.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self.max_pending_requests = max(1, max_pending_requests)
        self.stale_request_seconds = max(0.0, stale_request_seconds)
        self.clock = clock
        self._pending: ScheduledRequest | None = None
        self._running: ScheduledRequest | None = None
        self._last_started_at: float | None = None
        self.discarded_stale_requests = 0
        self._lock = Lock()

    def submit(self, request: InferenceRequest, now: float | None = None) -> None:
        submitted_at = self.clock() if now is None else now
        # max_pending_requests is intentionally bounded to one: replacing the
        # pending request is the latest-state-wins behavior.
        with self._lock:
            self._pending = ScheduledRequest(request, submitted_at)

    def start_next(self, now: float | None = None) -> InferenceRequest | None:
        with self._lock:
            if self._running is not None or self._pending is None:
                return None
            current = self.clock() if now is None else now
            if self._last_started_at is not None and current - self._last_started_at < self.min_interval_seconds:
                return None
            if current - self._pending.submitted_at > self.stale_request_seconds:
                self._pending = None
                self.discarded_stale_requests += 1
                return None
            self._running = self._pending
            self._pending = None
            self._last_started_at = current
            return self._running.request

    def complete(self) -> None:
        with self._lock:
            self._running = None

    def fail(self) -> None:
        """Release the running slot without retrying the failed request."""
        with self._lock:
            self._running = None

    def cancel_pending(self) -> None:
        """Drop queued work when the session is paused or shutting down."""
        with self._lock:
            self._pending = None

    def discard_stale(self, now: float | None = None) -> bool:
        with self._lock:
            if self._pending is None:
                return False
            current = self.clock() if now is None else now
            if current - self._pending.submitted_at <= self.stale_request_seconds:
                return False
            self._pending = None
            self.discarded_stale_requests += 1
            return True

    def next_wait_seconds(self, now: float | None = None) -> float | None:
        """Return the delay before a pending request may be admitted.

        ``None`` means there is no pending request or an inference is already
        running.  A stale pending request returns zero so the caller can let
        ``start_next`` discard it without sleeping.
        """
        with self._lock:
            if self._running is not None or self._pending is None:
                return None
            current = self.clock() if now is None else now
            if current - self._pending.submitted_at > self.stale_request_seconds:
                return 0.0
            if self._last_started_at is None:
                return 0.0
            return max(0.0, self.min_interval_seconds - (current - self._last_started_at))

    @property
    def pending_request(self) -> InferenceRequest | None:
        with self._lock:
            return self._pending.request if self._pending else None

    @property
    def running_request(self) -> InferenceRequest | None:
        with self._lock:
            return self._running.request if self._running else None

    @property
    def status(self) -> str:
        with self._lock:
            if self._running is not None:
                return "BUSY"
            if self._pending is not None:
                return "PENDING"
            return "READY"
