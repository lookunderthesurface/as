from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock, Thread, current_thread
from time import monotonic
from collections.abc import Iterable

from .capture.lifecycle import ScreenpipeLifecycleManager
from .engine import SecretaryEngine


@dataclass(frozen=True)
class ControllerStatus:
    capture_status: str
    worker_alive: bool
    owned_by_secretary: bool
    paused: bool
    pid: int | None = None
    error: str | None = None

    def __str__(self) -> str:
        detail = f" error={self.error}" if self.error else ""
        owner = "owned" if self.owned_by_secretary else "external/unmanaged"
        return f"capture_status={self.capture_status} worker_alive={self.worker_alive} {owner}{detail}"


class SecretaryController:
    """Own the worker loop while the caller owns the UI/event loop."""

    def __init__(
        self,
        engine: SecretaryEngine,
        capture: Iterable[dict[str, object]] | object,
        lifecycle: ScreenpipeLifecycleManager | None = None,
        poll_interval: float = 2.0,
        supervision_interval: float = 10.0,
    ) -> None:
        self.engine = engine
        self.capture = capture
        self.lifecycle = lifecycle
        self.poll_interval = max(0.05, poll_interval)
        self.supervision_interval = max(5.0, supervision_interval)
        self._stop_event = Event()
        self._resume_event = Event()
        self._resume_event.set()
        self._first_poll = Event()
        self._inference_wakeup = Event()
        self._inference_idle = Event()
        self._inference_idle.set()
        self._lock = Lock()
        self._latest_lock = Lock()
        self._latest_batch: list[dict[str, object]] | None = None
        self._worker: Thread | None = None
        self._capture_worker: Thread | None = None
        self._inference_worker: Thread | None = None
        self._worker_error: str | None = None
        self._generation = 0
        self._generation_lock = Lock()
        self._async_runtime = all(
            callable(getattr(self.engine, name, None))
            for name in ("prepare_inference_batch", "submit_inference", "run_scheduled_inference")
        )

    def start(self) -> ControllerStatus:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return self.status()
            self._stop_event.clear()
            self._resume_event.set()
            self._first_poll.clear()
            self._inference_wakeup.clear()
            self._inference_idle.set()
            with self._latest_lock:
                self._latest_batch = None
            self._worker_error = None
            with self._generation_lock:
                self._generation = 0
            if self.lifecycle is not None:
                self.lifecycle.start()
            if self._async_runtime:
                self._inference_worker = Thread(target=self._inference_loop, name="secretary-inference-worker", daemon=True)
                self._capture_worker = Thread(target=self._capture_loop, name="secretary-screenpipe-worker", daemon=True)
                self._worker = self._capture_worker
                self._inference_worker.start()
                self._capture_worker.start()
            else:
                self._worker = Thread(target=self._capture_loop, name="secretary-screenpipe-worker", daemon=True)
                self._capture_worker = self._worker
                self._worker.start()
        return self.status()

    def pause(self) -> ControllerStatus:
        self._resume_event.clear()
        with self._latest_lock:
            self._latest_batch = None
        self._inference_wakeup.set()
        cancel_pending = getattr(self.engine, "cancel_pending_inference", None)
        if callable(cancel_pending):
            cancel_pending()
        self.engine.pause()
        if self.lifecycle is not None:
            self.lifecycle.pause()
        return self.status()

    def resume(self) -> ControllerStatus:
        if self.lifecycle is not None:
            self.lifecycle.resume()
        self.engine.resume()
        self._worker_error = None
        self._resume_event.set()
        return self.status()

    def quit(self) -> None:
        self._stop_event.set()
        self._resume_event.set()
        self._inference_wakeup.set()
        workers = (self._capture_worker, self._inference_worker, self._worker)
        inference_timeout = getattr(getattr(self.engine, "inference", None), "timeout_seconds", 0.0)
        try:
            shutdown_timeout = max(5.0, self.poll_interval + 1.0, float(inference_timeout) + 1.0)
        except (TypeError, ValueError):
            shutdown_timeout = max(5.0, self.poll_interval + 1.0)
        for worker in workers:
            if worker is not None and worker is not current_thread():
                worker.join(timeout=shutdown_timeout)
        cancel_pending = getattr(self.engine, "cancel_pending_inference", None)
        if callable(cancel_pending):
            cancel_pending()
        # Cleanup is unconditional and remains scoped to the lifecycle manager's
        # verified owned PID. External Screenpipe is never controlled here.
        if self.lifecycle is not None:
            self.lifecycle.quit()
        self.engine.pause()

    def wait_for_first_poll(self, timeout: float | None = None) -> bool:
        return self._first_poll.wait(timeout)

    def wait_for_inference_idle(self, timeout: float | None = None) -> bool:
        """Stop new capture intake and wait for already queued work to finish."""
        if not self._async_runtime:
            return True
        self._resume_event.clear()
        self._inference_wakeup.set()
        return self._inference_idle.wait(timeout)

    def status(self) -> ControllerStatus:
        worker_alive = any(worker is not None and worker.is_alive() for worker in (self._capture_worker, self._inference_worker, self._worker))
        if self.lifecycle is None:
            paused = self.engine.session.paused
            return ControllerStatus(
                capture_status="PAUSED" if paused else "MOCK",
                worker_alive=worker_alive,
                owned_by_secretary=False,
                paused=paused,
                error=self._worker_error,
            )
        lifecycle_status = self.lifecycle.status()
        error = self._worker_error or lifecycle_status.error
        capture_status = "DEGRADED" if self._worker_error else lifecycle_status.capture_status
        return ControllerStatus(
            capture_status=capture_status,
            worker_alive=worker_alive,
            owned_by_secretary=lifecycle_status.owned_by_secretary,
            paused=lifecycle_status.paused or self.engine.session.paused,
            pid=lifecycle_status.pid,
            error=error,
        )

    def _capture_loop(self) -> None:
        next_supervision = monotonic() + self.supervision_interval
        while not self._stop_event.is_set():
            if not self._resume_event.wait(timeout=0.25):
                continue

            ready = self.lifecycle is None or self.lifecycle.status().capture_status == "READY"
            if ready:
                try:
                    items = [dict(item) for item in self.capture.poll()]  # type: ignore[attr-defined]
                    if self._async_runtime:
                        if items:
                            with self._generation_lock:
                                self._generation += 1
                                generation = self._generation
                            note_generation = getattr(self.engine, "note_generation", None)
                            if callable(note_generation):
                                note_generation(generation, items)
                        self._publish_latest(items)
                    else:
                        process_batch = getattr(self.engine, "process_coalesced", None)
                        if callable(process_batch):
                            process_batch(items)
                        else:
                            for item in items:
                                if self._stop_event.is_set():
                                    break
                                self.engine.process(item)
                except Exception as exc:
                    self._worker_error = f"capture worker failed: {exc.__class__.__name__}"
                finally:
                    self._first_poll.set()
            else:
                # A degraded real capture must not be replaced with fake data.
                self._first_poll.set()

            now = monotonic()
            if self.lifecycle is not None and now >= next_supervision and not self.lifecycle.paused:
                try:
                    self.lifecycle.supervise()
                except Exception as exc:
                    self._worker_error = f"Screenpipe supervision failed: {exc.__class__.__name__}"
                next_supervision = now + self.supervision_interval
            self._stop_event.wait(self.poll_interval)

    def _worker_loop(self) -> None:
        """Compatibility entry point for callers that used the old worker name."""
        self._capture_loop()

    def _publish_latest(self, items: list[dict[str, object]]) -> None:
        if not items:
            return
        with self._latest_lock:
            self._latest_batch = items
        self._inference_idle.clear()
        self._inference_wakeup.set()

    def _take_latest(self) -> list[dict[str, object]] | None:
        with self._latest_lock:
            batch = self._latest_batch
            self._latest_batch = None
        return batch

    def _inference_loop(self) -> None:
        pending_work = None
        while not self._stop_event.is_set():
            latest = self._take_latest()
            if latest is not None:
                try:
                    _, new_work = self.engine.prepare_inference_batch(latest)
                    if new_work is not None:
                        pending_work = new_work
                        self.engine.submit_inference(pending_work)
                    elif getattr(getattr(self.engine, "session", None), "paused", False):
                        pending_work = None
                except Exception as exc:
                    pending_work = None
                    self._worker_error = f"inference preparation failed: {exc.__class__.__name__}"

            if pending_work is None:
                if not self._has_latest():
                    self._inference_idle.set()
                self._inference_wakeup.wait(0.25)
                self._inference_wakeup.clear()
                continue

            wait_seconds = self.engine.inference_wait_seconds()
            if wait_seconds is not None and wait_seconds > 0:
                self._inference_wakeup.wait(min(0.25, wait_seconds))
                self._inference_wakeup.clear()
                continue

            try:
                result = self.engine.run_scheduled_inference(pending_work)
                # None with a remaining wait means the scheduler throttled the
                # request. None with no pending wait means it was stale/discarded.
                pending_work = pending_work if self.engine.inference_wait_seconds() is not None else None
                if result is not None:
                    pending_work = None
                if pending_work is None and not self._has_latest():
                    self._inference_idle.set()
            except Exception as exc:
                pending_work = None
                self._inference_idle.set()
                self._worker_error = f"inference worker failed: {exc.__class__.__name__}"

    def _has_latest(self) -> bool:
        with self._latest_lock:
            return self._latest_batch is not None
