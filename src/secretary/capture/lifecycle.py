from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol
import os
from pathlib import Path

from ..config import resolve_launcher
from ..platform.windows.job_object import owned_descendant_pids, terminate_owned_pid
from .screenpipe import ScreenpipeCaptureProvider


class ProcessLike(Protocol):
    pid: int

    def poll(self) -> int | None:
        ...

    def terminate(self) -> None:
        ...

    def wait(self, timeout: float | None = None) -> int | None:
        ...


class JobLike(Protocol):
    def add_process(self, pid: int) -> bool:
        ...

    def close(self) -> None:
        ...


@dataclass
class LifecycleStatus:
    mode: str
    capture_status: str
    owned_by_secretary: bool
    paused: bool = False
    pid: int | None = None
    error: str | None = None

    @property
    def available(self) -> bool:
        """Compatibility alias for callers that only need a ready boolean."""
        return self.capture_status == "READY"


class ScreenpipeLifecycleManager:
    """Manage only a Screenpipe child started by this Secretary instance."""

    def __init__(
        self,
        provider: ScreenpipeCaptureProvider,
        mode: str = "managed",
        command: Sequence[str] = (),
        process_factory=subprocess.Popen,
        job_factory=None,
        ready_timeout: float = 30.0,
        poll_interval: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = provider
        self.mode = mode
        self.command = tuple(command)
        self.process_factory = process_factory
        self.job_factory = job_factory
        self.ready_timeout = max(0.0, ready_timeout)
        self.poll_interval = max(0.01, poll_interval)
        self.sleep = sleep
        self.process: ProcessLike | None = None
        self.job: JobLike | None = None
        self.owned_by_secretary = False
        self.paused = False
        self.capture_status = "STOPPED"
        self.error: str | None = None
        self._resume_owned = False
        self._reused_external = False

    def start(self) -> LifecycleStatus:
        self.paused = False
        self.error = None
        self._resume_owned = False
        self._reused_external = False
        if hasattr(self.provider, "api_key") and not getattr(self.provider, "api_key"):
            self.capture_status = "DEGRADED"
            self.error = "SCREENPIPE_API_KEY is not configured"
            return self.status()
        if self._probe_ready():
            self._reused_external = True
            self.capture_status = "READY"
            self.owned_by_secretary = False
            return self.status()
        self._reused_external = False
        if self.mode != "managed" or not self.command:
            self.capture_status = "DEGRADED"
            self.error = "Screenpipe is not ready and managed lifecycle is disabled"
            return self.status()
        return self._start_owned()

    def pause(self) -> LifecycleStatus:
        self.paused = True
        if self.owned_by_secretary:
            self._resume_owned = True
            self._stop_owned()
        self.capture_status = "PAUSED"
        return self.status()

    def resume(self) -> LifecycleStatus:
        should_restart_owned = self._resume_owned
        self._resume_owned = False
        self.paused = False
        self.error = None
        if should_restart_owned:
            return self._start_owned()
        # A reused external process is never started or stopped by Resume.
        if self._reused_external and self._probe_ready():
            self.capture_status = "READY"
        else:
            self.capture_status = "DEGRADED"
            self.error = "external Screenpipe is not ready"
        return self.status()

    def supervise(self) -> LifecycleStatus:
        """Perform one low-frequency process/readiness check.

        This method is intentionally called by the controller on a coarse timer,
        not once per capture poll.  An owned child may be restarted safely.  An
        external instance is only observed and can become DEGRADED; it is never
        started, terminated, or replaced here.
        """
        if self.paused:
            return self.status()

        if hasattr(self.provider, "api_key") and not getattr(self.provider, "api_key"):
            self.capture_status = "DEGRADED"
            self.error = "SCREENPIPE_API_KEY is not configured"
            return self.status()

        if self.owned_by_secretary and self.process is not None:
            poll = getattr(self.process, "poll", None)
            if callable(poll) and poll() is not None:
                self.capture_status = "RESTARTING"
                self.error = "Secretary-owned Screenpipe child exited unexpectedly"
                self._stop_owned()
                return self._start_owned()
            if not self._probe_ready():
                restart_error = self.error or "Secretary-owned Screenpipe is not ready"
                self.capture_status = "RESTARTING"
                self._stop_owned()
                status = self._start_owned()
                if status.capture_status != "READY" and not status.error:
                    self.error = restart_error
                return status
            self.capture_status = "READY"
            self.error = None
            return self.status()

        if self._reused_external:
            if self._probe_ready():
                self.capture_status = "READY"
                self.error = None
            else:
                self.capture_status = "DEGRADED"
                self.error = self.error or "external Screenpipe is not ready"
        elif self.mode == "managed" and self.command:
            # A managed launch can fail transiently (for example while npx or
            # the recorder is still settling). Retry only on this coarse
            # supervision cadence; never turn an external instance into an
            # owned process after it was reused.
            self.capture_status = "RESTARTING"
            status = self._start_owned()
            if status.capture_status != "READY":
                self.capture_status = "DEGRADED"
        return self.status()

    def quit(self) -> None:
        self._stop_owned()

    def _start_owned(self) -> LifecycleStatus:
        self.error = None
        try:
            launch_command = list(self.command)
            resolved = resolve_launcher(launch_command)
            if resolved:
                launch_command[0] = resolved
            child_env = os.environ.copy()
            if resolved:
                # The Windows npx wrapper may invoke `node` by name. Supplying
                # the existing install directory to this child avoids changing
                # the user's global PATH.
                launcher_dir = str(Path(resolved).parent)
                child_env["PATH"] = launcher_dir + os.pathsep + child_env.get("PATH", "")
            self.process = self.process_factory(
                tuple(launch_command),
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=child_env,
            )
        except (OSError, ValueError) as exc:
            self.process = None
            self.capture_status = "DEGRADED"
            self.error = f"Screenpipe launch failed: {exc.__class__.__name__}"
            return self.status()
        self.owned_by_secretary = True
        if self.job_factory is not None:
            try:
                self.job = self.job_factory()
                if not self.job.add_process(self.process.pid):
                    self.job.close()
                    self.job = None
            except Exception:
                # Job support is best effort; never use a broad process-kill fallback.
                self.job = None
        if self._wait_until_ready():
            self.capture_status = "READY"
            return self.status()
        launch_error = self.error or "Screenpipe did not become ready"
        self._stop_owned()
        self.capture_status = "DEGRADED"
        self.error = launch_error
        return self.status()

    def _wait_until_ready(self) -> bool:
        deadline = time.monotonic() + self.ready_timeout
        while True:
            if self.process is not None and callable(getattr(self.process, "poll", None)) and self.process.poll() is not None:
                self.error = "Screenpipe child exited before becoming ready"
                return False
            if self._probe_ready():
                return True
            if time.monotonic() >= deadline:
                self.error = "Screenpipe did not become ready before timeout"
                return False
            self.sleep(self.poll_interval)

    def _probe_ready(self) -> bool:
        try:
            if not self.provider.health():
                return False
            authenticated_search = getattr(self.provider, "authenticated_search", None)
            if callable(authenticated_search):
                return bool(authenticated_search(limit=1))
            # Lightweight fakes can model health only; the real adapter always has
            # the authenticated probe above.
            return True
        except Exception as exc:
            self.error = f"Screenpipe readiness probe failed: {exc.__class__.__name__}"
            return False

    def _stop_owned(self) -> None:
        if not self.owned_by_secretary or self.process is None:
            return
        root_pid = self.process.pid
        descendants = owned_descendant_pids(root_pid)
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            # A failed graceful stop is intentionally not followed by a broad kill.
            pass
        # npx/cmd/node can have already-created descendants that a Job Object
        # cannot retroactively absorb. They are safe to clean because they are
        # descendants of the exact root PID Secretary started.
        descendants.update(owned_descendant_pids(root_pid))
        for pid in descendants:
            terminate_owned_pid(pid)
        if self.job is not None:
            try:
                self.job.close()
            except Exception:
                pass
        self.process = None
        self.job = None
        self.owned_by_secretary = False

    def status(self) -> LifecycleStatus:
        return LifecycleStatus(
            mode=self.mode,
            capture_status=self.capture_status,
            owned_by_secretary=self.owned_by_secretary,
            paused=self.paused,
            pid=self.process.pid if self.process is not None else None,
            error=self.error,
        )
