from __future__ import annotations

import unittest

from secretary.capture.lifecycle import ScreenpipeLifecycleManager


class FakeProvider:
    def __init__(self, healthy: bool) -> None:
        self.healthy = healthy
        self.calls = 0

    def health(self) -> bool:
        self.calls += 1
        return self.healthy

    def authenticated_search(self, limit: int = 1) -> bool:
        return self.healthy


class FakeProcess:
    def __init__(self, pid: int = 1234) -> None:
        self.pid = pid
        self.terminated = False
        self.exited = False

    def poll(self) -> int | None:
        return 0 if self.terminated or self.exited else None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None) -> int:
        return 0


class FakeJob:
    def __init__(self) -> None:
        self.pids: list[int] = []
        self.closed = False

    def add_process(self, pid: int) -> bool:
        self.pids.append(pid)
        return True

    def close(self) -> None:
        self.closed = True


class LifecycleTests(unittest.TestCase):
    def test_external_process_is_never_stopped(self) -> None:
        provider = FakeProvider(True)
        manager = ScreenpipeLifecycleManager(provider, mode="external", command=("unused",))
        status = manager.start()
        self.assertFalse(status.owned_by_secretary)
        manager.pause()
        manager.quit()
        self.assertIsNone(manager.process)

    def test_managed_process_is_owned_and_stopped(self) -> None:
        provider = FakeProvider(False)
        process = FakeProcess()
        job = FakeJob()
        def launch(*args, **kwargs):
            provider.healthy = True
            return process

        manager = ScreenpipeLifecycleManager(provider, mode="managed", command=("screenpipe",), process_factory=launch, job_factory=lambda: job, ready_timeout=0)
        status = manager.start()
        self.assertTrue(status.owned_by_secretary)
        self.assertEqual(job.pids, [1234])
        manager.quit()
        self.assertTrue(process.terminated)
        self.assertTrue(job.closed)

    def test_managed_pause_and_resume_only_control_owned_process(self) -> None:
        provider = FakeProvider(False)
        first_process = FakeProcess()
        second_process = FakeProcess()
        first_job = FakeJob()
        second_job = FakeJob()
        processes = [first_process, second_process]
        jobs = [first_job, second_job]
        def launch(*args, **kwargs):
            provider.healthy = True
            return processes.pop(0)

        manager = ScreenpipeLifecycleManager(
            provider,
            mode="managed",
            command=("screenpipe",),
            process_factory=launch,
            job_factory=lambda: jobs.pop(0),
            ready_timeout=0,
        )
        manager.start()
        manager.pause()
        self.assertTrue(manager.paused)
        self.assertTrue(first_process.terminated)
        self.assertTrue(first_job.closed)
        manager.resume()
        self.assertTrue(manager.owned_by_secretary)
        self.assertIs(manager.process, second_process)

    def test_failed_managed_launch_is_degraded_without_fake_capture(self) -> None:
        provider = FakeProvider(False)

        def launch(*args, **kwargs):
            raise FileNotFoundError("npx.cmd")

        manager = ScreenpipeLifecycleManager(provider, mode="managed", command=("npx.cmd",), process_factory=launch, ready_timeout=0)
        status = manager.start()
        self.assertEqual(status.capture_status, "DEGRADED")
        self.assertFalse(status.available)
        self.assertIsNone(manager.process)

    def test_owned_child_exit_is_restarted(self) -> None:
        provider = FakeProvider(False)
        first_process = FakeProcess(1234)
        second_process = FakeProcess(5678)
        processes = [first_process, second_process]

        def launch(*args, **kwargs):
            provider.healthy = True
            return processes.pop(0)

        manager = ScreenpipeLifecycleManager(
            provider,
            mode="managed",
            command=("screenpipe",),
            process_factory=launch,
            ready_timeout=0,
        )
        self.assertEqual(manager.start().capture_status, "READY")
        first_process.exited = True

        status = manager.supervise()

        self.assertEqual(status.capture_status, "READY")
        self.assertTrue(status.owned_by_secretary)
        self.assertIs(manager.process, second_process)
        self.assertTrue(first_process.terminated)
        manager.quit()

    def test_external_process_becomes_degraded_without_restart(self) -> None:
        provider = FakeProvider(True)
        manager = ScreenpipeLifecycleManager(provider, mode="external", command=("screenpipe",))
        self.assertEqual(manager.start().capture_status, "READY")
        provider.healthy = False

        status = manager.supervise()

        self.assertEqual(status.capture_status, "DEGRADED")
        self.assertFalse(status.owned_by_secretary)
        self.assertIsNone(manager.process)
