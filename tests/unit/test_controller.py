from __future__ import annotations

import time
import unittest
from dataclasses import dataclass
from threading import Event
from types import SimpleNamespace

from secretary.capture.lifecycle import LifecycleStatus
from secretary.controller import SecretaryController
from secretary.inference.scheduler import InferenceScheduler


class FakeCapture:
    def __init__(self) -> None:
        self.poll_calls = 0

    def poll(self):
        self.poll_calls += 1
        return [{"timestamp": "2026-08-28T10:00:00Z", "foreground_app": "Code.exe", "text": "editing"}]


class FakeSession:
    paused = False


class FakeEngine:
    def __init__(self) -> None:
        self.session = FakeSession()
        self.processed = Event()
        self.process_count = 0

    def process(self, item) -> None:
        self.process_count += 1
        self.processed.set()

    def pause(self) -> None:
        self.session.paused = True

    def resume(self) -> None:
        self.session.paused = False


@dataclass
class FakeLifecycle:
    supervised: int = 0
    paused: bool = False
    quit_called: bool = False
    ready: bool = False

    def start(self):
        self.ready = True
        return self.status()

    def pause(self):
        self.paused = True
        self.ready = False
        return self.status()

    def resume(self):
        self.paused = False
        self.ready = True
        return self.status()

    def supervise(self):
        self.supervised += 1
        return self.status()

    def quit(self) -> None:
        self.quit_called = True

    def status(self):
        return LifecycleStatus(
            mode="managed",
            capture_status="PAUSED" if self.paused else ("READY" if self.ready else "DEGRADED"),
            owned_by_secretary=True,
            paused=self.paused,
            pid=1234 if self.ready else None,
        )


class ControllerTests(unittest.TestCase):
    def test_worker_processes_capture_while_ui_controller_remains_responsive(self) -> None:
        engine = FakeEngine()
        capture = FakeCapture()
        lifecycle = FakeLifecycle()
        controller = SecretaryController(
            engine,
            capture,
            lifecycle=lifecycle,  # type: ignore[arg-type]
            poll_interval=0.05,
            supervision_interval=5,
        )

        status = controller.start()
        self.assertEqual(status.capture_status, "READY")
        self.assertTrue(engine.processed.wait(1))
        self.assertTrue(controller.status().worker_alive)

        calls_before_pause = capture.poll_calls
        paused = controller.pause()
        self.assertTrue(paused.paused)
        time.sleep(0.12)
        self.assertEqual(capture.poll_calls, calls_before_pause)

        controller.resume()
        self.assertTrue(self._wait_for(lambda: capture.poll_calls > calls_before_pause))
        controller.quit()

        self.assertTrue(lifecycle.quit_called)
        self.assertFalse(controller.status().worker_alive)
        self.assertTrue(engine.session.paused)

    def test_capture_continues_while_slow_inference_keeps_only_latest_state(self) -> None:
        class BlockingCapture:
            def __init__(self) -> None:
                self.poll_calls = 0
                self.states = iter(("A", "B", "C", "D", "E", "F"))

            def poll(self):
                self.poll_calls += 1
                state = next(self.states, "F")
                return [{"text": state}]

        class BlockingEngine:
            def __init__(self) -> None:
                self.session = FakeSession()
                self.scheduler = InferenceScheduler(min_interval_seconds=0, stale_request_seconds=30)
                self.started = Event()
                self.release = Event()
                self.completed: list[str] = []
                self.generations: list[int] = []

            def note_generation(self, generation, items):
                self.generations.append(generation)

            def prepare_inference_batch(self, items):
                text = items[-1]["text"]
                work = SimpleNamespace(text=text, request=SimpleNamespace())
                return [], work

            def submit_inference(self, work):
                self.scheduler.submit(work.request)

            def inference_wait_seconds(self):
                return self.scheduler.next_wait_seconds()

            def run_scheduled_inference(self, work):
                self.scheduler.start_next()
                if not self.completed:
                    self.started.set()
                    self.release.wait(2)
                self.completed.append(work.text)
                self.scheduler.complete()
                return object()

            def cancel_pending_inference(self):
                self.scheduler.cancel_pending()

            def pause(self) -> None:
                self.session.paused = True

        capture = BlockingCapture()
        engine = BlockingEngine()
        controller = SecretaryController(engine, capture, poll_interval=0.01, supervision_interval=5)
        controller.start()
        try:
            self.assertTrue(engine.started.wait(1))
            self.assertTrue(self._wait_for(lambda: capture.poll_calls >= 6))
            self.assertTrue(engine.generations)
            self.assertEqual(engine.generations, list(range(1, len(engine.generations) + 1)))
            engine.release.set()
            self.assertTrue(self._wait_for(lambda: len(engine.completed) >= 2))
            self.assertEqual(engine.completed[:2], ["A", "F"])
        finally:
            engine.release.set()
            controller.quit()

    @staticmethod
    def _wait_for(predicate, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()
