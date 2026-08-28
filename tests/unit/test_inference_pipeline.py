from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError

from secretary.config import SecretaryConfig
from secretary.engine import SecretaryEngine
from secretary.events.normalize import normalize_fixture_item
from secretary.events.schema import NormalizedEvent
from secretary.inference.coalescer import EventCoalescer
from secretary.inference.context import InferenceContextBuilder
from secretary.inference.image import EncodedImage, ImagePreprocessor
from secretary.inference.mock import MockInferenceProvider
from secretary.inference.ollama import OllamaInferenceProvider
from secretary.inference.scheduler import InferenceScheduler
from secretary.inference.schema import (
    Action,
    InferenceEvent,
    InferenceRequest,
    InferenceResult,
    SecretaryAssessment,
    validate_inference_result,
)
from secretary.inference.status import InferenceRuntimeState
from secretary.inference.vision import VisionGate
from secretary.main import run_inference_status, run_preflight
from secretary.notifications.mock import MockNotificationProvider


def make_event(*, app: str = "Code.exe", text: str = "editing attention.py", image_path: str | None = None, visual_required: bool = False) -> NormalizedEvent:
    return NormalizedEvent(
        timestamp=datetime(2026, 8, 28, 10, 0, tzinfo=timezone.utc),
        source="test",
        foreground_app=app,
        window_title="Editor",
        event_source="screen",
        text=text,
        text_source="fixture",
        focused=True,
        screen_changed=True,
        visual_required=visual_required,
        image_path=image_path,
    )


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class InferencePipelineTests(unittest.TestCase):
    def test_screenpipe_image_path_is_provider_neutral_and_not_safe_metadata(self) -> None:
        raw = {
            "type": "OCR",
            "content": {
                "app_name": "Code.exe",
                "text": "editing",
                "file_path": "C:\\Users\\test\\frame.jpg",
                "frame_id": 9,
                "timestamp": "2026-08-28T10:00:00Z",
            },
        }
        event = normalize_fixture_item(raw)

        self.assertEqual(event.image_path, "C:\\Users\\test\\frame.jpg")
        self.assertNotIn("image_path", event.safe_metadata())

    def test_context_builder_prioritizes_current_event_and_watch_state(self) -> None:
        current = make_event(text="x" * 2000)
        request = InferenceContextBuilder(max_text_chars=600).build(
            current,
            recent_events=[make_event(text="old event")],
            working_state={"current_objective": "resolve test-failure:python", "current_subgoal": "inspect traceback"},
            active_hypotheses=[{"hypothesis": "same failure is repeating", "evidence": 3}],
            recent_failures=["test-failure:python"],
            recent_assistant_decisions=["WATCH"],
        )

        self.assertLessEqual(len(request.context_text), 600)
        self.assertIn("CURRENT EVENT", request.context_text)
        self.assertIn("ACTIVE WATCH HYPOTHESIS", request.context_text)
        self.assertIn("test-failure:python", request.context_text)

    def test_vision_gate_is_deterministic(self) -> None:
        builder = InferenceContextBuilder()
        self.assertTrue(builder.build(make_event(text="", visual_required=False)).use_vision)
        self.assertFalse(builder.build(make_event(app="WindowsTerminal.exe", text="pytest FAILED with AssertionError")).use_vision)
        self.assertTrue(builder.build(make_event(app="Figma.exe", text="design", visual_required=False)).use_vision)
        self.assertTrue(builder.build(make_event(text="enough text", visual_required=True)).use_vision)

    def test_vision_cooldown_is_optional_and_clock_driven(self) -> None:
        clock = FakeClock()
        gate = VisionGate(cooldown_seconds=30, clock=clock)
        visual = make_event(text="visual", visual_required=False)
        self.assertTrue(gate.allow(visual, now=0))
        self.assertFalse(gate.allow(visual, now=10))
        self.assertTrue(gate.allow(visual, now=30))

    def test_event_coalescer_groups_short_activity_window(self) -> None:
        clock = FakeClock()
        coalescer = EventCoalescer(window_seconds=2, max_events=3, clock=clock)
        first = make_event(text="click")
        second = make_event(text="pytest")
        third = make_event(text="FAILED")
        self.assertIsNone(coalescer.add(first, now=0))
        self.assertIsNone(coalescer.add(second, now=1))
        completed = coalescer.add(third, now=4)

        self.assertIsNotNone(completed)
        self.assertEqual([item.text for item in completed.events], ["click", "pytest"])
        self.assertEqual(coalescer.pending_count, 1)

    def test_scheduler_latest_state_wins_and_discards_stale_requests(self) -> None:
        clock = FakeClock()
        scheduler = InferenceScheduler(min_interval_seconds=10, stale_request_seconds=30, clock=clock)
        first = InferenceRequest(make_event(text="A"))
        second = InferenceRequest(make_event(text="B"))
        third = InferenceRequest(make_event(text="C"))
        scheduler.submit(first, now=0)
        self.assertIs(scheduler.start_next(now=0), first)
        scheduler.submit(second, now=1)
        scheduler.submit(third, now=2)
        scheduler.complete()
        self.assertIsNone(scheduler.start_next(now=5))
        self.assertIs(scheduler.start_next(now=11), third)
        scheduler.complete()
        scheduler.submit(first, now=12)
        self.assertIsNone(scheduler.start_next(now=50))
        self.assertEqual(scheduler.discarded_stale_requests, 1)

    def test_validation_is_conservative_for_invalid_actions_and_numbers(self) -> None:
        result = validate_inference_result({
            "event": {"summary": "unsafe output", "importance": 2, "novelty": float("nan"), "confidence": -1},
            "secretary": {"candidate_action": "DO_ANYTHING", "interrupt_score": 4, "reason": "test"},
        }, provider="fake")

        self.assertEqual(result.secretary.candidate_action, Action.IGNORE)
        self.assertEqual(result.event.importance, 1.0)
        self.assertEqual(result.event.novelty, 0.1)
        self.assertEqual(result.event.confidence, 0.0)
        self.assertEqual(result.secretary.interrupt_score, 1.0)

    def test_mock_provider_implements_formal_result(self) -> None:
        result = MockInferenceProvider().analyze(InferenceRequest(make_event(text="pytest FAILED")))

        self.assertIsInstance(result, InferenceResult)
        self.assertEqual(result.event.failure_signature, "test-failure:python")
        self.assertEqual(result.secretary.candidate_action, Action.WATCH)
        self.assertEqual(MockInferenceProvider().status().status, InferenceRuntimeState.MOCK)

    def test_privacy_filter_runs_before_image_preprocessor(self) -> None:
        class ExplodingPreprocessor:
            def prepare_image(self, path):
                raise AssertionError("excluded image must never be opened")

        provider = OllamaInferenceProvider(text_model="test-model", image_preprocessor=ExplodingPreprocessor(), http_post=lambda *args: {})
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = SecretaryConfig(project_root=root, database_path=root / "state.db", log_directory=root / "logs", excluded_apps=("KeePass",))
            engine = SecretaryEngine(config, inference_provider=provider, notifier=MockNotificationProvider())
            try:
                result = engine.process({
                    "timestamp": "2026-08-28T10:00:00Z",
                    "foreground_app": "KeePass.exe",
                    "text": "secret",
                    "image_path": str(root / "does-not-open.jpg"),
                    "visual_required": True,
                })
                self.assertTrue(result.privacy_suppressed)
            finally:
                engine.close()

    def test_model_candidate_does_not_bypass_existing_policy(self) -> None:
        class AlwaysNotifyProvider:
            name = "fake"
            model = "fake-model"

            def analyze(self, request):
                return InferenceResult(
                    event=InferenceEvent(event_type="coding", activity="editor", summary="editing", confidence=1.0),
                    secretary=SecretaryAssessment(candidate_action=Action.NOTIFY, interrupt_score=1.0, reason="model asked"),
                    provider=self.name,
                    model=self.model,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = SecretaryConfig(project_root=root, database_path=root / "state.db", log_directory=root / "logs")
            engine = SecretaryEngine(config, inference_provider=AlwaysNotifyProvider(), notifier=MockNotificationProvider())
            try:
                result = engine.process({"timestamp": "2026-08-28T10:00:00Z", "foreground_app": "Code.exe", "text": "editing"})
                self.assertEqual(result.decision.action, Action.IGNORE)
            finally:
                engine.close()

    def test_engine_coalesced_capture_uses_one_inference_for_one_poll_batch(self) -> None:
        calls = []
        mock = MockInferenceProvider()

        class RecordingProvider:
            name = "recording-mock"
            model = None

            def analyze(self, request):
                calls.append(request)
                return mock.analyze(request)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = SecretaryConfig(project_root=root, database_path=root / "state.db", log_directory=root / "logs")
            engine = SecretaryEngine(config, inference_provider=RecordingProvider(), notifier=MockNotificationProvider())
            try:
                results = engine.process_coalesced([
                    {"timestamp": "2026-08-28T10:00:00Z", "foreground_app": "Code.exe", "text": "editing"},
                    {"timestamp": "2026-08-28T10:00:01Z", "foreground_app": "WindowsTerminal.exe", "text": "pytest FAILED"},
                    {"timestamp": "2026-08-28T10:00:02Z", "foreground_app": "WindowsTerminal.exe", "text": "AssertionError"},
                ])
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(calls[0].recent_events), 3)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].decision.action, Action.REMEMBER)
            finally:
                engine.close()

    def test_ollama_text_contract_is_fully_offline(self) -> None:
        calls = []

        def transport(url, payload, timeout):
            calls.append((url, payload, timeout))
            return {"message": {"content": json.dumps({
                "event": {"event_type": "coding", "activity": "editor", "summary": "editing", "importance": 0.2, "novelty": 0.3, "confidence": 0.9},
                "secretary": {"candidate_action": "IGNORE", "interrupt_score": 0.1, "reason": "quiet"},
            })}, "total_duration": 2_000_000, "load_duration": 500_000, "prompt_eval_count": 4, "prompt_eval_duration": 250_000, "eval_count": 3, "eval_duration": 1_000_000}

        provider = OllamaInferenceProvider(base_url="http://offline.invalid:11434", text_model="text-model", vision_model="vision-model", keep_alive="30m", http_post=transport, system_prompt="SYS")
        self.assertEqual(provider.status().status, InferenceRuntimeState.NOT_CHECKED)
        result = provider.analyze(InferenceRequest(make_event(text="enough text for a text-only request"), context_text="CONTEXT", use_vision=False))

        self.assertEqual(result.event.summary, "editing")
        self.assertEqual(calls[0][0], "http://offline.invalid:11434/api/chat")
        payload = calls[0][1]
        self.assertEqual(payload["model"], "text-model")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["keep_alive"], "30m")
        self.assertEqual(payload["messages"][0]["content"], "SYS")
        self.assertEqual(payload["messages"][1]["content"], "CONTEXT")
        self.assertIsInstance(payload["format"], dict)
        self.assertEqual(payload["options"], {"temperature": 0.0})
        self.assertFalse(payload["think"])
        self.assertEqual(result.metrics.mode, "text")
        self.assertEqual(result.metrics.ollama_total_duration_ms, 2.0)
        self.assertEqual(result.metrics.model_load_ms, 0.5)
        self.assertEqual(result.metrics.prompt_tokens, 4)
        self.assertEqual(result.metrics.output_tokens, 3)

    def test_ollama_probe_checks_version_and_exact_configured_models(self) -> None:
        calls = []

        def transport(url, timeout):
            calls.append(url)
            if url.endswith("/api/version"):
                return {"version": "0.12.7"}
            return {"models": [{"name": "qwen3-vl:2b-instruct-q4_K_M"}]}

        provider = OllamaInferenceProvider(
            text_model="qwen3-vl:2b-instruct-q4_K_M",
            vision_model="qwen3-vl:2b-instruct-q4_K_M",
            http_get=transport,
        )
        result = provider.probe()

        self.assertEqual(result.status, InferenceRuntimeState.READY)
        self.assertEqual(result.version, "0.12.7")
        self.assertTrue(result.model_available)
        self.assertEqual(calls, ["http://127.0.0.1:11434/api/version", "http://127.0.0.1:11434/api/tags"])

    def test_ollama_probe_reports_qwen3_vl_incompatible_runtime_without_upgrade(self) -> None:
        def transport(url, timeout):
            if url.endswith("/api/version"):
                return {"version": "0.12.6"}
            return {"models": [{"name": "qwen3-vl:2b-instruct-q4_K_M"}]}

        provider = OllamaInferenceProvider(text_model="qwen3-vl:2b-instruct-q4_K_M", http_get=transport)
        result = provider.probe()

        self.assertEqual(result.status, InferenceRuntimeState.DEGRADED)
        self.assertEqual(result.error_type, "incompatible_runtime")
        self.assertIn("requires >= 0.12.7", result.detail)

    def test_ollama_vision_contract_uses_preprocessed_fixture_data(self) -> None:
        calls = []

        class FakePreprocessor:
            def prepare_image(self, path):
                return EncodedImage(data="ENCODED_FIXTURE", mime_type="image/png", width=4, height=2)

        def transport(url, payload, timeout):
            calls.append(payload)
            return {"message": {"content": json.dumps({
                "event": {"event_type": "activity", "activity": "visual", "summary": "visual context", "importance": 0.1, "novelty": 0.1, "confidence": 0.8},
                "secretary": {"candidate_action": "IGNORE", "interrupt_score": 0, "reason": "quiet"},
            })}}

        provider = OllamaInferenceProvider(text_model="text", vision_model="vision", http_post=transport, image_preprocessor=FakePreprocessor(), system_prompt="SYS")
        provider.analyze(InferenceRequest(make_event(text="visual", image_path="fixture.ppm", visual_required=True), use_vision=True, image_path="fixture.ppm"))

        self.assertEqual(calls[0]["model"], "vision")
        self.assertEqual(calls[0]["messages"][1]["images"], ["ENCODED_FIXTURE"])

    def test_ollama_failures_return_safe_results_without_raising(self) -> None:
        failures = [
            ("malformed", lambda *args: {"message": {"content": "not-json"}}, "malformed_response"),
            ("missing", lambda *args: {"message": {}}, "malformed_response"),
            ("timeout", lambda *args: (_ for _ in ()).throw(TimeoutError()), "timeout"),
            ("connection", lambda *args: (_ for _ in ()).throw(ConnectionError()), "connection_error"),
            ("http", lambda *args: (_ for _ in ()).throw(HTTPError("url", 500, "error", {}, None)), "http_error"),
        ]
        for name, transport, error_type in failures:
            with self.subTest(name=name):
                provider = OllamaInferenceProvider(text_model="model", http_post=transport)
                result = provider.analyze(InferenceRequest(make_event(text="enough text for text inference")))
                self.assertEqual(result.error_type, error_type)
                self.assertEqual(result.secretary.candidate_action, Action.IGNORE)
                self.assertEqual(provider.status().status, InferenceRuntimeState.DEGRADED)

    def test_missing_model_never_calls_transport(self) -> None:
        calls = []
        provider = OllamaInferenceProvider(http_post=lambda *args: calls.append(args))
        result = provider.analyze(InferenceRequest(make_event()))
        self.assertEqual(result.error_type, "model_not_configured")
        self.assertEqual(calls, [])

    def test_inference_status_is_non_probing_and_mock_is_default(self) -> None:
        output = io.StringIO()
        config = SecretaryConfig(inference_provider="mock")

        result = run_inference_status(config, output)

        self.assertEqual(result, 0)
        rendered = output.getvalue()
        self.assertIn("Provider: mock", rendered)
        self.assertIn("Status: MOCK", rendered)
        self.assertIn("Real model required: no", rendered)

    def test_ollama_status_is_not_checked_without_runtime_access(self) -> None:
        output = io.StringIO()
        config = SecretaryConfig(inference_provider="ollama", ollama_text_model="configured-test-model")

        result = run_inference_status(config, output)

        self.assertEqual(result, 0)
        self.assertIn("Provider: ollama", output.getvalue())
        self.assertIn("Status: NOT_CHECKED", output.getvalue())

    def test_preflight_does_not_probe_ollama_when_configured(self) -> None:
        output = io.StringIO()
        config = SecretaryConfig(inference_provider="ollama", capture_provider="mock")

        self.assertTrue(run_preflight(config, output))
        self.assertIn("Ollama runtime not checked by offline preflight", output.getvalue())
