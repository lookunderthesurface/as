from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .capture.screenpipe import ScreenpipeCaptureProvider
from .config import SecretaryConfig, ensure_project_dirs
from .context.session import SessionState
from .context.working_state import WorkingState
from .events.filter import MeaningfulEventFilter
from .events.normalize import normalize_fixture_item
from .events.schema import NormalizedEvent
from .logging_utils import build_logger, close_logger
from .memory.store import MemoryStore
from .notifications.base import NotificationProvider
from .notifications.mock import MockNotificationProvider
from .cloud.base import CloudProvider
from .cloud.mock import MockCloudProvider
from .inference.base import InferenceProvider
from .inference.context import InferenceContextBuilder
from .inference.coalescer import EventCoalescer
from .inference.image import ImagePreprocessor
from .inference.mock import MockInferenceProvider
from .inference.ollama import OllamaInferenceProvider
from .inference.scheduler import InferenceScheduler
from .inference.schema import InferenceRequest
from .inference.status import InferenceRuntimeState, LocalInferenceStatus
from .perception.extractor import ExtractedEvent, EventExtractor
from .policy.hard_rules import HardRules
from .policy.proactive import Action, Decision, PolicyThresholds, ProactivePolicy
from .policy.watch import WatchManager
from .privacy.filter import PrivacyFilter


@dataclass
class ProcessResult:
    decision: Decision
    event: ExtractedEvent | None = None
    privacy_suppressed: bool = False


@dataclass(frozen=True)
class InferenceWork:
    """Prepared work owned and executed by the inference worker only."""

    event: NormalizedEvent
    request: InferenceRequest


class SecretaryEngine:
    def __init__(
        self,
        config: SecretaryConfig | None = None,
        store: MemoryStore | None = None,
        notifier: NotificationProvider | None = None,
        inference_provider: InferenceProvider | None = None,
        cloud_provider: CloudProvider | None = None,
    ) -> None:
        self.config = config or SecretaryConfig.from_environment()
        ensure_project_dirs(self.config)
        self.store = store or MemoryStore(self.config.database_path)
        self.notifier = notifier or MockNotificationProvider()
        if inference_provider is not None:
            self.inference = inference_provider
        elif self.config.inference_provider == "mock":
            self.inference = MockInferenceProvider()
        elif self.config.inference_provider == "ollama":
            self.inference = OllamaInferenceProvider(
                base_url=self.config.ollama_base_url,
                text_model=self.config.ollama_text_model,
                vision_model=self.config.ollama_vision_model,
                timeout_seconds=self.config.ollama_timeout_seconds,
                keep_alive=self.config.ollama_keep_alive,
                temperature=self.config.ollama_temperature,
                think=self.config.ollama_think,
                image_preprocessor=ImagePreprocessor(self.config.vision_max_long_edge, self.config.vision_jpeg_quality),
            )
        else:
            raise RuntimeError(f"inference provider is not implemented: {self.config.inference_provider}")
        if cloud_provider is not None:
            self.cloud = cloud_provider
        elif self.config.cloud_provider == "mock":
            self.cloud = MockCloudProvider()
        else:
            raise RuntimeError(f"cloud provider is not implemented: {self.config.cloud_provider}")
        self.session = SessionState()
        self.session_id = self.store.start_session()
        self._closed = False
        self.state = WorkingState()
        self.filter = MeaningfulEventFilter()
        self.privacy = PrivacyFilter(self.config.excluded_apps)
        self.context_builder = InferenceContextBuilder(
            self.config.inference_max_text_chars,
            vision_cooldown_seconds=self.config.inference_vision_cooldown_seconds,
        )
        self.coalescer = EventCoalescer()
        self.scheduler = InferenceScheduler(
            min_interval_seconds=self.config.inference_min_interval_seconds,
            max_pending_requests=self.config.inference_max_pending_requests,
            stale_request_seconds=self.config.inference_stale_request_seconds,
        )
        self.extractor = EventExtractor(self.inference)
        self.watch = WatchManager(self.config.watch_expiration_minutes, self.config.watch_max_active_hypotheses)
        self.hard_rules = HardRules(
            self.config.max_notifications_per_hour,
            min_importance=self.config.model_notify_min_importance,
            min_confidence=self.config.model_notify_min_confidence,
            cooldown_seconds=self.config.notification_cooldown_seconds,
        )
        self.policy = ProactivePolicy(
            self.watch,
            self.hard_rules,
            PolicyThresholds(
                remember_min_confidence=self.config.model_remember_min_confidence,
                remember_min_importance=self.config.model_remember_min_importance,
                watch_min_confidence=self.config.model_watch_min_confidence,
                watch_min_importance=self.config.model_watch_min_importance,
                investigate_min_confidence=self.config.model_investigate_min_confidence,
                investigate_min_importance=self.config.model_investigate_min_importance,
                notify_min_confidence=self.config.model_notify_min_confidence,
                notify_min_importance=self.config.model_notify_min_importance,
                notify_min_interrupt_score=self.config.model_notify_min_interrupt_score,
                notify_min_watch_evidence=self.config.model_notify_min_watch_evidence,
            ),
        )
        self.logger = build_logger(self.config.log_directory)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.store.end_session(self.session_id)
        self.store.close()
        close_logger(self.logger)

    def pause(self) -> None:
        self.session.pause()

    def resume(self) -> None:
        self.session.resume()

    def inference_status(self) -> LocalInferenceStatus:
        status = getattr(self.inference, "status", None)
        if callable(status):
            return status()
        return LocalInferenceStatus(provider=self.config.inference_provider, status=InferenceRuntimeState.DEGRADED)

    def process(self, raw: dict[str, object]) -> ProcessResult:
        if self.session.paused:
            decision = Decision(Action.IGNORE, "paused")
            return ProcessResult(decision)
        event = normalize_fixture_item(raw)
        privacy = self.privacy.check(event)
        if privacy.blocked:
            self.logger.info("event_type=privacy-suppressed app_category=excluded policy=IGNORE")
            return ProcessResult(Decision(Action.IGNORE, "privacy excluded app"), privacy_suppressed=True)
        if not self.filter.accept(event):
            return ProcessResult(Decision(Action.IGNORE, "duplicate capture frame"))
        self.coalescer.add(event)
        request = self.context_builder.build(
            event,
            recent_events=self.coalescer.snapshot(),
            working_state=self.state.snapshot(),
            active_hypotheses=self.watch.snapshot(),
            recent_failures=self.state.recent_failures,
            recent_assistant_decisions=self.state.decisions,
            image_path=event.image_path,
        )
        extracted = self.extractor.extract(event, request)
        return self._apply_extracted(event, extracted, inference_mode="vision" if request.use_vision else "text")

    def process_coalesced(self, raws: list[dict[str, object]]) -> list[ProcessResult]:
        """Process one capture poll synchronously for replay and unit callers."""
        results, work = self.prepare_inference_batch(raws)
        if work is None:
            return results
        extracted = self.extractor.extract(work.event, work.request)
        results.append(self._apply_extracted(work.event, extracted, inference_mode="vision" if work.request.use_vision else "text"))
        return results

    def prepare_inference_batch(self, raws: list[dict[str, object]]) -> tuple[list[ProcessResult], InferenceWork | None]:
        """Prepare one bounded request without calling a provider or mutating policy state."""
        if self.session.paused:
            return [ProcessResult(Decision(Action.IGNORE, "paused")) for _ in raws], None
        accepted: list[NormalizedEvent] = []
        results: list[ProcessResult] = []
        for raw in raws:
            event = normalize_fixture_item(raw)
            privacy = self.privacy.check(event)
            if privacy.blocked:
                self.logger.info("event_type=privacy-suppressed app_category=excluded policy=IGNORE")
                results.append(ProcessResult(Decision(Action.IGNORE, "privacy excluded app"), privacy_suppressed=True))
                continue
            if not self.filter.accept(event):
                results.append(ProcessResult(Decision(Action.IGNORE, "duplicate capture frame")))
                continue
            accepted.append(event)
        accepted.sort(key=lambda item: (item.timestamp, item.stable_id))
        for event in accepted:
            self.coalescer.add(event)
        if not accepted:
            return results, None
        event = accepted[-1]
        request = self.context_builder.build(
            event,
            recent_events=self.coalescer.snapshot(),
            working_state=self.state.snapshot(),
            active_hypotheses=self.watch.snapshot(),
            recent_failures=self.state.recent_failures,
            recent_assistant_decisions=self.state.decisions,
            image_path=event.image_path,
        )
        return results, InferenceWork(event=event, request=request)

    def submit_inference(self, work: InferenceWork) -> None:
        """Replace the bounded pending request with the newest prepared state."""
        self.scheduler.submit(work.request)

    def run_scheduled_inference(self, work: InferenceWork) -> ProcessResult | None:
        """Run one admitted request and apply policy on this single worker."""
        request = self.scheduler.start_next()
        if request is None:
            # A stale request is removed by start_next. A throttled request
            # remains pending and the controller will retry after the interval.
            return None
        try:
            extracted = self.extractor.extract(work.event, request)
            result = None if self.session.paused else self._apply_extracted(
                work.event,
                extracted,
                inference_mode="vision" if request.use_vision else "text",
            )
        except Exception:
            self.scheduler.fail()
            raise
        else:
            self.scheduler.complete()
            return result

    def inference_wait_seconds(self) -> float | None:
        return self.scheduler.next_wait_seconds()

    def cancel_pending_inference(self) -> None:
        self.scheduler.cancel_pending()

    def _apply_extracted(self, event: NormalizedEvent, extracted: ExtractedEvent, *, inference_mode: str = "text") -> ProcessResult:
        self.state.observe(extracted)
        failure_count = self.store.count_failures(extracted.failure_signature) if extracted.failure_signature else 0
        decision = self.policy.decide(extracted, self.state, failure_count + (1 if extracted.failure_signature else 0), event.timestamp)
        self.store.record_event(extracted, source=event.source, session_id=self.session_id)
        final_action = decision.action.value
        self.state.add_decision(final_action)
        if decision.action == Action.REMEMBER:
            self.store.record_memory(extracted.summary, importance=extracted.importance, tags=extracted.failure_signature or "")
        if self.watch.active:
            self.state.set_hypotheses(self.watch.snapshot())
            active = self.watch.active
            self.store.record_hypothesis(active.hypothesis, active.evidence, "watching", active.expires_at.isoformat())
        if decision.action == Action.NOTIFY:
            title = decision.notification_title or "Ambient Secretary"
            body = decision.notification_body or decision.reason
            shadow_notification = bool(getattr(self.notifier, "shadow", False))
            try:
                self.notifier.notify(title, body)
                if not shadow_notification:
                    self.hard_rules.mark_notified(extracted, event.timestamp)
                else:
                    final_action = "WOULD_NOTIFY"
                self.store.record_notification(title, body, final_action)
            except Exception as exc:
                self.logger.warning("event_type=notification_error error_class=%s", exc.__class__.__name__)
        self.store.record_decision(decision.action.value, decision.reason, decision.evidence)
        inference_result = self.extractor.last_result
        metrics = inference_result.metrics if inference_result is not None else None
        if metrics is not None:
            inference_mode = metrics.mode
        metric_fields = ""
        if metrics:
            metric_fields = (
                f" inference_wall_latency_ms={metrics.wall_latency_ms:.1f}"
                f" ollama_total_duration_ms={metrics.ollama_total_duration_ms}"
                f" model_load_ms={metrics.model_load_ms} prompt_tokens={metrics.prompt_tokens}"
                f" output_tokens={metrics.output_tokens} prompt_eval_ms={metrics.prompt_eval_ms}"
                f" generation_ms={metrics.generation_ms} inference_mode={metrics.mode}"
            )
        provider_name = inference_result.provider if inference_result is not None else "unknown"
        self.logger.info(
            "event_type=%s app=%s policy=%s inference_provider=%s%s error_class=none",
            extracted.event_type,
            extracted.app,
            decision.action.value,
            provider_name,
            metric_fields,
        )
        self.store.record_decision_trace(
            session_id=self.session_id,
            event_timestamp=event.timestamp,
            foreground_app=event.foreground_app,
            event_type=extracted.event_type,
            candidate_action=extracted.candidate_action.value,
            candidate_confidence=extracted.confidence,
            candidate_importance=extracted.importance,
            interrupt_score=extracted.interrupt_score,
            deterministic_evidence=decision.deterministic_evidence,
            watch_id=decision.watch_id,
            watch_evidence=decision.watch_evidence,
            policy_action=decision.action.value,
            final_action=final_action,
            suppression_reason=decision.suppression_reason,
            inference_latency_ms=metrics.wall_latency_ms if metrics is not None else None,
            inference_mode=inference_mode,
            reason=decision.reason,
            summary=extracted.summary,
            cloud_escalation_candidate=decision.cloud_escalation_candidate,
        )
        return ProcessResult(decision, extracted)

    def process_capture(self, provider: ScreenpipeCaptureProvider) -> list[ProcessResult]:
        return self.process_coalesced([dict(item) for item in provider.poll()])
