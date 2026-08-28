from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

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
from .inference.stale import ResultFreshness, assess_result
from .inference.status import InferenceRuntimeState, LocalInferenceStatus
from .memory.intervention import classify_situation
from .memory.retrieval import retrieve_relevant_intervention_preferences, retrieve_similar_intervention_episodes
from .perception.extractor import ExtractedEvent, EventExtractor
from .policy.context import DecisionContext
from .policy.hard_rules import HardRules
from .policy.proactive import Action, Decision, PolicyThresholds, ProactivePolicy
from .policy.watch import WatchManager
from .privacy.filter import PrivacyFilter
from .runtime import RuntimeCounters
from .vision.cognition import VisualCognition
from .vision.mock_gui import MockGUIPerceptionProvider
from .vision.ollama_gui import GUIPerceptionOllamaProvider
from .vision.world import DesktopWorldState


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
        self.counters = RuntimeCounters()
        self._generation_lock = Lock()
        self._current_generation = 0
        self._current_activity: tuple[str, ...] = ()
        self._current_topic: str | None = None
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
        self.session_id = self.store.start_session(
            decision_retention_days=self.config.decision_retention_days,
            session_retention_days=self.config.session_retention_days,
        )
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
        self.cognition = self._build_visual_cognition()
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
        self._persist_gui_trajectory()
        self.store.end_session(self.session_id, {"counters": self.counters.snapshot()})
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

    def _build_visual_cognition(self) -> VisualCognition:
        """Wire GUI perception to the provider; mock stays deterministically offline."""
        if self.config.inference_provider == "ollama" and isinstance(self.inference, OllamaInferenceProvider):
            perception = GUIPerceptionOllamaProvider(self.inference)
        else:
            perception = MockGUIPerceptionProvider()
        return VisualCognition(perception, world=DesktopWorldState(), excluded_apps=self.config.excluded_apps)

    def process(self, raw: dict[str, object]) -> ProcessResult:
        if self.session.paused:
            decision = Decision(Action.IGNORE, "paused", reason_code="PAUSED")
            return ProcessResult(decision)
        event = normalize_fixture_item(raw)
        self._count_raw_and_normalized(raw)
        privacy = self.privacy.check(event)
        if privacy.blocked:
            self.counters.increment("privacy_filtered_events")
            self.logger.info("event_type=privacy-suppressed app_category=excluded policy=IGNORE")
            return ProcessResult(Decision(Action.IGNORE, "privacy excluded app", reason_code="PRIVACY_FILTERED"), privacy_suppressed=True)
        if not self.filter.accept(event):
            self.counters.increment("duplicate_events_dropped")
            return ProcessResult(Decision(Action.IGNORE, "duplicate capture frame", reason_code="DUPLICATE_DROPPED"))
        self.coalescer.add(event)
        gui_state_text, gui_trajectory_text = self._observe_gui_state(event)
        request = self.context_builder.build(
            event,
            recent_events=self.coalescer.snapshot(),
            working_state=self.state.snapshot(),
            active_hypotheses=self.watch.snapshot(),
            recent_failures=self.state.recent_failures,
            recent_assistant_decisions=self.state.decisions,
            image_path=event.image_path,
            generation_id=self._generation_value(),
            topic_snapshot=self.state.current_topic or self.state.current_objective,
            gui_state_text=gui_state_text,
            gui_trajectory_text=gui_trajectory_text,
        )
        extracted = self.extractor.extract(event, request)
        if request.use_vision:
            self.counters.increment("vision_inference_calls")
        else:
            self.counters.increment("text_inference_calls")
        self._record_inference_result()
        self.counters.increment("inference_results_received")
        return self._apply_extracted(event, extracted, inference_mode="vision" if request.use_vision else "text", request=request)

    def process_coalesced(self, raws: list[dict[str, object]]) -> list[ProcessResult]:
        """Process one capture poll synchronously for replay and unit callers."""
        results, work = self.prepare_inference_batch(raws)
        if work is None:
            return results
        extracted = self.extractor.extract(work.event, work.request)
        if work.request.use_vision:
            self.counters.increment("vision_inference_calls")
        else:
            self.counters.increment("text_inference_calls")
        self._record_inference_result()
        self.counters.increment("inference_results_received")
        results.append(self._apply_extracted(work.event, extracted, inference_mode="vision" if work.request.use_vision else "text", request=work.request))
        return results

    def prepare_inference_batch(self, raws: list[dict[str, object]]) -> tuple[list[ProcessResult], InferenceWork | None]:
        """Prepare one bounded request without calling a provider or mutating policy state."""
        if self.session.paused:
            return [ProcessResult(Decision(Action.IGNORE, "paused", reason_code="PAUSED")) for _ in raws], None
        accepted: list[NormalizedEvent] = []
        results: list[ProcessResult] = []
        if any(_is_screenpipe_raw(raw) for raw in raws):
            self.counters.increment("raw_screenpipe_items", len(raws))
        for raw in raws:
            event = normalize_fixture_item(raw)
            self.counters.increment("normalized_events")
            privacy = self.privacy.check(event)
            if privacy.blocked:
                self.counters.increment("privacy_filtered_events")
                self.logger.info("event_type=privacy-suppressed app_category=excluded policy=IGNORE")
                results.append(ProcessResult(Decision(Action.IGNORE, "privacy excluded app", reason_code="PRIVACY_FILTERED"), privacy_suppressed=True))
                continue
            if not self.filter.accept(event):
                self.counters.increment("duplicate_events_dropped")
                results.append(ProcessResult(Decision(Action.IGNORE, "duplicate capture frame", reason_code="DUPLICATE_DROPPED")))
                continue
            accepted.append(event)
        accepted.sort(key=lambda item: (item.timestamp, item.stable_id))
        for event in accepted:
            self.coalescer.add(event)
        if not accepted:
            return results, None
        self.counters.increment("coalesced_batches")
        event = accepted[-1]
        gui_state_text, gui_trajectory_text = self._observe_gui_state(event)
        request = self.context_builder.build(
            event,
            recent_events=self.coalescer.snapshot(),
            working_state=self.state.snapshot(),
            active_hypotheses=self.watch.snapshot(),
            recent_failures=self.state.recent_failures,
            recent_assistant_decisions=self.state.decisions,
            image_path=event.image_path,
            generation_id=self._generation_value(),
            topic_snapshot=self.state.current_topic or self.state.current_objective,
            gui_state_text=gui_state_text,
            gui_trajectory_text=gui_trajectory_text,
        )
        return results, InferenceWork(event=event, request=request)

    def submit_inference(self, work: InferenceWork) -> None:
        """Replace the bounded pending request with the newest prepared state."""
        if self.scheduler.pending_request is not None:
            self.counters.increment("inference_replaced")
        self.scheduler.submit(work.request)
        self.counters.increment("inference_submitted")

    def run_scheduled_inference(self, work: InferenceWork) -> ProcessResult | None:
        """Run one admitted request and apply policy on this single worker."""
        discarded_before = self.scheduler.discarded_stale_requests
        request = self.scheduler.start_next()
        if request is None:
            # A stale request is removed by start_next. A throttled request
            # remains pending and the controller will retry after the interval.
            if self.scheduler.discarded_stale_requests > discarded_before:
                self.counters.increment("inference_stale_request_dropped")
            return None
        if request.use_vision:
            self.counters.increment("vision_inference_calls")
        else:
            self.counters.increment("text_inference_calls")
        try:
            extracted = self.extractor.extract(work.event, request)
            self._record_inference_result()
            self.counters.increment("inference_results_received")
            assessment = assess_result(
                request,
                current_generation=self._generation_value(),
                current_activity=self._activity_value(),
                current_topic=self._topic_value(),
                stale_seconds=self.config.inference_stale_result_seconds,
                stale_generation_gap=self.config.inference_stale_result_generation_gap,
            )
            if assessment.freshness == ResultFreshness.STALE:
                self.counters.increment("inference_results_stale_discarded")
                result = self._discard_stale_result(work.event, extracted, request, assessment)
            elif assessment.freshness == ResultFreshness.SLIGHTLY_STALE:
                result = self._apply_slightly_stale(work.event, extracted, request, assessment)
            else:
                result = None if self.session.paused else self._apply_extracted(
                    work.event,
                    extracted,
                    inference_mode="vision" if request.use_vision else "text",
                    request=request,
                    reason_code_override=None,
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

    def note_generation(self, generation: int, raws: list[dict[str, object]]) -> None:
        """Publish capture freshness metadata without touching policy or storage."""
        events = [normalize_fixture_item(raw) for raw in raws]
        if not events:
            return
        current = max(events, key=lambda item: (item.timestamp, item.stable_id))
        with self._generation_lock:
            if generation >= self._current_generation:
                self._current_generation = generation
                self._current_activity = (current.foreground_app, current.event_source, current.window_title)
                self._current_topic = self.state.current_topic or self.state.current_objective

    def counters_snapshot(self) -> dict[str, int]:
        return self.counters.snapshot()

    def _generation_value(self) -> int:
        with self._generation_lock:
            return self._current_generation

    def _activity_value(self) -> tuple[str, ...]:
        with self._generation_lock:
            return self._current_activity

    def _topic_value(self) -> str | None:
        with self._generation_lock:
            return self._current_topic

    def _count_raw_and_normalized(self, raw: dict[str, object]) -> None:
        if _is_screenpipe_raw(raw):
            self.counters.increment("raw_screenpipe_items")
        self.counters.increment("normalized_events")

    def _record_inference_result(self) -> None:
        result = self.extractor.last_result
        if result is None or not result.error_type:
            return
        self.counters.increment("provider_failures")
        if result.error_type in {"malformed_response", "malformed_result", "structured_output_failure"}:
            self.counters.increment("structured_output_failures")

    def _observe_gui_state(self, event: NormalizedEvent) -> tuple[str, str]:
        """Spend pixels only when the keyframe gate asks; fold into world state.

        Returns bounded prompt text (current GUI state + semantic trajectory)
        for the inference request. Never receives an excluded-app event.
        """
        update = self.cognition.on_accepted_event(event, event.timestamp, skip_vision=not event.image_path)
        if update.is_same:
            return self._gui_prompt_texts()
        if update.perception_failed:
            self.counters.increment("visual_perception_failures")
            self.counters.increment("visual_perception_calls")
            return self._gui_prompt_texts()
        if update.used_vision:
            self.counters.increment("visual_keyframes")
            self.counters.increment("visual_perception_calls")
        else:
            self.counters.increment("structured_gui_updates")
        self._persist_gui_update(update, event)
        return self._gui_prompt_texts()

    def _gui_prompt_texts(self) -> tuple[str, str]:
        state = self.cognition.current_gui_state
        if state is None:
            return "", ""
        state_text = (
            f"app={state.application}; window={state.window}; activity={state.activity}; "
            f"topic={state.topic or 'none'}; progress={state.progress}"
            + (f"; errors={len(state.errors)}" if state.errors else "")
        )
        return state_text, self.cognition.recent_trajectory_text()

    def _persist_gui_update(self, update, event: NormalizedEvent) -> None:
        state = update.state
        if state is None:
            return
        delta = self.cognition.world.last_delta
        changed = delta.changed_fields if delta is not None else ()
        trajectory_label = ""
        trajectory = self.cognition.world.trajectory.snapshot()
        if trajectory:
            trajectory_label = trajectory[-1].label
        try:
            self.store.record_gui_state(
                session_id=self.session_id,
                event_timestamp=event.timestamp,
                application=state.application,
                window=state.window,
                activity=state.activity,
                topic=state.topic,
                progress=state.progress,
                task_hint=state.task_hint,
                errors=state.errors,
                confidence=state.confidence,
                perception_mode="vision" if update.used_vision else "structured",
                keyframe_reason=update.decision.reason if update.decision else "unknown",
                changed_fields=changed,
                recovery=bool(delta is not None and delta.recovery),
                regression=bool(delta is not None and delta.regression),
                trajectory_label=trajectory_label,
                generation_id=self._generation_value(),
            )
            self.counters.increment("gui_states_recorded")
            if delta is not None and delta.recovery:
                self.counters.increment("gui_recoveries")
            if delta is not None and delta.regression:
                self.counters.increment("gui_regressions")
        except Exception:
            self.logger.warning("event_type=gui_state_persist_error error_class=sqlite")

    def _persist_gui_trajectory(self) -> None:
        """Persist the accumulated semantic trajectory once (bounded session write).

        The unique index on (session_id, event_timestamp, label) makes the
        write idempotent across restarts that re-close the same session.
        """
        try:
            for event in self.cognition.world.trajectory.snapshot()[-60:]:
                if self.store.record_gui_trajectory_event(
                    session_id=self.session_id,
                    event_timestamp=event.timestamp,
                    label=event.label,
                    activity=event.activity,
                    application=event.application,
                    topic=event.topic,
                    importance=event.importance,
                ):
                    self.counters.increment("gui_trajectory_events_recorded")
        except Exception:
            self.logger.warning("event_type=gui_trajectory_persist_error error_class=sqlite")

    def _discard_stale_result(
        self,
        event: NormalizedEvent,
        extracted: ExtractedEvent,
        request: InferenceRequest,
        assessment,
    ) -> ProcessResult:
        decision = Decision(
            Action.IGNORE,
            f"inference result discarded as {assessment.freshness.value.lower()}",
            candidate_action=extracted.candidate_action,
            candidate_confidence=extracted.confidence,
            candidate_importance=extracted.importance,
            interrupt_score=extracted.interrupt_score,
            suppression_reason="stale_result",
            reason_code="STALE_RESULT",
        )
        self.state.add_decision(decision.action.value)
        self._record_trace(event, extracted, decision, "IGNORE", request, decision.reason_code)
        return ProcessResult(decision, extracted)

    def _apply_slightly_stale(
        self,
        event: NormalizedEvent,
        extracted: ExtractedEvent,
        request: InferenceRequest,
        assessment,
    ) -> ProcessResult:
        self.state.observe(extracted, allow_objective_update=False)
        action = Action.REMEMBER if (
            extracted.confidence >= self.config.model_remember_min_confidence
            and extracted.importance >= self.config.model_remember_min_importance
        ) else Action.IGNORE
        decision = Decision(
            action,
            "slightly stale result retained without escalation",
            evidence=0,
            candidate_action=extracted.candidate_action,
            candidate_confidence=extracted.confidence,
            candidate_importance=extracted.importance,
            interrupt_score=extracted.interrupt_score,
            suppression_reason="slightly_stale_low_risk" if action == Action.IGNORE else None,
            reason_code="SLIGHTLY_STALE_LOW_RISK",
        )
        self.store.record_event(extracted, source=event.source, session_id=self.session_id)
        self.state.add_decision(action.value)
        self.counters.increment(f"policy_{action.value.casefold()}")
        if action == Action.REMEMBER:
            self.store.record_memory(extracted.summary, importance=extracted.importance, tags=extracted.failure_signature or "")
        self._record_trace(event, extracted, decision, action.value, request, decision.reason_code)
        return ProcessResult(decision, extracted)

    def _record_trace(
        self,
        event: NormalizedEvent,
        extracted: ExtractedEvent,
        decision: Decision,
        final_action: str,
        request: InferenceRequest | None,
        reason_code: str,
        intervention_episode_id: int | None = None,
    ) -> None:
        result = self.extractor.last_result
        metrics = result.metrics if result is not None else None
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
            inference_mode=metrics.mode if metrics is not None else ("vision" if request and request.use_vision else "text"),
            reason=decision.reason,
            summary=extracted.summary,
            cloud_escalation_candidate=decision.cloud_escalation_candidate,
            reason_code=reason_code,
            context_chars=request.context_chars if request is not None else 0,
            context_event_count=request.context_event_count if request is not None else 0,
            context_watch_count=request.context_watch_count if request is not None else 0,
            preference_ids=decision.preference_ids,
            preference_effect=decision.preference_effect,
            similar_episode_ids=decision.similar_episode_ids,
            intervention_episode_id=intervention_episode_id,
        )

    def _apply_extracted(
        self,
        event: NormalizedEvent,
        extracted: ExtractedEvent,
        *,
        inference_mode: str = "text",
        request: InferenceRequest | None = None,
        reason_code_override: str | None = None,
    ) -> ProcessResult:
        self.state.observe(extracted)
        resolved_at = self.watch.resolved_at(extracted.failure_signature) if extracted.failure_signature else None
        failure_count = self.store.count_failures(extracted.failure_signature, since=resolved_at) if extracted.failure_signature else 0
        preferences = retrieve_relevant_intervention_preferences(self.store, extracted)
        similar_episodes = retrieve_similar_intervention_episodes(self.store, extracted)
        if preferences:
            self.counters.increment("preference_matches")
        decision = self.policy.decide_context(
            DecisionContext(
                event=extracted,
                working_state=self.state,
                now=event.timestamp,
                failure_count=failure_count + (1 if extracted.failure_signature else 0),
                activation_failure_count=failure_count + (1 if extracted.failure_signature else 0),
                active_watch=self.watch.active,
                watch_snapshot=tuple(self.watch.snapshot()),
                world_state=self.cognition.world,
                preferences=tuple(preferences),
                similar_episodes=tuple(similar_episodes),
            )
        )
        transitions = self.watch.peek_transitions()
        self.store.record_event(extracted, source=event.source, session_id=self.session_id)
        if decision.action.value.casefold() in {"ignore", "remember", "watch", "investigate"}:
            self.counters.increment(f"policy_{decision.action.value.casefold()}")
        if decision.candidate_action == Action.NOTIFY:
            self.counters.increment("policy_notify_candidate")
        if decision.candidate_action == Action.ASK_CLOUD:
            self.counters.increment("policy_ask_cloud")
        final_action = decision.action.value
        self.state.add_decision(final_action)
        if decision.action == Action.REMEMBER:
            self.store.record_memory(extracted.summary, importance=extracted.importance, tags=extracted.failure_signature or "")
        self.state.set_hypotheses(self.watch.snapshot())
        if self.watch.active:
            active = self.watch.active
            self.store.record_hypothesis(active.hypothesis, active.evidence, "watching", active.expires_at.isoformat())
        was_notified = False
        notification_id: int | None = None
        if decision.action == Action.NOTIFY:
            title = decision.notification_title or "Ambient Secretary"
            body = decision.notification_body or decision.reason
            shadow_notification = bool(getattr(self.notifier, "shadow", False))
            try:
                self.notifier.notify(title, body)
                if not shadow_notification:
                    self.hard_rules.mark_notified(extracted, event.timestamp)
                    self.counters.increment("real_notify")
                    was_notified = True
                else:
                    final_action = "WOULD_NOTIFY"
                    self.counters.increment("would_notify")
                notification_id = self.store.record_notification(title, body, final_action)
            except Exception as exc:
                self.logger.warning("event_type=notification_error error_class=%s", exc.__class__.__name__)

        intervention_episode_id: int | None = None
        if self._is_intervention_opportunity(extracted, decision):
            transition = next((item for item in transitions if item.get("watch_id") == decision.watch_id), None)
            watch_status = str(transition.get("status")) if transition else ("ACTIVE" if decision.watch_id else "RECORDED")
            watch_context = {
                "watch_id": decision.watch_id,
                "evidence": decision.watch_evidence,
                "status": watch_status,
            }
            reason_codes = [decision.reason_code]
            if decision.preference_effect:
                reason_codes.append(decision.preference_effect)
            intervention_episode_id = self.store.record_intervention_episode(
                session_id=self.session_id,
                event_timestamp=event.timestamp,
                situation_type=classify_situation(extracted.event_type, extracted.activity, extracted.failure_signature),
                activity=extracted.activity,
                event_type=extracted.event_type,
                topic=extracted.topic,
                failure_signature=extracted.failure_signature,
                summary=extracted.summary,
                watch_id=decision.watch_id,
                watch_context=watch_context,
                candidate_action=decision.candidate_action.value,
                final_action=final_action,
                reason_codes=reason_codes,
                model_confidence=decision.candidate_confidence,
                importance=decision.candidate_importance,
                interrupt_score=decision.interrupt_score,
                was_notified=was_notified,
                status=watch_status,
                outcome=str(transition.get("outcome") or "UNKNOWN") if transition else "UNKNOWN",
                notification_id=notification_id,
                preference_ids=decision.preference_ids,
            )
            self.counters.increment("intervention_episodes_recorded")
        for transition in transitions:
            watch_id = transition.get("watch_id")
            if watch_id:
                transition_status = str(transition.get("status") or "")
                if transition_status == "RESOLVED":
                    self.counters.increment("watch_resolved")
                elif transition_status == "EXPIRED":
                    self.counters.increment("watch_expired")
                self.store.update_intervention_outcome(
                    str(watch_id),
                    str(transition.get("status") or "ACTIVE"),
                    str(transition.get("outcome") or "UNKNOWN"),
                )
        self.watch.acknowledge_transitions(transitions)
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
            reason_code=reason_code_override or decision.reason_code,
            context_chars=request.context_chars if request is not None else 0,
            context_event_count=request.context_event_count if request is not None else 0,
            context_watch_count=request.context_watch_count if request is not None else 0,
            preference_ids=decision.preference_ids,
            preference_effect=decision.preference_effect,
            similar_episode_ids=decision.similar_episode_ids,
            intervention_episode_id=intervention_episode_id,
        )
        return ProcessResult(decision, extracted)

    @staticmethod
    def _is_intervention_opportunity(event: ExtractedEvent, decision: Decision) -> bool:
        """Avoid turning ordinary desktop activity into unbounded episode memory."""
        return bool(
            event.event_type in {"failure", "recovery", "success"}
            or event.candidate_action != Action.IGNORE
            or decision.action != Action.IGNORE
            or decision.reason_code.startswith("WATCH_")
        )

    def process_capture(self, provider: ScreenpipeCaptureProvider) -> list[ProcessResult]:
        return self.process_coalesced([dict(item) for item in provider.poll()])


def _is_screenpipe_raw(raw: dict[str, object]) -> bool:
    return raw.get("source") == "screenpipe" or "content" in raw or "type" in raw
