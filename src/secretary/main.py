from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .capture.lifecycle import ScreenpipeLifecycleManager
from .benchmark import run_benchmark
from .capture.mock import MockCaptureProvider
from .capture.screenpipe import ScreenpipeCaptureProvider
from .config import SecretaryConfig, ensure_project_dirs, resolve_launcher
from .controller import SecretaryController
from .engine import SecretaryEngine
from .events.normalize import normalize_fixture_item
from .inference.mock import MockInferenceProvider
from .inference.image import ImagePreprocessor
from .inference.ollama import OllamaInferenceProvider
from .inference.schema import InferenceRequest
from .inference.status import InferenceRuntimeState, LocalInferenceStatus
from .instance import InstanceLock
from .memory.intervention import parse_outcome, parse_reaction
from .memory.profile import build_secretary_profile
from .memory.store import MemoryStore
from .notifications.mock import MockNotificationProvider
from .notifications.shadow import ShadowNotificationProvider
from .notifications.windows import WindowsNotificationProvider
from .platform.windows.job_object import WindowsJobObject
from .ui.tray import TrayApplication, TrayUnavailable


class _StoreOnce(argparse.Action):
    """Reject duplicate option values instead of silently keeping the last one."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        if getattr(namespace, self.dest, None) is not None:
            raise argparse.ArgumentError(self, f"{option_string or self.dest} may only be specified once")
        setattr(namespace, self.dest, values)


def _scenario_items(path: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"scenario line {line_number} must be an object")
        items.append(value)
    return items


def _elapsed_label(timestamp: datetime, start: datetime) -> str:
    seconds = max(0, int((timestamp - start).total_seconds()))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def run_replay(path: Path, config: SecretaryConfig | None = None) -> int:
    items = _scenario_items(path)
    capture = MockCaptureProvider(items)
    replay_items = [dict(item) for item in capture.poll()]
    replay_config = config or SecretaryConfig.from_environment()
    # Replays use an explicit MockCapture-equivalent input source and isolated DB.
    engine = SecretaryEngine(replay_config, store=MemoryStore(":memory:"))
    try:
        timestamps = [normalize_fixture_item(item).timestamp for item in replay_items]
        start = min(timestamps) if timestamps else datetime.now(timezone.utc)
        for item in replay_items:
            result = engine.process(item)
            timestamp = normalize_fixture_item(item).timestamp
            suffix = f" evidence={result.decision.evidence}" if result.decision.evidence else ""
            if result.privacy_suppressed:
                suffix += " privacy-suppressed"
            print(f"{_elapsed_label(timestamp, start)} {result.decision.action.value}{suffix}")
        print(f"notifications={len(getattr(engine.notifier, 'notifications', []))}")
        return 0
    finally:
        engine.close()


def run_label_summary(config: SecretaryConfig | None = None, output=sys.stdout) -> int:
    """Show label counts only; TP/FP are computed only in evaluation benchmarks."""
    config = config or SecretaryConfig.from_environment()
    store = MemoryStore(config.database_path)
    try:
        summary = store.intervention_label_summary()
        print("Ambient Secretary Intervention Labels", file=output)
        print("\nLabeled:", file=output)
        if not summary["labels"]:
            print("  (none)", file=output)
        for label, count in sorted(summary["labels"].items()):
            print(f"  {label}: {count}", file=output)
        print(f"\nLabeled total: {summary['labeled_total']}", file=output)
        print(f"Unlabeled total: {summary['unlabeled_total']}", file=output)
        print("\nLabel vocabulary:", ", ".join(summary["suggested_feedback"]), file=output)
        return 0
    finally:
        store.close()


def run_consolidate(config: SecretaryConfig | None = None, output=sys.stdout, *, use_llm: bool = False) -> int:
    """Explicit background consolidation; never mutates source episodes."""
    config = config or SecretaryConfig.from_environment()
    store = MemoryStore(config.database_path)
    try:
        from .memory.consolidation import LLMConsolidator, MemoryConsolidator

        if use_llm:
            if config.inference_provider != "ollama":
                print("LLM consolidation requires INFERENCE_PROVIDER=ollama; using deterministic consolidation.", file=output)
                result = MemoryConsolidator(store).consolidate().as_dict()
            else:
                from .inference.completion import OllamaTextCompleter

                completer = OllamaTextCompleter(
                    base_url=config.ollama_base_url,
                    model=config.ollama_text_model,
                    timeout_seconds=config.ollama_timeout_seconds,
                    keep_alive=config.ollama_keep_alive,
                    temperature=config.ollama_temperature,
                )
                result = LLMConsolidator(store, completer.complete).consolidate().as_dict()
        else:
            result = MemoryConsolidator(store).consolidate().as_dict()
        print("Ambient Secretary Memory Consolidation", file=output)
        mode = "llm+deterministic-fallback" if use_llm else "deterministic-v1"
        print(f"\nMode: {mode}", file=output)
        print(f"Episodes considered: {result['episodes_considered']}", file=output)
        print(f"Durable memories produced: {result['memories_produced']}", file=output)
        print(f"Superseded older equivalents: {result['superseded']}", file=output)
        if result.get("skipped_reason"):
            print(f"Skipped: {result['skipped_reason']}", file=output)
        memories = store.active_memories(tier="SEMANTIC", limit=10)
        for row in memories:
            print(f"- {row['content']}  [confidence={float(row['confidence']):.2f}]", file=output)
        return 0
    finally:
        store.close()


def run_memory_doctor(config: SecretaryConfig | None = None, output=sys.stdout) -> int:
    """READ-ONLY memory hygiene diagnostics; never mutates."""
    config = config or SecretaryConfig.from_environment()
    store = MemoryStore(config.database_path)
    try:
        from .memory.doctor import diagnose

        report = diagnose(store)
        print("Ambient Secretary Memory Doctor", file=output)
        print("\nSummary:", file=output)
        print(f"  Core memory: {report.core_chars} chars / {report.core_budget} budget", file=output)
        print(f"  Active memories: {report.active_memory_count}", file=output)
        print(f"  Superseded rows retained: {report.superseded_count}", file=output)
        findings = report.findings
        if not findings:
            print("\nNo findings.", file=output)
        else:
            print("\nFindings:", file=output)
            for finding in findings:
                print(f"- [{finding.severity}] {finding.kind}: {finding.message}", file=output)
        return 0
    finally:
        store.close()


def run_evaluate(config: SecretaryConfig | None = None, output=sys.stdout) -> int:
    """Labeled intervention evaluation. Counts always shown; rates only with ground truth."""
    config = config or SecretaryConfig.from_environment()
    store = MemoryStore(config.database_path)
    try:
        summary = store.intervention_label_summary()
        from .evaluation.matrix import evaluate_labels

        labels: list[str] = []
        for episode in store.labeled_intervention_episodes(limit=500):
            label = str(episode.get("user_label") or "").strip()
            if label:
                labels.append(label)
        matrix = evaluate_labels(labels)
        print("Ambient Secretary Intervention Evaluation", file=output)
        print(f"\nLabeled opportunities: {summary['labeled_total']}", file=output)
        for label in sorted(summary["labels"]):
            print(f"  {label}: {summary['labels'][label]}", file=output)
        print("\nMatrix (labels only; no fabricated ground truth):", file=output)
        print(f"  TP: {matrix.tp}  FP: {matrix.fp}  TN: {matrix.tn}  FN: {matrix.fn}", file=output)
        print(f"  Precision: {matrix.precision if matrix.precision is not None else 'n/a'}", file=output)
        print(f"  Recall: {matrix.recall if matrix.recall is not None else 'n/a'}", file=output)
        print(f"  False alarm rate: {matrix.false_alarm_rate if matrix.false_alarm_rate is not None else 'n/a (no TN ground truth)'}", file=output)
        print(f"  Missed need rate: {matrix.missed_need_rate if matrix.missed_need_rate is not None else 'n/a'}", file=output)
        if summary["unlabeled_total"]:
            print(f"\n{summary['unlabeled_total']} unlabeled episodes; rates require human labels (or 'n/a' stays safe).", file=output)
        return 0
    finally:
        store.close()


def run_session_report(config: SecretaryConfig | None = None, output=sys.stdout) -> int:
    config = config or SecretaryConfig.from_environment()
    store = MemoryStore(config.database_path)
    try:
        report = store.latest_session_report()
        print("Ambient Secretary Session", file=output)
        if report is None:
            print("\nNo session data recorded.", file=output)
            return 0
        duration = float(report["duration_seconds"])
        print(f"\nSession: {report['session_id']}", file=output)
        print(f"Status: {report.get('status', 'UNKNOWN')}", file=output)
        print(f"Duration: {int(duration // 60):02d}:{int(duration % 60):02d}", file=output)
        print(f"Screenpipe events: {report['screenpipe_events']}", file=output)
        print(f"Semantic inference requests: {report['semantic_inference_requests']}", file=output)
        modes = report["inference_modes"]
        print(f"Text inference: {modes.get('text', 0)}", file=output)
        print(f"Vision inference: {modes.get('vision', 0)}", file=output)
        print("\nCandidate actions:", file=output)
        _print_action_counts(report["candidate_actions"], output)
        print("\nFinal actions:", file=output)
        _print_action_counts(report["final_actions"], output)
        print(f"\nSuppressed model NOTIFY: {report['suppressed_model_notify']}", file=output)
        print(f"Cloud escalation candidates (Mock only): {report['cloud_escalation_candidates']}", file=output)
        custom_metrics = store.gui_perception_stats()
        print(f"Intervention episodes: {report['intervention_episodes']}", file=output)
        print(f"Personalized decisions: {report['personalized_decisions']}", file=output)
        print("\nVisual perception:", file=output)
        print(f"GUI states recorded: {custom_metrics['gui_states']}", file=output)
        print(f"Vision VLM calls: {custom_metrics['vision_perceptions']}", file=output)
        print(f"Structured-only updates: {custom_metrics['structured_updates']}", file=output)
        print(f"GUI recoveries: {custom_metrics['recoveries']}  regressions: {custom_metrics['regressions']}", file=output)
        latency = custom_metrics["latency"]
        if latency.get("count"):
            print(f"Perception latency: median={latency['median_ms']:.1f}ms p90={latency['p90_ms']:.1f}ms p95={latency['p95_ms']:.1f}ms max={latency['max_ms']:.1f}ms", file=output)
        if report["suppression_reasons"]:
            print("Reasons:", file=output)
            for reason, count in sorted(report["suppression_reasons"].items()):
                print(f"- {reason}: {count}", file=output)
        average = report["average_latency_ms"]
        p95 = report["p95_latency_ms"]
        print(f"\nAverage inference latency: {average:.1f} ms" if average is not None else "\nAverage inference latency: n/a", file=output)
        print(f"P95 inference latency: {p95:.1f} ms" if p95 is not None else "P95 inference latency: n/a", file=output)
        counters = report.get("counters", {})
        if counters:
            print("\nRuntime funnel:", file=output)
            for name in (
                "raw_screenpipe_items", "normalized_events", "privacy_filtered_events",
                "duplicate_events_dropped", "coalesced_batches", "inference_submitted",
                "inference_replaced", "inference_results_received", "inference_results_stale_discarded",
                "policy_ignore", "policy_remember", "policy_watch", "policy_investigate",
                "policy_notify_candidate", "policy_ask_cloud", "would_notify", "real_notify",
                "intervention_episodes_recorded", "preference_matches", "watch_resolved", "watch_expired",
            ):
                if counters.get(name, 0):
                    print(f"{name}: {counters[name]}", file=output)
        print("\nLatency distribution:", file=output)
        for mode, stats in report.get("latency", {}).items():
            if stats.get("count"):
                print(f"{mode}: count={stats['count']} median={stats['median_ms']:.1f}ms p90={stats['p90_ms']:.1f}ms p95={stats['p95_ms']:.1f}ms max={stats['max_ms']:.1f}ms", file=output)
        return 0
    finally:
        store.close()


def _print_action_counts(counts: dict[str, int], output) -> None:
    for action in ("IGNORE", "REMEMBER", "WATCH", "INVESTIGATE", "ASK_CLOUD", "NOTIFY", "WOULD_NOTIFY"):
        if counts.get(action, 0):
            print(f"{action:<13}{counts[action]}", file=output)


def run_recent_decisions(config: SecretaryConfig | None = None, limit: int = 20, output=sys.stdout, *, action: str | None = None, suppressed: bool = False) -> int:
    config = config or SecretaryConfig.from_environment()
    store = MemoryStore(config.database_path)
    try:
        traces = list(reversed(store.recent_decision_traces(limit, action=action, suppressed=suppressed)))
        print("Ambient Secretary Recent Decisions", file=output)
        if not traces:
            print("\nNo decision traces recorded.", file=output)
            return 0
        for trace in traces:
            timestamp = str(trace["event_timestamp"]).replace("T", " ")[:19]
            print(f"\n{timestamp}", file=output)
            print(f"App: {trace['foreground_app']}", file=output)
            print(f"Event: {trace['event_type']}", file=output)
            print(f"Model: {trace['candidate_action']}", file=output)
            print(f"Final: {trace['final_action']}", file=output)
            print(f"reason_code={trace.get('reason_code', 'POLICY_IGNORE')}", file=output)
            print(f"confidence={float(trace['candidate_confidence']):.2f} importance={float(trace['candidate_importance']):.2f} interrupt={float(trace['interrupt_score']):.2f}", file=output)
            print(f"deterministic_evidence={trace['deterministic_evidence']} watch_evidence={trace['watch_evidence']}", file=output)
            if trace["suppression_reason"]:
                print(f"suppression={trace['suppression_reason']}", file=output)
            if trace.get("preference_effect"):
                print(f"preference_effect={trace['preference_effect']}", file=output)
            if trace.get("intervention_episode_id"):
                print(f"intervention_episode=#{trace['intervention_episode_id']}", file=output)
        return 0
    finally:
        store.close()


def run_recent_interventions(config: SecretaryConfig | None = None, limit: int = 20, output=sys.stdout) -> int:
    config = config or SecretaryConfig.from_environment()
    store = MemoryStore(config.database_path)
    try:
        episodes = list(reversed(store.recent_intervention_episodes(limit)))
        print("Ambient Secretary Intervention Episodes", file=output)
        if not episodes:
            print("\nNo intervention episodes recorded.", file=output)
            return 0
        for episode in episodes:
            timestamp = str(episode.get("event_timestamp") or "").replace("T", " ")[:19]
            print(f"\n#{episode['id']} {timestamp}", file=output)
            print(f"Situation: {episode['situation_type']} / {episode['activity']}", file=output)
            print(f"Event: {episode['event_type']}", file=output)
            print(f"Final: {episode['final_action']}  status={episode['status']} outcome={episode['outcome']}", file=output)
            print(f"Notified: {'yes' if episode['was_notified'] else 'no'}  reaction={episode['user_reaction']}", file=output)
            reason_codes = episode.get("reason_codes") or []
            if reason_codes:
                print(f"Reasons: {', '.join(str(code) for code in reason_codes)}", file=output)
            if episode.get("explicit_feedback"):
                feedback_label = str(episode["explicit_feedback"]).split(":", 1)[0]
                if feedback_label not in {"USEFUL", "MORE_PROACTIVE", "DONT_REMIND", "TIMING_BAD", "FORGET"}:
                    feedback_label = "USER_NOTE_RECORDED"
                print(f"Feedback: {feedback_label}", file=output)
        return 0
    finally:
        store.close()


def run_feedback(
    config: SecretaryConfig | None = None,
    episode_id: int | None = None,
    value: str | None = None,
    output=sys.stdout,
    *,
    reaction: str | None = None,
    outcome: str | None = None,
    note: str = "",
    timing: str | None = None,
    content: str | None = None,
) -> int:
    args_timing, args_content = timing, content
    if episode_id is None:
        print("Feedback requires an intervention episode id.", file=output)
        return 2
    if value is None or not value.strip():
        value = "OBSERVED" if (reaction or outcome or args_timing or args_content) else None
    if value is None:
        print("Feedback requires a value such as useful, dont-remind, more-proactive, or timing-bad.", file=output)
        return 2
    config = config or SecretaryConfig.from_environment()
    store: MemoryStore | None = None
    try:
        store = MemoryStore(config.database_path)
        # Validate CLI values before mutating the database and keep the accepted
        # vocabulary visible to callers using this function as an internal API.
        if reaction is not None:
            parse_reaction(reaction)
        if outcome is not None:
            parse_outcome(outcome)
        if args_timing or args_content:
            from .memory.intervention import normalize_content_feedback, normalize_timing_feedback

            normalize_timing_feedback(args_timing)
            normalize_content_feedback(args_content)
            result = store.record_dimensional_feedback(
                int(episode_id),
                timing=args_timing,
                content=args_content,
                note=note,
            )
            print(f"Recorded dimensional feedback for episode #{result['episode_id']}.", file=output)
            if result["timing"]:
                print(f"  timing: {result['timing']}", file=output)
            if result["content"]:
                print(f"  content: {result['content']}", file=output)
            for memory_id in result["knowledge_memory_ids"]:
                print(f"  learned memory #{memory_id}", file=output)
            return 0
        result = store.record_intervention_feedback(
            int(episode_id),
            value,
            reaction=reaction,
            outcome=outcome,
            note=note,
        )
        print(f"Recorded feedback for episode #{result['episode_id']}: {result['value']}", file=output)
        if result["preference_id"] is not None:
            print(f"Active preference #{result['preference_id']} updated.", file=output)
        return 0
    except (ValueError, OverflowError, sqlite3.Error) as exc:
        print(f"Feedback not recorded: {exc}", file=output)
        return 2
    finally:
        if store is not None:
            store.close()


def run_secretary_profile(config: SecretaryConfig | None = None, output=sys.stdout) -> int:
    config = config or SecretaryConfig.from_environment()
    store = MemoryStore(config.database_path)
    try:
        profile = build_secretary_profile(store.active_intervention_preferences(limit=200))
        print("Ambient Secretary Profile", file=output)
        print(f"\nGeneral: {profile.general}", file=output)
        print(f"Learned rules: {profile.source_count}", file=output)
        for rule in profile.rules:
            print(f"- {rule}", file=output)
        return 0
    finally:
        store.close()


def run_pending_labels(config: SecretaryConfig | None = None, output=sys.stdout, *, notify_only: bool = True, limit: int = 20) -> int:
    """List unlabeled shadow intervention opportunities for human review."""
    config = config or SecretaryConfig.from_environment()
    store = MemoryStore(config.database_path)
    try:
        episodes = store.unlabeled_intervention_episodes(notify_only=notify_only, limit=limit)
        print("Ambient Secretary Unlabeled Interventions", file=output)
        if not episodes:
            print("\nNo unlabeled intervention episodes.", file=output)
            return 0
        for episode in episodes:
            timestamp = str(episode.get("event_timestamp") or "").replace("T", " ")[:19]
            print(f"\n#{episode['id']} {timestamp}", file=output)
            print(f"  Situation: {episode['situation_type']} / {episode['activity']}", file=output)
            print(f"  Final: {episode['final_action']}  notified={'yes' if episode['was_notified'] else 'no'}", file=output)
            reason_codes = episode.get("reason_codes") or []
            if reason_codes:
                print(f"  Reasons: {', '.join(str(code) for code in reason_codes)}", file=output)
            if episode.get("summary"):
                print(f"  Summary: {episode['summary'][:160]}", file=output)
            print(f"  Label with: secretary label {episode['id']} <useful|not-useful|needed-but-bad-timing|not-needed|unsure>", file=output)
        return 0
    finally:
        store.close()


def run_label_episode(config: SecretaryConfig | None = None, episode_id: int | None = None, value: str | None = None, output=sys.stdout, *, note: str = "") -> int:
    """Attach a human truth label to one intervention episode."""
    if episode_id is None or value is None:
        print("label requires an episode id and a value (useful, not-useful, needed-but-bad-timing, not-needed, unsure).", file=output)
        return 2
    config = config or SecretaryConfig.from_environment()
    store: MemoryStore | None = None
    try:
        from .memory.intervention import normalize_label

        label = normalize_label(value)
        store = MemoryStore(config.database_path)
        result = store.label_intervention_episode(int(episode_id), label, note=note)
        if result is None:
            print(f"Intervention episode #{episode_id} not found.", file=output)
            return 2
        if result.get("already_labeled"):
            print(f"Episode #{episode_id} already labeled: {result['already_labeled']} (unchanged).", file=output)
            return 0
        print(f"Labeled episode #{episode_id}: {result['user_label']}", file=output)
        return 0
    except (ValueError, OverflowError, sqlite3.Error) as exc:
        print(f"Label not recorded: {exc}", file=output)
        return 2
    finally:
        if store is not None:
            store.close()


def run_current_state(config: SecretaryConfig | None = None, output=sys.stdout) -> int:
    config = config or SecretaryConfig.from_environment()
    store = MemoryStore(config.database_path)
    try:
        state = store.latest_gui_state()
        print("Ambient Secretary Current Visual State", file=output)
        if state is None:
            print("\nNo visual state recorded yet.", file=output)
            return 0
        print(f"\nApplication: {state['application']}", file=output)
        print(f"Window: {state['window'] or '(none)'}", file=output)
        print(f"Activity: {state['activity']}", file=output)
        print(f"Topic: {state['topic'] or '(none)'}", file=output)
        print(f"Task: {state['task_hint'] or '(none)'}", file=output)
        print(f"Progress: {state['progress']}  confidence={float(state['confidence']):.2f}", file=output)
        errors = state.get("errors") or []
        if errors:
            print(f"Errors: {len(errors)}", file=output)
            for error in errors[:3]:
                print(f"- {error}", file=output)
        if state.get("delta_recovery"):
            print("Recovery: yes", file=output)
        trajectory = store.recent_trajectory_events(limit=20)
        if trajectory:
            print("\nRecent trajectory:", file=output)
            for item in trajectory[-10:]:
                timestamp = str(item["event_timestamp"]).replace("T", " ")[:16]
                print(f"{timestamp}  {item['label']}", file=output)
        watch = store.recent_gui_states(limit=6)
        if watch:
            print("\nLast transitions:", file=output)
            for item in watch:
                timestamp = str(item["event_timestamp"]).replace("T", " ")[:16]
                fields = ", ".join(item.get("changed_fields") or [])
                print(f"{timestamp}  [{item['perception_mode']}] {fields or 'no significant change'}", file=output)
        return 0
    finally:
        store.close()


def run_gui_trajectory(config: SecretaryConfig | None = None, minutes: int = 20, output=sys.stdout, *, limit: int = 100) -> int:
    config = config or SecretaryConfig.from_environment()
    store = MemoryStore(config.database_path)
    try:
        events = store.recent_trajectory_events(limit=max(1, min(500, limit)))
        cutoff = datetime.now(timezone.utc).timestamp() - max(1, minutes) * 60
        eligible = [item for item in events if _parse_db_timestamp(item["event_timestamp"]) >= cutoff]
        print("Ambient Secretary Semantic Trajectory", file=output)
        if not eligible:
            print(f"\nNo semantic trajectory recorded in the last {minutes} minute(s).", file=output)
            return 0
        for item in eligible:
            timestamp = str(item["event_timestamp"]).replace("T", " ")[:16]
            topic = f" ({item['topic']})" if item.get("topic") else ""
            print(f"{timestamp}  {item['label']}{topic}", file=output)
        return 0
    finally:
        store.close()


def _parse_db_timestamp(value: object) -> float:
    from datetime import datetime as _dt

    text = str(value).replace("Z", "+00:00")
    try:
        parsed = _dt.fromisoformat(text)
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _command_has_version(command: tuple[str, ...], version: str) -> bool:
    return any(f"screenpipe@{version}" == token for token in command)


def run_preflight(config: SecretaryConfig | None = None, output=sys.stdout) -> bool:
    config = config or SecretaryConfig.from_environment()
    ensure_project_dirs(config)
    print("Ambient Secretary Preflight\n", file=output)

    real_capture = config.capture_provider == "screenpipe"
    launcher = resolve_launcher(config.screenpipe_command) is not None
    version_ok = _command_has_version(config.screenpipe_command, "0.4.41")
    key_available = bool(config.screenpipe_api_key)
    managed = config.screenpipe_mode == "managed"
    inference_configured = config.inference_provider in {"mock", "ollama"}
    inference_label = "Local inference: MOCK" if config.inference_provider == "mock" else f"Local inference: {config.inference_provider}"
    checks: list[tuple[str, str, bool, bool, str]] = [
        ("OK" if platform.system() == "Windows" else "INFO", "Windows", True, False, platform.system()),
        ("OK", "Python", sys.version_info >= (3, 10), True, platform.python_version()),
        ("OK", "Project directory writable", os.access(config.project_root, os.W_OK), True, str(config.project_root)),
        ("OK", "SQLite", _sqlite_check(), True, "stdlib"),
        ("OK" if config.inference_provider == "mock" else "INFO", inference_label, inference_configured, config.inference_provider not in {"mock", "ollama"}, config.inference_provider),
        ("OK", "Mock cloud", config.cloud_provider == "mock", True, config.cloud_provider),
    ]
    if real_capture:
        checks.extend([
            ("OK", "Screenpipe CLI launcher available", launcher, True, config.screenpipe_command[0]),
            ("OK", "Configured Screenpipe version: 0.4.41", version_ok, True, "pinned command"),
            ("OK", "SCREENPIPE_API_KEY available", key_available, True, "environment only"),
            ("OK", "Managed lifecycle enabled", managed, True, config.screenpipe_mode),
            ("OK", "Managed Screenpipe audio disabled", "--disable-audio" in config.screenpipe_command, managed, "record flag"),
            ("OK", "Managed Screenpipe clipboard capture disabled", "--disable-clipboard-capture" in config.screenpipe_command, managed, "record flag"),
            ("OK", "Managed Screenpipe excluded windows configured", (not config.excluded_apps) or "--ignored-windows" in config.screenpipe_command, managed, ",".join(config.excluded_apps)),
        ])
        provider = ScreenpipeCaptureProvider(config.screenpipe_base_url, config.screenpipe_api_key)
        # Without credentials there is no useful authenticated readiness check;
        # avoid even a localhost probe in the portable/unit path.
        healthy = provider.health() if key_available else False
        authenticated = provider.authenticated_search(limit=1) if healthy and key_available else False
        if healthy and authenticated:
            checks.append(("OK", "Screenpipe currently running", True, False, "health + authenticated search"))
            checks.append(("OK", "Audio disabled", provider.audio_disabled(), True, "health"))
            runtime_ready = True
        elif provider.last_error_kind == "connection_refused":
            checks.append(("INFO", "Screenpipe currently stopped", True, False, "127.0.0.1:3030"))
            can_start = managed and launcher and version_ok and key_available
            checks.append(("OK", "Secretary can start it when needed", can_start, True, "managed lifecycle"))
            runtime_ready = can_start
        else:
            checks.append(("WARN", "Screenpipe runtime unavailable", False, True, provider.last_error or "not ready"))
            runtime_ready = False
        capture_ready = runtime_ready
    else:
        capture_ready = True
        checks.append(("INFO", "Mock capture explicitly selected", True, False, "explicit test/replay mode"))

    hard_fail = False
    for level, name, ok, required, detail in checks:
        shown_level = level if ok else ("WARN" if not required else "WARN")
        print(f"[{shown_level}] {name}", file=output)
        if not ok:
            print(f"       {detail}", file=output)
            if required:
                hard_fail = True
    if config.inference_provider == "mock":
        print("[INFO] Ollama not required in current configuration", file=output)
    elif config.inference_provider == "ollama":
        print("[INFO] Ollama runtime not checked by offline preflight", file=output)
    print(f"\nCapture readiness: {'PASS' if capture_ready else 'FAIL'}", file=output)
    print(f"Development readiness: {'PASS' if not hard_fail and capture_ready else 'FAIL'}", file=output)
    return not hard_fail and capture_ready


def run_inference_status(config: SecretaryConfig | None = None, output=sys.stdout, probe: bool = False) -> int:
    """Report configured local inference without probing any model runtime."""
    config = config or SecretaryConfig.from_environment()
    if config.inference_provider == "mock":
        provider = MockInferenceProvider()
    elif config.inference_provider == "ollama":
        provider = OllamaInferenceProvider(
            base_url=config.ollama_base_url,
            text_model=config.ollama_text_model,
            vision_model=config.ollama_vision_model,
            timeout_seconds=config.ollama_timeout_seconds,
            keep_alive=config.ollama_keep_alive,
            temperature=config.ollama_temperature,
            think=config.ollama_think,
        )
    else:
        print(f"Local inference\n\nProvider: {config.inference_provider}\nStatus: DEGRADED\nReal model required: unknown", file=output)
        return 1
    status: LocalInferenceStatus = provider.status()
    print("Local Inference", file=output)
    print(f"\nProvider: {status.provider}", file=output)
    print(f"Status: {status.status.value}", file=output)
    if status.model:
        print(f"Configured model: {status.model}", file=output)
    print(f"Real model required: {'yes' if status.real_model_required else 'no'}", file=output)
    if probe and config.inference_provider == "mock":
        print("Runtime probe: skipped for MockInferenceProvider", file=output)
    elif probe and config.inference_provider == "ollama":
        result = provider.probe()
        print(f"Runtime status: {result.status.value}", file=output)
        print(f"Ollama version: {result.version or 'unavailable'}", file=output)
        print(f"Configured model available: {'yes' if result.model_available else 'no'}", file=output)
        if result.detail:
            print(f"Probe detail: {result.detail}", file=output)
        return 0 if result.status == InferenceRuntimeState.READY else 1
    return 0


def run_inference_smoke(config: SecretaryConfig | None, *, text: bool, image_path: Path | None, output=sys.stdout) -> int:
    """Run one explicit Ollama request without starting Screenpipe."""
    config = config or SecretaryConfig.from_environment()
    if config.inference_provider != "ollama":
        print("Inference smoke requires INFERENCE_PROVIDER=ollama; no Mock fallback is used.", file=output)
        return 2
    if not text and image_path is None:
        print("Vision smoke requires an explicit image path.", file=output)
        return 2

    provider = OllamaInferenceProvider(
        base_url=config.ollama_base_url,
        text_model=config.ollama_text_model,
        vision_model=config.ollama_vision_model,
        timeout_seconds=config.ollama_timeout_seconds,
        keep_alive=config.ollama_keep_alive,
        temperature=config.ollama_temperature,
        think=config.ollama_think,
        image_preprocessor=ImagePreprocessor(config.vision_max_long_edge, config.vision_jpeg_quality),
    )
    use_vision = not text
    if use_vision and provider.image_preprocessor.prepare_image(image_path) is None:
        print("Vision smoke image preprocessing failed safely.", file=output)
        return 2
    smoke_context = (
        "CURRENT EVENT\n"
        "App: WindowsTerminal.exe\n"
        "User ran pytest and received an AssertionError twice.\n\n"
        "RECENT TRAJECTORY\n"
        "VSCode editing test_attention.py\n"
        "Terminal pytest failed\n"
        "VSCode changed attention mask\n"
        "Terminal pytest failed again"
    )
    raw_event = {
        "source": "smoke",
        "foreground_app": "WindowsTerminal.exe",
        "window_title": "Secretary safe smoke fixture",
        "event_source": "fixture",
        "text": "pytest AssertionError repeated while debugging attention test",
        "visual_required": use_vision,
    }
    event = normalize_fixture_item(raw_event)
    request = InferenceRequest(
        current_event=event,
        image_path=str(image_path) if image_path else None,
        use_vision=use_vision,
        context_text=smoke_context,
    )
    result = provider.analyze(request)
    mode = "vision" if use_vision else "text"
    print("Inference smoke", file=output)
    print(f"\nProvider: {result.provider}", file=output)
    print(f"Model: {result.model or 'unconfigured'}", file=output)
    print(f"Mode: {mode}", file=output)
    if result.error_type:
        print(f"Result: DEGRADED ({result.error_type})", file=output)
    else:
        print(f"event_type: {result.event.event_type}", file=output)
        print(f"activity: {result.event.activity}", file=output)
        print(f"topic: {result.event.topic or 'none'}", file=output)
        print(f"importance: {result.event.importance:.3f}", file=output)
        print(f"confidence: {result.event.confidence:.3f}", file=output)
        print(f"candidate_action: {result.secretary.candidate_action.value}", file=output)
        print(f"interrupt_score: {result.secretary.interrupt_score:.3f}", file=output)
    if result.metrics:
        print("\nMetrics:", file=output)
        for key, value in result.metrics.as_dict().items():
            print(f"{key}: {value}", file=output)
    return 0 if result.error_type is None else 1


def _sqlite_check() -> bool:
    try:
        store = MemoryStore(":memory:")
        store.close()
        return True
    except Exception:
        return False


def run_live(config: SecretaryConfig, args: argparse.Namespace) -> int:
    lock = InstanceLock(config.paths.runtime_root / "secretary.lock")
    if not lock.acquire():
        print("Another Ambient Secretary instance is already running.")
        return 2
    try:
        return _run_live_unlocked(config, args)
    finally:
        lock.release()


def _run_live_unlocked(config: SecretaryConfig, args: argparse.Namespace) -> int:
    mock_capture = args.mock_capture or config.capture_provider == "mock"
    shadow = bool(getattr(args, "shadow", False) or config.shadow_mode)
    if shadow:
        if config.inference_provider != "ollama":
            print("Shadow mode requires INFERENCE_PROVIDER=ollama; no Mock inference run was started.")
            return 2
        notifier = ShadowNotificationProvider()
        print("Shadow notification mode enabled: final NOTIFY is recorded as WOULD_NOTIFY; no Toast is shown.")
    else:
        notifier = MockNotificationProvider() if args.mock_notifications else WindowsNotificationProvider()
    lifecycle: ScreenpipeLifecycleManager | None = None
    if mock_capture:
        print("Mock capture explicitly enabled; no desktop observation will be performed.")
        capture: ScreenpipeCaptureProvider | MockCaptureProvider = MockCaptureProvider()
    else:
        provider = ScreenpipeCaptureProvider(config.screenpipe_base_url, config.screenpipe_api_key)
        mode = "external" if args.external else ("managed" if args.managed else config.screenpipe_mode)
        lifecycle = ScreenpipeLifecycleManager(
            provider,
            mode,
            config.screenpipe_command,
            job_factory=WindowsJobObject if mode == "managed" and os.name == "nt" else None,
            ready_timeout=config.screenpipe_ready_timeout,
        )
        capture = provider

    engine = SecretaryEngine(config, notifier=notifier)
    controller = SecretaryController(
        engine,
        capture,
        lifecycle=lifecycle,
        poll_interval=args.interval,
        supervision_interval=config.screenpipe_supervision_interval,
    )
    try:
        status = controller.start()
        if mock_capture:
            print(f"Capture=MOCK (worker_alive={status.worker_alive}).")
        elif status.capture_status != "READY":
            print(f"capture_status=DEGRADED\nScreen perception unavailable: {status.error or 'Screenpipe is not ready'}")
        else:
            print(f"Capture=REAL Screenpipe (owned_by_secretary={status.owned_by_secretary}).")

        def status_text() -> str:
            inference_status = engine.inference_status()
            return "\n".join((
                "Ambient Secretary",
                f"Capture: {controller.status().capture_status}",
                f"Screenpipe: {controller.status().capture_status if lifecycle is not None else 'NOT_USED'}",
                f"Local AI: {inference_status.status.value}",
                f"Cloud: {config.cloud_provider.upper()}",
                f"Watching: {'yes' if engine.watch.active else 'no'}",
            ))

        if args.tray:
            try:
                TrayApplication(controller.pause, controller.resume, status_text, controller.quit).run()
                return 0
            except TrayUnavailable as exc:
                print(f"Tray unavailable: {exc}")
        while True:
            if args.once:
                controller.wait_for_first_poll(timeout=max(2.0, args.interval + 1.0))
                inference_timeout = getattr(getattr(engine, "inference", None), "timeout_seconds", 0.0)
                try:
                    drain_timeout = max(5.0, float(inference_timeout) + 5.0)
                except (TypeError, ValueError):
                    drain_timeout = 5.0
                controller.wait_for_inference_idle(timeout=drain_timeout)
                return 0 if controller.status().capture_status in {"READY", "MOCK"} else 2
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        controller.quit()
        if config.inference_provider == "ollama":
            final_inference_status = engine.inference_status()
            if final_inference_status.last_mode:
                metrics = final_inference_status.last_metrics
                metric_text = ""
                if metrics:
                    metric_text = " " + " ".join(f"{key}={value}" for key, value in metrics.as_dict().items())
                print(f"Local AI={final_inference_status.status.value} mode={final_inference_status.last_mode}{metric_text}")
        engine.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="secretary", description="Ambient Secretary privacy-first work companion")
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay", help="replay a deterministic JSONL scenario")
    replay.add_argument("scenario", type=Path)
    sub.add_parser("preflight", help="check development and real-capture readiness")
    sub.add_parser("doctor", help="offline diagnostics for local dependencies and configuration")
    status_parser = sub.add_parser("inference-status", help="show configured local inference")
    status_parser.add_argument("--probe", action="store_true", help="explicitly query Ollama version, tags, and configured model")
    smoke = sub.add_parser("inference-smoke", help="run one explicit local Ollama inference without Screenpipe")
    smoke_mode = smoke.add_mutually_exclusive_group(required=True)
    smoke_mode.add_argument("--text", action="store_true", help="run the safe text smoke context")
    smoke_mode.add_argument("--vision", type=Path, metavar="IMAGE", help="run vision smoke with this explicit image path")
    run = sub.add_parser("run", help="observe real Screenpipe by default")
    run.add_argument("--once", action="store_true", help="poll once and exit")
    run.add_argument("--interval", type=float, default=2.0)
    lifecycle = run.add_mutually_exclusive_group()
    lifecycle.add_argument("--managed", action="store_true", help="use managed Screenpipe lifecycle")
    lifecycle.add_argument("--external", action="store_true", help="reuse an existing Screenpipe only")
    run.add_argument("--mock-capture", action="store_true", help="explicit test-only fake capture mode")
    run.add_argument("--mock-notifications", action="store_true", help="explicit test-only mock notification mode")
    run.add_argument("--shadow", action="store_true", help="real capture/inference/policy with notifications suppressed")
    run.add_argument("--notify", action="store_true", help="compatibility flag; real Toast is the default")
    run.add_argument("--tray", action="store_true", help="run the optional real system tray shell")
    sub.add_parser("session-report", help="show the latest bounded session decision report")
    recent = sub.add_parser("recent-decisions", help="show recent bounded decision traces")
    recent.add_argument("--limit", type=int, default=20)
    recent.add_argument("--action", choices=("IGNORE", "REMEMBER", "WATCH", "INVESTIGATE", "ASK_CLOUD", "NOTIFY", "WOULD_NOTIFY"))
    recent.add_argument("--suppressed", action="store_true")
    pending = sub.add_parser("pending-labels", help="list unlabeled shadow intervention opportunities")
    pending.add_argument("--limit", type=int, default=20)
    pending.add_argument("--all", dest="notify_only", action="store_false", help="include non-notify episodes too")
    label_parser = sub.add_parser("label", help="attach a human truth label to an intervention episode")
    label_parser.add_argument("episode_id", type=int, metavar="EPISODE_ID")
    label_parser.add_argument("value", metavar="VALUE", help="useful, not-useful, needed-but-bad-timing, not-needed, or unsure")
    label_parser.add_argument("--note", default="", help="optional short bounded note, no secrets")
    labels =     sub.add_parser("label-summary", help="show intervention label counts (evaluation-safe, no fabricated TP/FP)")
    consolidate = sub.add_parser("consolidate", help="run one background memory consolidation (deferred, safe)")
    consolidate.add_argument("--llm", action="store_true", help="use the local text model with strict validation")
    sub.add_parser("memory-doctor", help="read-only memory hygiene diagnostics (dry run)")
    sub.add_parser("evaluate", help="intervention evaluation from human labels (safe: no fabricated TP/FP)")
    interventions = sub.add_parser("recent-interventions", help="show recent bounded intervention episodes")
    interventions.add_argument("--limit", type=int, default=20)
    feedback = sub.add_parser("feedback", help="record explicit feedback for an intervention episode")
    feedback.add_argument("episode_id_arg", nargs="?", metavar="EPISODE_ID")
    feedback.add_argument("value_arg", nargs="?", metavar="VALUE")
    feedback.add_argument("--episode", "--episode-id", dest="episode_option", type=int, action=_StoreOnce, help="intervention episode id")
    feedback.add_argument("--value", dest="value_option", action=_StoreOnce, help="useful, dont-remind, more-proactive, timing-bad, or forget")
    feedback.add_argument("--reaction", help="observed reaction such as opened, followed, or dismissed")
    feedback.add_argument("--outcome", help="observed outcome such as resolved, unresolved, or deferred")
    feedback.add_argument("--timing", default=None, help="timing feedback: good, too-early, too-late, bad, silent")
    feedback.add_argument("--content", default=None, help="content feedback: relevant, irrelevant, already-knew, wrong, useful, too-generic")
    feedback.add_argument("--note", default="", help="short bounded note; do not include secrets or raw screen content")
    sub.add_parser("profile", help="show the compact explainable secretary profile")
    state_parser = sub.add_parser("current-state", help="show the current sanitized semantic GUI state")
    state_parser.add_argument("--limit", type=int, default=None, help="bind recent transitions shown")
    trajectory = sub.add_parser("trajectory", help="show the recent semantic trajectory")
    trajectory.add_argument("--last", type=int, default=20, metavar="MINUTES", help="window in minutes (default 20)")
    sub.add_parser("benchmark", help="run the deterministic ten-scenario CPU benchmark")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = SecretaryConfig.from_environment()
    if args.command == "replay":
        raise SystemExit(run_replay(args.scenario, config))
    if args.command in {"preflight", "doctor"}:
        raise SystemExit(0 if run_preflight(config) else 1)
    if args.command == "inference-status":
        raise SystemExit(run_inference_status(config, probe=args.probe))
    if args.command == "inference-smoke":
        raise SystemExit(run_inference_smoke(config, text=args.text, image_path=args.vision))
    if args.command == "session-report":
        raise SystemExit(run_session_report(config))
    if args.command == "recent-decisions":
        raise SystemExit(run_recent_decisions(config, limit=args.limit, action=args.action, suppressed=args.suppressed))
    if args.command == "recent-interventions":
        raise SystemExit(run_recent_interventions(config, limit=args.limit))
    if args.command == "pending-labels":
        raise SystemExit(run_pending_labels(config, limit=args.limit, notify_only=args.notify_only))
    if args.command == "label":
        raise SystemExit(run_label_episode(config, args.episode_id, args.value, note=args.note))
    if args.command == "label-summary":
        raise SystemExit(run_label_summary(config))
    if args.command == "consolidate":
        raise SystemExit(run_consolidate(config, use_llm=args.llm))
    if args.command == "memory-doctor":
        raise SystemExit(run_memory_doctor(config))
    if args.command == "evaluate":
        raise SystemExit(run_evaluate(config))
    if args.command == "feedback":
        if args.episode_option is not None and (args.value_arg is not None or (args.value_option is not None and args.episode_id_arg is not None)):
            parser.error("feedback accepts one episode id and one value; do not mix duplicate positional and option values")
        if args.episode_option is not None:
            episode_id = args.episode_option
            value = args.value_option or args.value_arg or args.episode_id_arg
        else:
            if args.value_option is not None and args.value_arg is not None:
                parser.error("feedback accepts either VALUE or --value, not both")
            episode_id = args.episode_id_arg
            value = args.value_option or args.value_arg
        raise SystemExit(run_feedback(config, episode_id, value, reaction=args.reaction, outcome=args.outcome, note=args.note, timing=args.timing, content=args.content))
    if args.command == "profile":
        raise SystemExit(run_secretary_profile(config))
    if args.command == "current-state":
        raise SystemExit(run_current_state(config, output=sys.stdout))
    if args.command == "trajectory":
        raise SystemExit(run_gui_trajectory(config, minutes=args.last))
    if args.command == "benchmark":
        raise SystemExit(run_benchmark(config))
    if args.command == "run":
        raise SystemExit(run_live(config, args))


if __name__ == "__main__":
    main()
