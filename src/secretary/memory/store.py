from __future__ import annotations

import sqlite3
import math
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .intervention import (
    InterventionOutcome,
    InterventionStatus,
    PreferenceKind,
    PreferenceSource,
    UserReaction,
    build_scope_key,
    normalize_feedback,
    parse_outcome,
    parse_reaction,
)
from ..events.schema import sanitize_failure_signature, sanitize_semantic_label
from ..perception.extractor import ExtractedEvent


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    app TEXT NOT NULL,
    event_type TEXT NOT NULL,
    activity TEXT NOT NULL,
    summary TEXT NOT NULL,
    importance REAL NOT NULL,
    novelty REAL NOT NULL,
    confidence REAL NOT NULL,
    failure_signature TEXT,
    session_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_failure ON events(failure_signature);
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    objective TEXT,
    summary TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    tags TEXT NOT NULL DEFAULT ''
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content, tags, content='memories', content_rowid='id');
CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    evidence INTEGER NOT NULL DEFAULT 0,
    expires_at TEXT
);
CREATE TABLE IF NOT EXISTS assistant_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    evidence INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    action TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    notification_id INTEGER,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cloud_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    summary_json TEXT,
    owner_pid INTEGER
);
CREATE TABLE IF NOT EXISTS decision_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    created_at TEXT NOT NULL,
    event_timestamp TEXT NOT NULL,
    foreground_app TEXT NOT NULL,
    event_type TEXT NOT NULL,
    candidate_action TEXT NOT NULL,
    candidate_confidence REAL NOT NULL,
    candidate_importance REAL NOT NULL,
    interrupt_score REAL NOT NULL,
    deterministic_evidence INTEGER NOT NULL DEFAULT 0,
    watch_id TEXT,
    watch_evidence INTEGER NOT NULL DEFAULT 0,
    policy_action TEXT NOT NULL,
    final_action TEXT NOT NULL,
    suppression_reason TEXT,
    inference_latency_ms REAL,
    inference_mode TEXT,
    reason TEXT NOT NULL,
    summary TEXT NOT NULL,
    cloud_escalation_candidate INTEGER NOT NULL DEFAULT 0,
    reason_code TEXT NOT NULL DEFAULT 'POLICY_IGNORE',
    context_chars INTEGER NOT NULL DEFAULT 0,
    context_event_count INTEGER NOT NULL DEFAULT 0,
    context_watch_count INTEGER NOT NULL DEFAULT 0,
    preference_ids TEXT NOT NULL DEFAULT '[]',
    preference_effect TEXT,
    similar_episode_ids TEXT NOT NULL DEFAULT '[]',
    intervention_episode_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_decision_traces_session ON decision_traces(session_id, id);
CREATE INDEX IF NOT EXISTS idx_decision_traces_time ON decision_traces(event_timestamp);
CREATE TABLE IF NOT EXISTS intervention_episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    event_timestamp TEXT NOT NULL,
    situation_type TEXT NOT NULL,
    activity TEXT NOT NULL,
    event_type TEXT NOT NULL,
    topic TEXT,
    failure_signature TEXT,
    summary TEXT NOT NULL DEFAULT '',
    watch_id TEXT,
    watch_context TEXT NOT NULL DEFAULT '{}',
    candidate_action TEXT NOT NULL,
    final_action TEXT NOT NULL,
    reason_codes TEXT NOT NULL DEFAULT '[]',
    model_confidence REAL NOT NULL DEFAULT 0.0,
    importance REAL NOT NULL DEFAULT 0.0,
    interrupt_score REAL NOT NULL DEFAULT 0.0,
    was_notified INTEGER NOT NULL DEFAULT 0,
    notification_id INTEGER,
    status TEXT NOT NULL DEFAULT 'RECORDED',
    user_reaction TEXT NOT NULL DEFAULT 'UNKNOWN',
    outcome TEXT NOT NULL DEFAULT 'UNKNOWN',
    explicit_feedback TEXT,
    preference_ids TEXT NOT NULL DEFAULT '[]',
    learned_preference_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_intervention_episodes_time ON intervention_episodes(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_intervention_episodes_watch ON intervention_episodes(watch_id, status);
CREATE TABLE IF NOT EXISTS intervention_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    scope_key TEXT NOT NULL,
    situation_type TEXT NOT NULL,
    activity TEXT NOT NULL,
    event_type TEXT NOT NULL,
    topic TEXT,
    failure_signature TEXT,
    preference TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    last_episode_id INTEGER,
    supersedes_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_intervention_preferences_active ON intervention_preferences(status, scope_key);
CREATE INDEX IF NOT EXISTS idx_intervention_preferences_context ON intervention_preferences(situation_type, event_type, activity);
CREATE TABLE IF NOT EXISTS gui_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    created_at TEXT NOT NULL,
    event_timestamp TEXT NOT NULL,
    application TEXT NOT NULL,
    window TEXT NOT NULL DEFAULT '',
    activity TEXT NOT NULL DEFAULT 'desktop',
    topic TEXT,
    progress TEXT NOT NULL DEFAULT 'unknown',
    task_hint TEXT,
    errors_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.0,
    perception_mode TEXT NOT NULL DEFAULT 'structured',
    keyframe_reason TEXT NOT NULL DEFAULT 'none',
    delta_changed_json TEXT NOT NULL DEFAULT '[]',
    delta_recovery INTEGER NOT NULL DEFAULT 0,
    delta_regression INTEGER NOT NULL DEFAULT 0,
    trajectory_label TEXT NOT NULL DEFAULT '',
    perception_latency_ms REAL,
    generation_id INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_gui_states_time ON gui_states(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_gui_states_session ON gui_states(session_id, id);
CREATE TABLE IF NOT EXISTS gui_trajectory_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    created_at TEXT NOT NULL,
    event_timestamp TEXT NOT NULL,
    label TEXT NOT NULL,
    activity TEXT NOT NULL DEFAULT 'desktop',
    application TEXT NOT NULL DEFAULT 'unknown',
    topic TEXT,
    importance REAL NOT NULL DEFAULT 0.1
);
CREATE INDEX IF NOT EXISTS idx_gui_trajectory_time ON gui_trajectory_events(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_gui_trajectory_session ON gui_trajectory_events(session_id, id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_gui_trajectory_unique
    ON gui_trajectory_events(COALESCE(session_id, 0), event_timestamp, label);
"""


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    # Intervention tables and trace columns are an additive schema migration.
    SCHEMA_VERSION = 5

    def __init__(self, path: str | Path = "data/state.db") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # The controller owns one inference worker which performs SQLite
        # mutations. Allow that worker to use the engine-created connection;
        # callers still serialize access through the existing worker boundary.
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self._ensure_column("events", "session_id", "INTEGER")
        self._ensure_column("sessions", "status", "TEXT NOT NULL DEFAULT 'ACTIVE'")
        self._ensure_column("sessions", "summary_json", "TEXT")
        self._ensure_column("sessions", "owner_pid", "INTEGER")
        self._ensure_column("decision_traces", "reason_code", "TEXT NOT NULL DEFAULT 'POLICY_IGNORE'")
        self._ensure_column("decision_traces", "context_chars", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("decision_traces", "context_event_count", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("decision_traces", "context_watch_count", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("decision_traces", "preference_ids", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("decision_traces", "preference_effect", "TEXT")
        self._ensure_column("decision_traces", "similar_episode_ids", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("decision_traces", "intervention_episode_id", "INTEGER")
        self._ensure_column("feedback", "episode_id", "INTEGER")
        self._ensure_column("feedback", "feedback_type", "TEXT")
        self._ensure_column("feedback", "reaction", "TEXT")
        self._ensure_column("feedback", "outcome", "TEXT")
        self._ensure_column("feedback", "note", "TEXT")
        self._ensure_column("intervention_episodes", "learned_preference_id", "INTEGER")
        self._migrate_privacy_labels()
        self.connection.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.connection.execute(
            "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(self.SCHEMA_VERSION),),
        )
        self.connection.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {str(row[1]) for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _migrate_privacy_labels(self) -> None:
        """Normalize old semantic labels when an existing DB is reopened."""
        for row in self.connection.execute("SELECT id, failure_signature, summary FROM events WHERE failure_signature IS NOT NULL OR summary IS NOT NULL").fetchall():
            failure = sanitize_failure_signature(row["failure_signature"])
            summary = sanitize_semantic_label(row["summary"], 500)
            if failure != row["failure_signature"] or summary != row["summary"]:
                self.connection.execute(
                    "UPDATE events SET failure_signature = ?, summary = ? WHERE id = ?",
                    (failure, summary, int(row["id"])),
                )
        for row in self.connection.execute(
            "SELECT id, failure_signature, topic, summary, explicit_feedback FROM intervention_episodes"
        ).fetchall():
            failure = sanitize_failure_signature(row["failure_signature"])
            topic = sanitize_semantic_label(row["topic"], 160) or None
            summary = sanitize_semantic_label(row["summary"], 500)
            feedback = sanitize_semantic_label(row["explicit_feedback"], 500) or None
            if (failure, topic, summary, feedback) != (
                row["failure_signature"], row["topic"], row["summary"], row["explicit_feedback"]
            ):
                self.connection.execute(
                    """UPDATE intervention_episodes
                       SET failure_signature = ?, topic = ?, summary = ?, explicit_feedback = ?
                       WHERE id = ?""",
                    (failure, topic, summary, feedback, int(row["id"])),
                )
        for row in self.connection.execute(
            "SELECT id, scope_key, situation_type, activity, event_type, topic, failure_signature, content FROM intervention_preferences"
        ).fetchall():
            topic = sanitize_semantic_label(row["topic"], 160) or None
            failure = sanitize_failure_signature(row["failure_signature"])
            content = sanitize_semantic_label(row["content"], 500)
            scope_key = build_scope_key(
                situation_type=str(row["situation_type"] or "desktop"),
                activity=str(row["activity"] or "desktop"),
                event_type=str(row["event_type"] or "activity"),
                topic=topic,
                failure_signature=failure,
            )
            if (scope_key, topic, failure, content) != (
                row["scope_key"], row["topic"], row["failure_signature"], row["content"]
            ):
                self.connection.execute(
                    """UPDATE intervention_preferences
                       SET scope_key = ?, topic = ?, failure_signature = ?, content = ?
                       WHERE id = ?""",
                    (scope_key, topic, failure, content, int(row["id"])),
                )
        for row in self.connection.execute("SELECT id, summary FROM decision_traces WHERE summary IS NOT NULL").fetchall():
            summary = sanitize_semantic_label(row["summary"], 300)
            if summary != row["summary"]:
                self.connection.execute("UPDATE decision_traces SET summary = ? WHERE id = ?", (summary, int(row["id"])))
        for row in self.connection.execute("SELECT id, content FROM memories WHERE content IS NOT NULL").fetchall():
            content = sanitize_semantic_label(row["content"], 1000)
            if content != row["content"]:
                self.connection.execute("UPDATE memories SET content = ? WHERE id = ?", (content, int(row["id"])))
                self.connection.execute("DELETE FROM memories_fts WHERE rowid = ?", (int(row["id"]),))
                self.connection.execute(
                    "INSERT INTO memories_fts(rowid, content, tags) SELECT id, content, tags FROM memories WHERE id = ?",
                    (int(row["id"]),),
                )
        for row in self.connection.execute("SELECT id, note FROM feedback WHERE note IS NOT NULL").fetchall():
            note = sanitize_semantic_label(row["note"], 400) or None
            if note != row["note"]:
                self.connection.execute("UPDATE feedback SET note = ? WHERE id = ?", (note, int(row["id"])))

    def close(self) -> None:
        self.connection.close()

    def start_session(self, *, decision_retention_days: int = 30, session_retention_days: int = 90) -> int:
        now = _utc_iso()
        # Only close sessions whose owner process is gone. The normal Windows
        # InstanceLock prevents concurrent runtimes, while this PID check avoids
        # one store instance expiring another live instance's WATCH state.
        active_rows = self.connection.execute(
            "SELECT id, owner_pid FROM sessions WHERE status = 'ACTIVE' AND ended_at IS NULL"
        ).fetchall()
        stale_session_ids = [
            int(row["id"])
            for row in active_rows
            if row["owner_pid"] is None or not _process_alive(int(row["owner_pid"]))
        ]
        if stale_session_ids:
            placeholders = ", ".join("?" for _ in stale_session_ids)
            self.connection.execute(
                f"UPDATE sessions SET ended_at = COALESCE(ended_at, ?), status = 'ABORTED' WHERE id IN ({placeholders})",
                (now, *stale_session_ids),
            )
            self.connection.execute(
                f"UPDATE intervention_episodes SET updated_at = ?, status = 'EXPIRED', outcome = 'EXPIRED' WHERE status = 'ACTIVE' AND session_id IN ({placeholders})",
                (now, *stale_session_ids),
            )
        self.apply_retention(decision_retention_days, session_retention_days, commit=False)
        cursor = self.connection.execute(
            "INSERT INTO sessions(started_at, status, owner_pid) VALUES (?, 'ACTIVE', ?)",
            (now, os.getpid()),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def end_session(self, session_id: int | None, summary: dict[str, Any] | None = None) -> None:
        if session_id is None:
            return
        self.connection.execute(
            "UPDATE intervention_episodes SET updated_at = ?, status = 'EXPIRED', outcome = 'EXPIRED' WHERE session_id = ? AND status = 'ACTIVE'",
            (_utc_iso(), session_id),
        )
        self.connection.execute(
            "UPDATE sessions SET ended_at = COALESCE(ended_at, ?), status = 'COMPLETE', summary_json = COALESCE(?, summary_json) WHERE id = ?",
            (_utc_iso(), json.dumps(summary, sort_keys=True) if summary is not None else None, session_id),
        )
        self.connection.commit()

    def record_event(self, event: ExtractedEvent, source: str = "screenpipe", session_id: int | None = None) -> None:
        safe_failure_signature = sanitize_failure_signature(event.failure_signature)
        safe_summary = sanitize_semantic_label(event.summary, 500)
        self.connection.execute(
            """INSERT INTO events(timestamp, source, app, event_type, activity, summary,
               importance, novelty, confidence, failure_signature, session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event.timestamp.isoformat(), source, event.app, event.event_type, event.activity,
             safe_summary, event.importance, event.novelty, event.confidence, safe_failure_signature, session_id),
        )
        self.connection.commit()

    def count_failures(self, signature: str, since: datetime | None = None) -> int:
        safe_signature = sanitize_failure_signature(signature)
        if not safe_signature:
            return 0
        if since is None:
            row = self.connection.execute(
                "SELECT COUNT(*) AS count FROM events WHERE failure_signature = ?", (safe_signature,)
            ).fetchone()
        else:
            row = self.connection.execute(
                "SELECT COUNT(*) AS count FROM events WHERE failure_signature = ? AND timestamp > ?",
                (safe_signature, since.isoformat()),
            ).fetchone()
        return int(row["count"] if row else 0)

    def record_memory(self, content: str, source: str = "secretary", importance: float = 0.5, tags: str = "") -> None:
        safe_content = sanitize_semantic_label(content, 1000)
        safe_tags = sanitize_semantic_label(tags, 300)
        cursor = self.connection.execute(
            "INSERT INTO memories(created_at, source, content, importance, tags) VALUES (?, ?, ?, ?, ?)",
            (_utc_iso(), source, safe_content, importance, safe_tags),
        )
        row_id = cursor.lastrowid
        self.connection.execute("INSERT INTO memories_fts(rowid, content, tags) VALUES (?, ?, ?)", (row_id, safe_content, safe_tags))
        self.connection.commit()

    def search_memories(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT m.* FROM memories m JOIN memories_fts f ON f.rowid = m.id WHERE memories_fts MATCH ? ORDER BY m.importance DESC, m.created_at DESC LIMIT ?",
            (query, max(1, min(20, limit))),
        ).fetchall()
        return [dict(row) for row in rows]

    def record_intervention_episode(
        self,
        *,
        session_id: int | None,
        event_timestamp: datetime,
        situation_type: str,
        activity: str,
        event_type: str,
        candidate_action: str,
        final_action: str,
        reason_codes: Sequence[str] = (),
        model_confidence: float = 0.0,
        importance: float = 0.0,
        interrupt_score: float = 0.0,
        was_notified: bool = False,
        status: InterventionStatus | str = InterventionStatus.RECORDED,
        user_reaction: UserReaction | str = UserReaction.UNKNOWN,
        outcome: InterventionOutcome | str = InterventionOutcome.UNKNOWN,
        topic: str | None = None,
        failure_signature: str | None = None,
        summary: str = "",
        watch_id: str | None = None,
        watch_context: Mapping[str, object] | None = None,
        notification_id: int | None = None,
        explicit_feedback: str | None = None,
        preference_ids: Sequence[int] = (),
        learned_preference_id: int | None = None,
    ) -> int:
        """Persist a bounded opportunity without storing raw desktop content."""
        now = _utc_iso()
        status_value = _enum_text(status, InterventionStatus.RECORDED)
        reaction_value = _enum_text(user_reaction, UserReaction.UNKNOWN)
        outcome_value = _enum_text(outcome, InterventionOutcome.UNKNOWN)
        safe_reason_codes = _safe_codes(reason_codes)
        safe_preference_ids = _safe_ids(preference_ids)
        safe_failure_signature = sanitize_failure_signature(failure_signature)
        cursor = self.connection.execute(
            """INSERT INTO intervention_episodes(
                session_id, created_at, updated_at, event_timestamp, situation_type,
                activity, event_type, topic, failure_signature, summary, watch_id,
                watch_context, candidate_action, final_action, reason_codes,
                model_confidence, importance, interrupt_score, was_notified,
                notification_id, status, user_reaction, outcome, explicit_feedback,
                preference_ids, learned_preference_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                now,
                now,
                _timestamp_iso(event_timestamp),
                _bounded(situation_type, 80, "desktop"),
                _bounded(activity, 80, "desktop"),
                _bounded(event_type, 80, "activity"),
                sanitize_semantic_label(topic, 160) or None,
                _bounded(safe_failure_signature, 180, "") or None,
                sanitize_semantic_label(summary, 500),
                _bounded(watch_id, 180, "") or None,
                _bounded_json(watch_context or {}, 2000),
                _bounded(candidate_action, 30, "IGNORE"),
                _bounded(final_action, 30, "IGNORE"),
                json.dumps(safe_reason_codes, ensure_ascii=True),
                _probability(model_confidence),
                _probability(importance),
                _probability(interrupt_score),
                1 if was_notified else 0,
                notification_id,
                status_value,
                reaction_value,
                outcome_value,
                sanitize_semantic_label(explicit_feedback, 500) or None,
                json.dumps(safe_preference_ids, ensure_ascii=True),
                learned_preference_id,
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def get_intervention_episode(self, episode_id: int) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM intervention_episodes WHERE id = ?", (int(episode_id),)
        ).fetchone()
        return _decode_intervention_episode(dict(row)) if row else None

    def recent_intervention_episodes(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM intervention_episodes ORDER BY id DESC LIMIT ?",
            (max(1, min(100, int(limit))),),
        ).fetchall()
        return [_decode_intervention_episode(dict(row)) for row in rows]

    def active_intervention_preferences(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM intervention_preferences WHERE status = 'ACTIVE' ORDER BY updated_at DESC, id DESC LIMIT ?",
            (max(1, min(200, int(limit))),),
        ).fetchall()
        return [dict(row) for row in rows]

    def record_intervention_feedback(
        self,
        episode_id: int,
        value: str,
        *,
        reaction: UserReaction | str | None = None,
        outcome: InterventionOutcome | str | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Record explicit feedback and, when applicable, update preference memory."""
        episode = self.get_intervention_episode(episode_id)
        if episode is None:
            raise ValueError(f"intervention episode not found: {episode_id}")
        instruction = normalize_feedback(value)
        selected_reaction = parse_reaction(reaction) if reaction is not None else instruction.reaction
        current_reaction = parse_reaction(str(episode.get("user_reaction") or "UNKNOWN"))
        if selected_reaction == UserReaction.UNKNOWN:
            selected_reaction = current_reaction
        selected_outcome = parse_outcome(outcome)
        current_outcome = parse_outcome(str(episode.get("outcome") or "UNKNOWN"))
        if selected_outcome == InterventionOutcome.UNKNOWN:
            selected_outcome = current_outcome
        safe_note = sanitize_semantic_label(note, 400)
        explicit_feedback = instruction.value if instruction.value not in {"OBSERVED", "REACTION"} else None
        if safe_note:
            explicit_feedback = sanitize_semantic_label(f"{explicit_feedback}: {safe_note}" if explicit_feedback else safe_note, 500)

        now = _utc_iso()
        cursor = self.connection.execute(
            """INSERT INTO feedback(
                created_at, notification_id, value, episode_id, feedback_type,
                reaction, outcome, note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now,
                episode.get("notification_id"),
                instruction.value,
                int(episode_id),
                "EXPLICIT" if instruction.preference is not None or instruction.value == "FORGET" else "OBSERVED",
                selected_reaction.value,
                selected_outcome.value,
                safe_note or None,
            ),
        )
        self.connection.execute(
            """UPDATE intervention_episodes
               SET updated_at = ?, user_reaction = ?, outcome = ?,
                   status = CASE WHEN ? = 'RESOLVED' AND status = 'ACTIVE' THEN 'RESOLVED' ELSE status END,
                   explicit_feedback = COALESCE(?, explicit_feedback)
               WHERE id = ?""",
            (now, selected_reaction.value, selected_outcome.value, selected_outcome.value, explicit_feedback, int(episode_id)),
        )

        preference_id: int | None = None
        if instruction.preference is not None:
            preference_id = self.upsert_intervention_preference(
                episode_id=int(episode_id),
                preference=instruction.preference,
                source=PreferenceSource.EXPLICIT_USER,
                confidence=1.0,
                commit=False,
            )
            self.connection.execute(
                "UPDATE intervention_episodes SET learned_preference_id = ? WHERE id = ?",
                (preference_id, int(episode_id)),
            )
        elif instruction.value == "FORGET":
            self._disable_preferences_for_episode(int(episode_id), commit=False)
        self.connection.commit()
        return {
            "feedback_id": int(cursor.lastrowid),
            "episode_id": int(episode_id),
            "value": instruction.value,
            "reaction": selected_reaction.value,
            "outcome": selected_outcome.value,
            "preference_id": preference_id,
        }

    def record_feedback(
        self,
        episode_id: int,
        value: str,
        *,
        reaction: UserReaction | str | None = None,
        outcome: InterventionOutcome | str | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Compatibility-friendly internal API for feedback clients."""
        return self.record_intervention_feedback(
            episode_id,
            value,
            reaction=reaction,
            outcome=outcome,
            note=note,
        )

    def upsert_intervention_preference(
        self,
        *,
        episode_id: int,
        preference: PreferenceKind | str,
        source: PreferenceSource | str = PreferenceSource.EXPLICIT_USER,
        confidence: float = 1.0,
        content: str | None = None,
        commit: bool = True,
    ) -> int:
        episode = self.get_intervention_episode(episode_id)
        if episode is None:
            raise ValueError(f"intervention episode not found: {episode_id}")
        preference_value = _enum_text(preference, PreferenceKind.MORE_PROACTIVE)
        source_value = _enum_text(source, PreferenceSource.EXPLICIT_USER)
        try:
            PreferenceKind(preference_value)
        except ValueError as exc:
            raise ValueError(f"unsupported intervention preference: {preference_value}") from exc
        try:
            PreferenceSource(source_value)
        except ValueError as exc:
            raise ValueError(f"unsupported preference source: {source_value}") from exc
        scope_key = build_scope_key(
            situation_type=str(episode.get("situation_type") or "desktop"),
            activity=str(episode.get("activity") or "desktop"),
            event_type=str(episode.get("event_type") or "activity"),
            topic=sanitize_semantic_label(episode.get("topic"), 160) or None,
            failure_signature=sanitize_failure_signature(str(episode.get("failure_signature"))) if episode.get("failure_signature") else None,
        )
        preference_content = content or _default_preference_content(preference_value, episode)
        now = _utc_iso()
        existing = self.connection.execute(
            "SELECT id, evidence_count FROM intervention_preferences WHERE scope_key = ? AND preference = ? AND status = 'ACTIVE' ORDER BY id DESC LIMIT 1",
            (scope_key, preference_value),
        ).fetchone()
        if existing is not None:
            self.connection.execute(
                """UPDATE intervention_preferences
                   SET updated_at = ?, source = ?, confidence = MAX(confidence, ?),
                       evidence_count = evidence_count + 1, content = ?, last_episode_id = ?
                   WHERE id = ?""",
                (now, source_value, _probability(confidence), sanitize_semantic_label(preference_content, 500), int(episode_id), int(existing["id"])),
            )
            preference_id = int(existing["id"])
        else:
            old_rows = self.connection.execute(
                "SELECT id FROM intervention_preferences WHERE scope_key = ? AND status = 'ACTIVE' ORDER BY id DESC",
                (scope_key,),
            ).fetchall()
            supersedes_id = int(old_rows[0]["id"]) if old_rows else None
            if old_rows:
                self.connection.execute(
                    "UPDATE intervention_preferences SET status = 'SUPERSEDED', updated_at = ? WHERE scope_key = ? AND status = 'ACTIVE'",
                    (now, scope_key),
                )
            cursor = self.connection.execute(
                """INSERT INTO intervention_preferences(
                    created_at, updated_at, status, scope_key, situation_type,
                    activity, event_type, topic, failure_signature, preference,
                    content, source, confidence, evidence_count, last_episode_id,
                    supersedes_id)
                    VALUES (?, ?, 'ACTIVE', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                (
                    now,
                    now,
                    scope_key,
                    _bounded(episode.get("situation_type"), 80, "desktop"),
                    _bounded(episode.get("activity"), 80, "desktop"),
                    _bounded(episode.get("event_type"), 80, "activity"),
                    sanitize_semantic_label(episode.get("topic"), 160) or None,
                    sanitize_failure_signature(str(episode.get("failure_signature"))) if episode.get("failure_signature") else None,
                    preference_value,
                    sanitize_semantic_label(preference_content, 500),
                    source_value,
                    _probability(confidence),
                    int(episode_id),
                    supersedes_id,
                ),
            )
            preference_id = int(cursor.lastrowid)
        if commit:
            self.connection.commit()
        return preference_id

    def disable_intervention_preferences(
        self,
        *,
        episode_id: int | None = None,
        preference_id: int | None = None,
    ) -> int:
        if episode_id is None and preference_id is None:
            return 0
        if preference_id is not None:
            cursor = self.connection.execute(
                "UPDATE intervention_preferences SET status = 'DISABLED', content = '', updated_at = ? WHERE id = ? AND status = 'ACTIVE'",
                (_utc_iso(), int(preference_id)),
            )
        else:
            episode = self.get_intervention_episode(int(episode_id))
            if episode is None:
                return 0
            scope_key = build_scope_key(
                situation_type=str(episode.get("situation_type") or "desktop"),
                activity=str(episode.get("activity") or "desktop"),
                event_type=str(episode.get("event_type") or "activity"),
                topic=sanitize_semantic_label(episode.get("topic"), 160) or None,
                failure_signature=sanitize_failure_signature(str(episode.get("failure_signature"))) if episode.get("failure_signature") else None,
            )
            cursor = self.connection.execute(
                "UPDATE intervention_preferences SET status = 'DISABLED', content = '', updated_at = ? WHERE scope_key = ? AND status = 'ACTIVE'",
                (_utc_iso(), scope_key),
            )
        self.connection.commit()
        return int(cursor.rowcount)

    def update_intervention_outcome(self, watch_id: str, status: str, outcome: InterventionOutcome | str) -> int:
        if not watch_id:
            return 0
        cursor = self.connection.execute(
            """UPDATE intervention_episodes
               SET updated_at = ?, status = ?, outcome = ?
               WHERE watch_id = ? AND status = 'ACTIVE'""",
            (_utc_iso(), _bounded(status, 30, "ACTIVE"), _enum_text(outcome, InterventionOutcome.UNKNOWN), _bounded(watch_id, 180, "")),
        )
        self.connection.commit()
        return int(cursor.rowcount)

    def _disable_preferences_for_episode(self, episode_id: int, *, commit: bool = True) -> int:
        episode = self.get_intervention_episode(episode_id)
        if episode is None:
            return 0
        scope_key = build_scope_key(
            situation_type=str(episode.get("situation_type") or "desktop"),
            activity=str(episode.get("activity") or "desktop"),
            event_type=str(episode.get("event_type") or "activity"),
            topic=sanitize_semantic_label(episode.get("topic"), 160) or None,
            failure_signature=sanitize_failure_signature(str(episode.get("failure_signature"))) if episode.get("failure_signature") else None,
        )
        cursor = self.connection.execute(
            "UPDATE intervention_preferences SET status = 'DISABLED', content = '', updated_at = ? WHERE scope_key = ? AND status = 'ACTIVE'",
            (_utc_iso(), scope_key),
        )
        if commit:
            self.connection.commit()
        return int(cursor.rowcount)

    def record_hypothesis(self, hypothesis: str, evidence: int, status: str, expires_at: str | None) -> None:
        self.connection.execute(
            "INSERT INTO hypotheses(created_at, status, hypothesis, evidence, expires_at) VALUES (?, ?, ?, ?, ?)",
            (_utc_iso(), status, sanitize_semantic_label(hypothesis, 500), evidence, expires_at),
        )
        self.connection.commit()

    def record_decision(self, action: str, reason: str, evidence: int = 0) -> None:
        self.connection.execute(
            "INSERT INTO assistant_decisions(created_at, action, reason, evidence) VALUES (?, ?, ?, ?)",
            (_utc_iso(), action, sanitize_semantic_label(reason, 500), evidence),
        )
        self.connection.commit()

    def record_decision_trace(
        self,
        *,
        session_id: int | None,
        event_timestamp: datetime,
        foreground_app: str,
        event_type: str,
        candidate_action: str,
        candidate_confidence: float,
        candidate_importance: float,
        interrupt_score: float,
        deterministic_evidence: int,
        watch_id: str | None,
        watch_evidence: int,
        policy_action: str,
        final_action: str,
        suppression_reason: str | None,
        inference_latency_ms: float | None,
        inference_mode: str | None,
        reason: str,
        summary: str,
        cloud_escalation_candidate: bool = False,
        reason_code: str = "POLICY_IGNORE",
        context_chars: int = 0,
        context_event_count: int = 0,
        context_watch_count: int = 0,
        preference_ids: Sequence[int] = (),
        preference_effect: str | None = None,
        similar_episode_ids: Sequence[int] = (),
        intervention_episode_id: int | None = None,
    ) -> None:
        """Persist only bounded, non-content decision metadata."""
        self.connection.execute(
            """INSERT INTO decision_traces(
                session_id, created_at, event_timestamp, foreground_app, event_type,
                candidate_action, candidate_confidence, candidate_importance,
                interrupt_score, deterministic_evidence, watch_id, watch_evidence,
                policy_action, final_action, suppression_reason, inference_latency_ms,
                inference_mode, reason, summary, cloud_escalation_candidate, reason_code,
                context_chars, context_event_count, context_watch_count, preference_ids,
                preference_effect, similar_episode_ids, intervention_episode_id)
                VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?
                )""",
            (
                session_id,
                _utc_iso(),
                event_timestamp.isoformat(),
                foreground_app[:160],
                event_type[:80],
                candidate_action[:30],
                max(0.0, min(1.0, float(candidate_confidence))),
                max(0.0, min(1.0, float(candidate_importance))),
                max(0.0, min(1.0, float(interrupt_score))),
                max(0, int(deterministic_evidence)),
                watch_id[:180] if watch_id else None,
                max(0, int(watch_evidence)),
                policy_action[:30],
                final_action[:30],
                suppression_reason[:120] if suppression_reason else None,
                max(0.0, float(inference_latency_ms)) if inference_latency_ms is not None else None,
                inference_mode[:20] if inference_mode else None,
                sanitize_semantic_label(reason, 500),
                sanitize_semantic_label(summary, 300),
                1 if cloud_escalation_candidate else 0,
                reason_code[:80],
                max(0, int(context_chars)),
                max(0, int(context_event_count)),
                max(0, int(context_watch_count)),
                json.dumps(_safe_ids(preference_ids), ensure_ascii=True),
                _bounded(preference_effect, 160, "") or None,
                json.dumps(_safe_ids(similar_episode_ids), ensure_ascii=True),
                intervention_episode_id,
            ),
        )
        self.connection.commit()

    def recent_decision_traces(self, limit: int = 20, *, action: str | None = None, suppressed: bool = False) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if action:
            where.append("final_action = ?")
            params.append(action.upper())
        if suppressed:
            where.append("(suppression_reason IS NOT NULL OR final_action = 'WOULD_NOTIFY')")
        query = "SELECT * FROM decision_traces"
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, min(100, limit)))
        rows = self.connection.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def latest_session_report(self) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM sessions ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        session_id = int(row["id"])
        traces = self.connection.execute(
            "SELECT * FROM decision_traces WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
        started_at = _parse_utc(row["started_at"])
        ended_at = _parse_utc(row["ended_at"]) if row["ended_at"] else datetime.now(timezone.utc)
        latencies = sorted(float(item["inference_latency_ms"]) for item in traces if item["inference_latency_ms"] is not None)
        p95_index = max(0, math.ceil(len(latencies) * 0.95) - 1) if latencies else None
        candidate_counts = Counter(str(item["candidate_action"]) for item in traces)
        final_counts = Counter(str(item["final_action"]) for item in traces)
        modes = Counter(str(item["inference_mode"]) for item in traces if item["inference_mode"])
        suppressed = Counter(str(item["suppression_reason"]) for item in traces if item["suppression_reason"])
        event_count = self.connection.execute("SELECT COUNT(*) FROM events WHERE session_id = ?", (session_id,)).fetchone()[0]
        intervention_count = self.connection.execute(
            "SELECT COUNT(*) FROM intervention_episodes WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        personalized_count = sum(1 for item in traces if str(item["preference_ids"] or "[]") != "[]")
        summary: dict[str, Any] = {}
        if row["summary_json"]:
            try:
                parsed = json.loads(str(row["summary_json"]))
                if isinstance(parsed, dict):
                    summary = parsed
            except (TypeError, ValueError, json.JSONDecodeError):
                summary = {}
        saved_counters = summary.get("counters") if isinstance(summary.get("counters"), dict) else {}
        report = {
            "session_id": session_id,
            "status": str(row["status"] or ("COMPLETE" if row["ended_at"] else "ACTIVE")),
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": max(0.0, (ended_at - started_at).total_seconds()),
            "screenpipe_events": int(saved_counters.get("raw_screenpipe_items", event_count)) if isinstance(saved_counters, dict) else int(event_count),
            "semantic_inference_requests": len(traces),
            "candidate_actions": dict(candidate_counts),
            "final_actions": dict(final_counts),
            "inference_modes": dict(modes),
            "suppression_reasons": dict(suppressed),
            "suppressed_model_notify": sum(1 for item in traces if item["candidate_action"] == "NOTIFY" and item["policy_action"] != "NOTIFY"),
            "cloud_escalation_candidates": sum(1 for item in traces if item["cloud_escalation_candidate"]),
            "intervention_episodes": int(intervention_count),
            "personalized_decisions": personalized_count,
            "average_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "p95_latency_ms": latencies[p95_index] if p95_index is not None else None,
            "latency": {
                "overall": _latency_stats(latencies),
                "text": _latency_stats([float(item["inference_latency_ms"]) for item in traces if item["inference_latency_ms"] is not None and item["inference_mode"] == "text"]),
                "vision": _latency_stats([float(item["inference_latency_ms"]) for item in traces if item["inference_latency_ms"] is not None and item["inference_mode"] == "vision"]),
            },
        }
        if isinstance(saved_counters, dict):
            report["counters"] = {str(key): int(value) for key, value in saved_counters.items() if isinstance(value, (int, float))}
        else:
            report["counters"] = {}
        return report

    def apply_retention(self, decision_days: int = 30, session_days: int = 90, *, commit: bool = True) -> None:
        """Remove bounded operational traces while retaining semantic memories."""
        decision_cutoff = datetime.now(timezone.utc).timestamp() - max(1, decision_days) * 86400
        session_cutoff = datetime.now(timezone.utc).timestamp() - max(1, session_days) * 86400
        decision_iso = datetime.fromtimestamp(decision_cutoff, timezone.utc).isoformat()
        session_iso = datetime.fromtimestamp(session_cutoff, timezone.utc).isoformat()
        self.connection.execute("DELETE FROM decision_traces WHERE created_at < ?", (decision_iso,))
        # Keep explicit feedback and learned preferences, but discard old
        # unlabelled opportunities together with other operational traces.
        self.connection.execute(
            "DELETE FROM intervention_episodes WHERE created_at < ? AND user_reaction = 'UNKNOWN' AND explicit_feedback IS NULL",
            (session_iso,),
        )
        self.connection.execute("DELETE FROM sessions WHERE ended_at IS NOT NULL AND ended_at < ?", (session_iso,))
        if commit:
            self.connection.commit()

    def record_notification(self, title: str, body: str, action: str) -> int:
        cursor = self.connection.execute(
            "INSERT INTO notifications(created_at, title, body, action) VALUES (?, ?, ?, ?)",
            (_utc_iso(), sanitize_semantic_label(title, 200), sanitize_semantic_label(body, 1000), action),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def trim(self, max_events: int = 5000) -> None:
        self.connection.execute(
            "DELETE FROM events WHERE id NOT IN (SELECT id FROM events ORDER BY timestamp DESC LIMIT ?)",
            (max(100, max_events),),
        )
        self.connection.commit()

    def record_gui_state(
        self,
        *,
        session_id: int | None,
        event_timestamp: datetime,
        application: str,
        window: str,
        activity: str,
        topic: str | None,
        progress: str,
        task_hint: str | None,
        errors: Sequence[str],
        confidence: float,
        perception_mode: str,
        keyframe_reason: str,
        changed_fields: Sequence[str],
        recovery: bool,
        regression: bool,
        trajectory_label: str,
        perception_latency_ms: float | None = None,
        generation_id: int = 0,
    ) -> int:
        cursor = self.connection.execute(
            """INSERT INTO gui_states(
                session_id, created_at, event_timestamp, application, window,
                activity, topic, progress, task_hint, errors_json, confidence,
                perception_mode, keyframe_reason, delta_changed_json,
                delta_recovery, delta_regression, trajectory_label,
                perception_latency_ms, generation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                _utc_iso(),
                _timestamp_iso(event_timestamp),
                _bounded(application, 160, "unknown"),
                _bounded(window, 200, ""),
                _bounded(activity, 80, "desktop"),
                sanitize_semantic_label(topic, 160) or None,
                _bounded(progress, 40, "unknown"),
                sanitize_semantic_label(task_hint, 240) or None,
                json.dumps([sanitize_semantic_label(item, 200) for item in errors[:6]], ensure_ascii=True),
                _probability(confidence),
                _bounded(perception_mode, 40, "structured"),
                _bounded(keyframe_reason, 120, "none"),
                json.dumps([_bounded(item, 60, "") for item in changed_fields[:10]], ensure_ascii=True),
                1 if recovery else 0,
                1 if regression else 0,
                sanitize_semantic_label(trajectory_label, 200),
                max(0.0, float(perception_latency_ms)) if perception_latency_ms is not None and math.isfinite(float(perception_latency_ms)) else None,
                max(0, int(generation_id)),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def record_gui_trajectory_event(
        self,
        *,
        session_id: int | None,
        event_timestamp: datetime,
        label: str,
        activity: str,
        application: str,
        topic: str | None,
        importance: float,
    ) -> int:
        cursor = self.connection.execute(
            """INSERT OR IGNORE INTO gui_trajectory_events(
                session_id, created_at, event_timestamp, label, activity,
                application, topic, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                _utc_iso(),
                _timestamp_iso(event_timestamp),
                sanitize_semantic_label(label, 240),
                _bounded(activity, 80, "desktop"),
                _bounded(application, 160, "unknown"),
                sanitize_semantic_label(topic, 160) or None,
                _probability(importance),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid) if cursor.rowcount else 0

    def latest_gui_state(self) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM gui_states ORDER BY id DESC LIMIT 1").fetchone()
        return _decode_gui_state(dict(row)) if row else None

    def recent_gui_states(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM gui_states ORDER BY event_timestamp DESC, id DESC LIMIT ?",
            (max(1, min(400, int(limit))),),
        ).fetchall()
        return [_decode_gui_state(dict(row)) for row in rows]

    def recent_gui_state_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return GUI states in ascending order for diagnostics."""
        rows = self.connection.execute(
            "SELECT * FROM gui_states ORDER BY event_timestamp ASC, id ASC LIMIT ?",
            (max(1, min(400, int(limit))),),
        ).fetchall()
        return [_decode_gui_state(dict(row)) for row in rows]

    def recent_trajectory_events(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM gui_trajectory_events ORDER BY event_timestamp ASC, id ASC LIMIT ?",
            (max(1, min(400, int(limit))),),
        ).fetchall()
        return [_decode_trajectory_event(dict(row)) for row in rows]

    def gui_perception_stats(self) -> dict[str, Any]:
        counts = self.connection.execute(
            "SELECT COUNT(*) AS n, "
            "SUM(CASE WHEN perception_mode = 'vision' THEN 1 ELSE 0 END) AS vision, "
            "SUM(CASE WHEN perception_mode = 'structured' THEN 1 ELSE 0 END) AS structured, "
            "SUM(delta_recovery) AS recoveries, SUM(delta_regression) AS regressions "
            "FROM gui_states"
        ).fetchone()
        latency_rows = [
            float(row[0])
            for row in self.connection.execute(
                "SELECT perception_latency_ms FROM gui_states WHERE perception_latency_ms IS NOT NULL"
            ).fetchall()
        ]
        return {
            "gui_states": int(counts["n"] or 0),
            "vision_perceptions": int(counts["vision"] or 0),
            "structured_updates": int(counts["structured"] or 0),
            "recoveries": int(counts["recoveries"] or 0),
            "regressions": int(counts["regressions"] or 0),
            "latency": _latency_stats(latency_rows),
        }


def _timestamp_iso(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return _bounded(value, 80, _utc_iso())


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError, PermissionError, ValueError):
        return False
    return True


def _bounded(value: object, limit: int, default: str) -> str:
    if value is None:
        return default
    text = " ".join(str(value).split()).strip()
    return text[:limit] if text else default


def _probability(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return min(1.0, max(0.0, number)) if math.isfinite(number) else 0.0


def _enum_text(value: object, default: object) -> str:
    raw = getattr(value, "value", value)
    fallback = getattr(default, "value", default)
    text = _bounded(raw, 80, _bounded(fallback, 80, "UNKNOWN"))
    return text.upper()


def _safe_codes(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        code = _bounded(value, 80, "")
        if code and code not in seen:
            result.append(code)
            seen.add(code)
        if len(result) >= 8:
            break
    return result


def _safe_ids(values: Sequence[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if item > 0 and item not in seen:
            result.append(item)
            seen.add(item)
        if len(result) >= 12:
            break
    return result


def _bounded_json(value: Mapping[str, object], limit: int) -> str:
    safe: dict[str, object] = {}
    for key, item in value.items():
        safe_key = _bounded(key, 80, "")
        if not safe_key:
            continue
        if item is None or isinstance(item, (bool, int, float, str)):
            safe[safe_key] = sanitize_semantic_label(item, 300) if isinstance(item, str) else item
        else:
            safe[safe_key] = sanitize_semantic_label(str(item), 300)
    encoded = json.dumps(safe, ensure_ascii=True, sort_keys=True)
    return encoded if len(encoded) <= limit else '{"truncated":true}'


def _decode_json_list(value: object) -> list[object]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _decode_intervention_episode(row: dict[str, Any]) -> dict[str, Any]:
    row["reason_codes"] = _decode_json_list(row.get("reason_codes", "[]"))
    row["preference_ids"] = _decode_json_list(row.get("preference_ids", "[]"))
    try:
        watch_context = json.loads(str(row.get("watch_context", "{}")))
    except (TypeError, ValueError, json.JSONDecodeError):
        watch_context = {}
    row["watch_context"] = watch_context if isinstance(watch_context, dict) else {}
    row["was_notified"] = bool(row.get("was_notified", 0))
    return row


def _decode_gui_state(row: dict[str, Any]) -> dict[str, Any]:
    row["errors"] = _decode_json_list(row.get("errors_json", "[]"))
    row["changed_fields"] = _decode_json_list(row.get("delta_changed_json", "[]"))
    row["delta_recovery"] = bool(row.get("delta_recovery", 0))
    row["delta_regression"] = bool(row.get("delta_regression", 0))
    row["confidence"] = float(row.get("confidence", 0.0))
    return row


def _decode_trajectory_event(row: dict[str, Any]) -> dict[str, Any]:
    row["importance"] = float(row.get("importance", 0.1))
    return row


def _default_preference_content(preference: str, episode: Mapping[str, object]) -> str:
    label = _bounded(episode.get("failure_signature"), 160, "")
    if not label:
        label = _bounded(episode.get("situation_type"), 80, "this situation")
    messages = {
        PreferenceKind.AVOID_ISOLATED.value: f"Avoid interrupting for isolated {label} signals.",
        PreferenceKind.EARLIER_WARNING.value: f"Warn earlier when the {label} pattern starts repeating.",
        PreferenceKind.MORE_PROACTIVE.value: f"Be more proactive when this {label} situation appears.",
        PreferenceKind.TIMING_SENSITIVE.value: f"Wait for stronger evidence and better timing before interrupting for {label}.",
    }
    return messages.get(preference, f"Adjust intervention timing for {label}.")


def _parse_utc(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _latency_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p90_ms": None, "p95_ms": None, "max_ms": None}
    ordered = sorted(values)

    def percentile(percent: float) -> float:
        index = max(0, math.ceil(len(ordered) * percent) - 1)
        return ordered[index]

    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "count": len(ordered),
        "mean_ms": sum(ordered) / len(ordered),
        "median_ms": median,
        "p90_ms": percentile(0.90),
        "p95_ms": percentile(0.95),
        "max_ms": ordered[-1],
    }
