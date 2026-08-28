from __future__ import annotations

import sqlite3
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    ended_at TEXT
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
    cloud_escalation_candidate INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_decision_traces_session ON decision_traces(session_id, id);
CREATE INDEX IF NOT EXISTS idx_decision_traces_time ON decision_traces(event_timestamp);
"""


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
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
        self.connection.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {str(row[1]) for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def close(self) -> None:
        self.connection.close()

    def start_session(self) -> int:
        cursor = self.connection.execute("INSERT INTO sessions(started_at) VALUES (?)", (_utc_iso(),))
        self.connection.commit()
        return int(cursor.lastrowid)

    def end_session(self, session_id: int | None) -> None:
        if session_id is None:
            return
        self.connection.execute(
            "UPDATE sessions SET ended_at = COALESCE(ended_at, ?) WHERE id = ?",
            (_utc_iso(), session_id),
        )
        self.connection.commit()

    def record_event(self, event: ExtractedEvent, source: str = "screenpipe", session_id: int | None = None) -> None:
        self.connection.execute(
            """INSERT INTO events(timestamp, source, app, event_type, activity, summary,
               importance, novelty, confidence, failure_signature, session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event.timestamp.isoformat(), source, event.app, event.event_type, event.activity,
             event.summary, event.importance, event.novelty, event.confidence, event.failure_signature, session_id),
        )
        self.connection.commit()

    def count_failures(self, signature: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM events WHERE failure_signature = ?", (signature,)
        ).fetchone()
        return int(row["count"] if row else 0)

    def record_memory(self, content: str, source: str = "secretary", importance: float = 0.5, tags: str = "") -> None:
        cursor = self.connection.execute(
            "INSERT INTO memories(created_at, source, content, importance, tags) VALUES (?, ?, ?, ?, ?)",
            (_utc_iso(), source, content[:1000], importance, tags[:300]),
        )
        row_id = cursor.lastrowid
        self.connection.execute("INSERT INTO memories_fts(rowid, content, tags) VALUES (?, ?, ?)", (row_id, content[:1000], tags[:300]))
        self.connection.commit()

    def search_memories(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT m.* FROM memories m JOIN memories_fts f ON f.rowid = m.id WHERE memories_fts MATCH ? ORDER BY m.importance DESC, m.created_at DESC LIMIT ?",
            (query, max(1, min(20, limit))),
        ).fetchall()
        return [dict(row) for row in rows]

    def record_hypothesis(self, hypothesis: str, evidence: int, status: str, expires_at: str | None) -> None:
        self.connection.execute(
            "INSERT INTO hypotheses(created_at, status, hypothesis, evidence, expires_at) VALUES (?, ?, ?, ?, ?)",
            (_utc_iso(), status, hypothesis[:500], evidence, expires_at),
        )
        self.connection.commit()

    def record_decision(self, action: str, reason: str, evidence: int = 0) -> None:
        self.connection.execute(
            "INSERT INTO assistant_decisions(created_at, action, reason, evidence) VALUES (?, ?, ?, ?)",
            (_utc_iso(), action, reason[:500], evidence),
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
    ) -> None:
        """Persist only bounded, non-content decision metadata."""
        self.connection.execute(
            """INSERT INTO decision_traces(
                session_id, created_at, event_timestamp, foreground_app, event_type,
                candidate_action, candidate_confidence, candidate_importance,
                interrupt_score, deterministic_evidence, watch_id, watch_evidence,
                policy_action, final_action, suppression_reason, inference_latency_ms,
                inference_mode, reason, summary, cloud_escalation_candidate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                reason[:500],
                summary[:300],
                1 if cloud_escalation_candidate else 0,
            ),
        )
        self.connection.commit()

    def recent_decision_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM decision_traces ORDER BY id DESC LIMIT ?", (max(1, min(100, limit)),)
        ).fetchall()
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
        return {
            "session_id": session_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": max(0.0, (ended_at - started_at).total_seconds()),
            "screenpipe_events": int(event_count),
            "semantic_inference_requests": len(traces),
            "candidate_actions": dict(candidate_counts),
            "final_actions": dict(final_counts),
            "inference_modes": dict(modes),
            "suppression_reasons": dict(suppressed),
            "suppressed_model_notify": sum(1 for item in traces if item["candidate_action"] == "NOTIFY" and item["policy_action"] != "NOTIFY"),
            "cloud_escalation_candidates": sum(1 for item in traces if item["cloud_escalation_candidate"]),
            "average_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "p95_latency_ms": latencies[p95_index] if p95_index is not None else None,
        }

    def record_notification(self, title: str, body: str, action: str) -> None:
        self.connection.execute(
            "INSERT INTO notifications(created_at, title, body, action) VALUES (?, ?, ?, ?)",
            (_utc_iso(), title[:200], body[:1000], action),
        )
        self.connection.commit()

    def trim(self, max_events: int = 5000) -> None:
        self.connection.execute(
            "DELETE FROM events WHERE id NOT IN (SELECT id FROM events ORDER BY timestamp DESC LIMIT ?)",
            (max(100, max_events),),
        )
        self.connection.commit()


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
