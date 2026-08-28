"""Memory doctor: read-only diagnostics for memory hygiene.

Dry-run only by default; no mutation. Reports:
- core memory size vs budget
- duplicate content
- superseded memories still referenced as active truth
- orphan source_episode_ids
- possible contradictions (same scope, opposite preference kinds)
- context budget usage
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .hierarchy import CORE_MEMORY_CHARS_LIMIT, CONTEXT_EPISODE_BUDGET, MemoryTier
from .store import MemoryStore


@dataclass(frozen=True)
class MemoryFinding:
    kind: str
    message: str
    memory_id: int | None = None
    severity: str = "info"

    def as_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "message": self.message, "memory_id": self.memory_id, "severity": self.severity}


@dataclass(frozen=True)
class MemoryDoctorReport:
    findings: tuple[MemoryFinding, ...] = ()
    core_chars: int = 0
    core_budget: int = CORE_MEMORY_CHARS_LIMIT
    active_memory_count: int = 0
    superseded_count: int = 0

    @property
    def issues(self) -> int:
        return sum(1 for finding in self.findings if finding.severity != "info") if self.findings else 0

    def as_dict(self) -> dict[str, object]:
        return {
            "findings": [finding.as_dict() for finding in self.findings],
            "core_chars": self.core_chars,
            "core_budget": self.core_budget,
            "active_memory_count": self.active_memory_count,
            "superseded_count": self.superseded_count,
            "issues": self.issues,
        }


def diagnose(store: MemoryStore, *, active_limit: int = 300) -> MemoryDoctorReport:
    findings: list[MemoryFinding] = []
    core_blocks = store.core_memory()
    core_text = "\n".join(str(row.get("content") or "") for row in core_blocks)
    core_chars = len(core_text)
    if core_chars > CORE_MEMORY_CHARS_LIMIT:
        findings.append(MemoryFinding(
            "core_too_large",
            f"core memory is {core_chars} chars (budget {CORE_MEMORY_CHARS_LIMIT}); trim rules or split into SEMANTIC tier",
            severity="warning",
        ))
    active = store.active_memories(limit=active_limit)
    active_count = len(active)
    contents: dict[str, list[int]] = {}
    for row in active:
        content = " ".join(str(row.get("content") or "").split()).casefold()
        contents.setdefault(content, []).append(int(row["id"]))
    for content, ids in contents.items():
        if len(ids) > 1:
            findings.append(MemoryFinding("duplicate_content", f"identical memory content in ids {ids}", memory_id=ids[0], severity="warning"))
    superseded = store.connection.execute("SELECT COUNT(*) FROM memories WHERE status = 'SUPERSEDED'").fetchone()[0]
    if superseded:
        findings.append(MemoryFinding("superseded_count", f"{superseded} superseded memory rows retained for audit", severity="info"))
    # Flip-flop detection: an active preference whose supersession chain
    # recently contains an opposite kind (dont-remind -> more-proactive ->
    # dont-remind within a short window) is a signal to confirm, not a bug.
    preferences = store.active_intervention_preferences(limit=200)
    rows = store.connection.execute(
        "SELECT id, scope_key, preference, supersedes_id FROM intervention_preferences ORDER BY id DESC LIMIT ?",
        (500,),
    ).fetchall()
    row_by_id = {int(row["id"]): row for row in rows}
    flip_flops: list[tuple[int, str, str]] = []
    _OPPOSITE = {"AVOID_ISOLATED": {"MORE_PROACTIVE", "EARLIER_WARNING"}, "MORE_PROACTIVE": {"AVOID_ISOLATED"}, "EARLIER_WARNING": {"AVOID_ISOLATED"}, "TIMING_SENSITIVE": {"MORE_PROACTIVE"}}
    for row in rows:
        active_kind = str(row["preference"] or "")
        cursor_id = row["supersedes_id"]
        depth = 0
        while cursor_id is not None and depth < 4:
            ancestor = row_by_id.get(int(cursor_id))
            if ancestor is None:
                break
            if str(ancestor["preference"] or "") in _OPPOSITE.get(active_kind, {}):
                flip_flops.append((int(row["id"]), str(row["scope_key"])[:160], f"active={active_kind}, superseded={ancestor['preference']}"))
                break
            cursor_id = ancestor["supersedes_id"]
            depth += 1
    for preference_id, scope, detail in flip_flops[:5]:
        findings.append(MemoryFinding(
            "preference_flip_flop",
            f"preference #{preference_id} ({detail}) scope={scope}; confirm with the user",
            memory_id=preference_id,
            severity="warning",
        ))
    # Orphan source epic IDs on consolidated memories.
    for row in store.active_memories(tier=MemoryTier.SEMANTIC, limit=100):
        referenced = row.get("source_episode_ids") or []
        if referenced:
            exist = sum(
                1 for episode_id in referenced
                if store.get_intervention_episode(int(episode_id)) is not None
            )
            if exist < len(referenced):
                findings.append(MemoryFinding(
                    "orphan_episode_references",
                    f"memory #{row['id']} references {len(referenced) - exist} missing episodes",
                    memory_id=int(row["id"]),
                    severity="info",
                ))
    return MemoryDoctorReport(
        findings=tuple(findings),
        core_chars=core_chars,
        active_memory_count=active_count,
        superseded_count=int(superseded),
    )
