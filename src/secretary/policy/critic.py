"""InterventionCritic: is this intervention worth happening now?

A deterministic, explainable judge between candidate and final decision.
It is NOT another autonomous agent and never sees pixels; it scores the
bounded DecisionContext:

    expected value ~ usefulness x confidence x relevance x urgency
                     - interrupt cost - false-alarm risk

Recommendation vocabulary: SILENT / WAIT / INVESTIGATE / NOTIFY.
Every score carries reason codes so decisions stay auditable. The policy
remains the final authority (HardRules always run last); the critic can
only suppress or annotate, never escalate on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .context import DecisionContext


REASON_RELATED_PAST_SOLUTION_AVAILABLE = "RELATED_PAST_SOLUTION_AVAILABLE"
REASON_HIGH_MEMORY_RELEVANCE = "HIGH_MEMORY_RELEVANCE"
REASON_USER_ACCEPTED_SIMILAR = "USER_PREVIOUSLY_ACCEPTED_SIMILAR"
REASON_RECENT_SIMILAR_REJECTION = "RECENT_SIMILAR_REJECTION"
REASON_GENERIC_CONTENT_PENALIZED = "GENERIC_CONTENT_PENALIZED"
REASON_LOW_URGENCY = "LOW_URGENCY"
REASON_TIMING_PREFERENCE_PENALTY = "USER_PREFERRED_BETTER_TIMING"

# Utility thresholds for the recommendation ladder.
NOTIFY_UTILITY = 0.55
WAIT_UTILITY = 0.35
INVESTIGATE_UTILITY = 0.20


@dataclass(frozen=True)
class Critique:
    recommendation: str
    utility: float
    memory_relevance: float
    timing_quality: float
    content_quality: float
    urgency: float
    interrupt_cost: float
    false_alarm_risk: float
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "recommendation": self.recommendation,
            "utility": round(self.utility, 3),
            "memory_relevance": round(self.memory_relevance, 3),
            "timing_quality": round(self.timing_quality, 3),
            "content_quality": round(self.content_quality, 3),
            "urgency": round(self.urgency, 3),
            "interrupt_cost": round(self.interrupt_cost, 3),
            "false_alarm_risk": round(self.false_alarm_risk, 3),
            "reasons": list(self.reasons),
        }


class InterventionCritic:
    """Deterministic v1 critic. No model call, fully explainable."""

    def critique(self, context: DecisionContext, *, watch_evidence: int = 0) -> Critique:
        reasons: list[str] = []

        memories = context.relevant_memories or ()
        best_score = max((float(item.get("match_score") or 0.0) for item in memories), default=0.0)
        memory_relevance = min(1.0, best_score / 12.0)
        specific_solution = any(
            str(item.get("failure_signature") or "") and str(item.get("failure_signature") or "") in str(item.get("content") or "")
            or float(item.get("match_score") or 0.0) >= 8.0
            for item in memories
        )
        if specific_solution:
            reasons.append(REASON_RELATED_PAST_SOLUTION_AVAILABLE)
        if memory_relevance >= 0.5:
            reasons.append(REASON_HIGH_MEMORY_RELEVANCE)

        # Timing quality: user-taught timing knowledge and preferences.
        timing_quality = 0.5
        kinds = set()
        for preference in context.preferences or ():
            kind = str(preference.get("preference") or "").upper()
            evidence_count = 0
            try:
                evidence_count = int(preference.get("evidence_count") or 1)
            except (TypeError, ValueError):
                evidence_count = 1
            if evidence_count >= 1:
                kinds.add(kind)
        if "AVOID_ISOLATED" in kinds and watch_evidence <= 1:
            timing_quality -= 0.35
            reasons.append(REASON_TIMING_PREFERENCE_PENALTY)
        if "TIMING_SENSITIVE" in kinds:
            timing_quality -= 0.15
        if "MORE_PROACTIVE" in kinds or "EARLIER_WARNING" in kinds:
            timing_quality += 0.15
        for item in memories or ():
            if str(item.get("tags") or "").find("timing-knowledge") >= 0:
                text = str(item.get("content") or "")
                if "well timed" in text or "wanted to hear this earlier" in text:
                    timing_quality += 0.2
                elif "staying silent" in text or "wait for stronger" in text:
                    timing_quality -= 0.2
        timing_quality = max(0.0, min(1.0, timing_quality))

        # Content quality: specific past conclusions beat generic advice.
        content_quality = 0.5
        has_specific = specific_solution or memory_relevance >= 0.4
        if has_specific:
            content_quality += 0.3
        for item in memories or ():
            if str(item.get("tags") or "").find("content-knowledge") >= 0:
                text = str(item.get("content") or "")
                if "generic advice" in text or "low value" in text:
                    if not has_specific:
                        content_quality -= 0.3
                        reasons.append(REASON_GENERIC_CONTENT_PENALIZED)
                elif "useful" in text:
                    content_quality += 0.15
        content_quality = max(0.0, min(1.0, content_quality))

        # Urgency from deterministic evidence + trajectory support.
        failure_count = context.failure_count
        urgency = min(1.0, 0.25 * max(0, failure_count))
        if context.world_state is not None and context.world_state.current_gui is not None:
            if context.world_state.current_gui.progress == "stalled":
                urgency += 0.15
            if context.world_state.current_gui.activity == "research":
                urgency += 0.1
        if urgency < 0.25:
            reasons.append(REASON_LOW_URGENCY)

        # False-alarm risk from similar past interventions the user disliked.
        negative = 0
        positive = 0
        for episode in context.similar_episodes or ():
            reaction = str(episode.get("user_reaction") or "").upper()
            if reaction in {"EXPLICIT_NEGATIVE", "REJECTED"}:
                negative += 1
            elif reaction in {"EXPLICIT_POSITIVE", "ACCEPTED"}:
                positive += 1
        false_alarm_risk = 0.0
        if negative:
            false_alarm_risk = min(1.0, 0.35 * negative)
            reasons.append(REASON_RECENT_SIMILAR_REJECTION)
        if positive and positive >= negative:
            reasons.append(REASON_USER_ACCEPTED_SIMILAR)
            false_alarm_risk = max(0.0, false_alarm_risk - 0.2)

        interrupt_cost = 0.3

        utility = (
            0.30 * memory_relevance
            + 0.25 * timing_quality
            + 0.25 * content_quality
            + 0.20 * urgency
            - 0.15 * interrupt_cost
            - 0.35 * false_alarm_risk
        )
        utility = max(0.0, min(1.0, utility))

        if utility >= NOTIFY_UTILITY:
            recommendation = "NOTIFY"
        elif utility >= WAIT_UTILITY:
            recommendation = "WAIT"
        elif utility >= INVESTIGATE_UTILITY:
            recommendation = "INVESTIGATE"
        else:
            recommendation = "SILENT"
        return Critique(
            recommendation=recommendation,
            utility=utility,
            memory_relevance=memory_relevance,
            timing_quality=timing_quality,
            content_quality=content_quality,
            urgency=urgency,
            interrupt_cost=interrupt_cost,
            false_alarm_risk=false_alarm_risk,
            reasons=tuple(dict.fromkeys(reasons)),
        )


def summarize_knowledge(memories: Sequence[dict[str, object]], *, limit: int = 2) -> str:
    """Human-readable, bounded summary of retrieved knowledge for a suggestion."""
    lines: list[str] = []
    for item in memories[:limit]:
        content = str(item.get("content") or "").strip()
        if content:
            lines.append(content[:220])
    return " ".join(lines)
