from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from .schema import InferenceRequest


class ResultFreshness(str, Enum):
    FRESH = "FRESH"
    SLIGHTLY_STALE = "SLIGHTLY_STALE"
    STALE = "STALE"


@dataclass(frozen=True)
class StaleResultAssessment:
    freshness: ResultFreshness
    age_seconds: float
    generation_gap: int
    activity_compatible: bool


def assess_result(
    request: InferenceRequest,
    *,
    current_generation: int,
    current_activity: tuple[str, ...] = (),
    current_topic: str | None = None,
    stale_seconds: float = 30.0,
    stale_generation_gap: int = 2,
    now: datetime | None = None,
) -> StaleResultAssessment:
    current_time = now or datetime.now(timezone.utc)
    created = request.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age = max(0.0, (current_time.astimezone(timezone.utc) - created.astimezone(timezone.utc)).total_seconds())
    gap = max(0, current_generation - request.generation_id)
    request_activity = tuple(str(value).casefold() for value in request.activity_snapshot if str(value))
    current_activity_folded = tuple(str(value).casefold() for value in current_activity if str(value))
    activity_compatible = bool(request_activity and current_activity_folded and set(request_activity) & set(current_activity_folded))
    topic_compatible = bool(request.topic_snapshot and current_topic and request.topic_snapshot.casefold() == current_topic.casefold())
    compatible = activity_compatible or topic_compatible
    time_limit = max(0.0, stale_seconds)
    generation_limit = max(1, stale_generation_gap)
    if age <= time_limit and gap < generation_limit:
        freshness = ResultFreshness.FRESH
    elif compatible and age <= max(time_limit * 2.0, time_limit + 1.0) and gap < generation_limit * 2:
        freshness = ResultFreshness.SLIGHTLY_STALE
    else:
        freshness = ResultFreshness.STALE
    return StaleResultAssessment(freshness, age, gap, compatible)

