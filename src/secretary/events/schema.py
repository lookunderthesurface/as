from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        try:
            result = datetime.fromisoformat(text)
        except ValueError:
            result = utc_now()
    else:
        result = utc_now()
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


@dataclass(frozen=True)
class NormalizedEvent:
    timestamp: datetime
    source: str
    foreground_app: str
    window_title: str
    event_source: str
    text: str
    text_source: str
    focused: bool
    screen_changed: bool
    visual_required: bool
    frame_id: int | None = None
    browser_url: str | None = None
    image_path: str | None = None

    @property
    def stable_id(self) -> str:
        frame = self.frame_id if self.frame_id is not None else "no-frame"
        return f"{self.source}:{frame}:{self.timestamp.isoformat()}:{self.foreground_app}"

    def safe_metadata(self) -> dict[str, object]:
        """Return metadata safe for logs and persistence; text is intentionally omitted."""
        data = asdict(self)
        data.pop("text", None)
        # A recorder path is a transient capability, not safe log or memory data.
        data.pop("image_path", None)
        data["timestamp"] = self.timestamp.isoformat()
        return data
