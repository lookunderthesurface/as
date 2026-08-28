from __future__ import annotations

import hashlib
import re
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


_SAFE_FAILURE_SIGNATURE = re.compile(
    r"[a-z0-9][a-z0-9_.-]{0,79}:[a-z0-9][a-z0-9_.-]{0,79}",
    re.IGNORECASE,
)
_SENSITIVE_FAILURE_SIGNATURE = re.compile(
    r"(?:password|passwd|secret|token|api[-_ ]?key|authorization|bearer|cookie|private[-_ ]?key|sk-[a-z0-9])",
    re.IGNORECASE,
)
_SENSITIVE_LABEL = re.compile(
    r"\b[a-z0-9_]*?(?:password|passwd|secret|token|authorization|cookie|credential|jwt|"
    r"private[\s_-]*key|api[\s_-]*key)\b\s*(?:[:=]|is\b)?"
    r"|\bbearer\s+[a-z0-9._~+/=-]+"
    r"|\bsk-[a-z0-9]{10,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----",
    re.IGNORECASE,
)


def sanitize_failure_signature(value: str | None) -> str | None:
    """Keep stable labels while making accidental model leakage opaque."""
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    if not text:
        return None
    if _SAFE_FAILURE_SIGNATURE.fullmatch(text) and not _SENSITIVE_FAILURE_SIGNATURE.search(text):
        return text.casefold()
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"failure:opaque-{digest}"


def sanitize_semantic_label(value: object, limit: int = 500) -> str:
    """Bound semantic text and redact common credential-shaped labels."""
    text = " ".join(str(value).split()).strip() if value is not None else ""
    if not text:
        return ""
    if _SENSITIVE_LABEL.search(text):
        return "[redacted]"
    return text[: max(1, limit)]


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
