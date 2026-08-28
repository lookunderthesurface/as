from __future__ import annotations

from collections.abc import Mapping

from .schema import NormalizedEvent, parse_timestamp


def _content(raw: Mapping[str, object]) -> Mapping[str, object]:
    value = raw.get("content")
    if isinstance(value, Mapping):
        return value
    # Screenpipe can return the selected fields flattened as
    # ``content.app_name``/``content.timestamp`` when a fields projection is
    # requested. Reconstruct only that bounded content mapping at the
    # provider-neutral boundary.
    flattened = {
        str(key)[len("content."):]: item
        for key, item in raw.items()
        if str(key).startswith("content.")
    }
    return flattened or raw


def normalize_screenpipe_item(raw: Mapping[str, object]) -> NormalizedEvent:
    content = _content(raw)
    text = content.get("text") or content.get("transcription") or content.get("text_content") or ""
    if not isinstance(text, str):
        text = str(text)
    # Keep text transient and bounded. Privacy filtering happens before persistence/model use.
    text = text[:4000]
    app = content.get("app_name") or content.get("application") or "unknown"
    window = content.get("window_name") or content.get("window_title") or ""
    frame_id = content.get("frame_id")
    try:
        frame_id = int(frame_id) if frame_id is not None else None
    except (TypeError, ValueError):
        frame_id = None
    return NormalizedEvent(
        timestamp=parse_timestamp(content.get("timestamp")),
        source="screenpipe",
        foreground_app=str(app),
        window_title=str(window),
        event_source=str(content.get("event_source") or raw.get("type") or "screen"),
        text=text,
        text_source=str(content.get("text_source") or "unknown"),
        focused=bool(content.get("focused", True)),
        screen_changed=bool(content.get("screen_changed", True)),
        visual_required=bool(content.get("visual_required", False)),
        frame_id=frame_id,
        browser_url=str(content.get("browser_url")) if content.get("browser_url") else None,
        image_path=str(content.get("file_path")) if content.get("file_path") else None,
    )


def normalize_fixture_item(raw: Mapping[str, object]) -> NormalizedEvent:
    """Normalize a fixture or replay event without exposing provider-specific fields."""
    if raw.get("source") == "screenpipe" or "content" in raw or "type" in raw:
        return normalize_screenpipe_item(raw)
    return NormalizedEvent(
        timestamp=parse_timestamp(raw.get("timestamp")),
        source=str(raw.get("source") or "replay"),
        foreground_app=str(raw.get("foreground_app") or raw.get("app") or "unknown"),
        window_title=str(raw.get("window_title") or ""),
        event_source=str(raw.get("event_source") or "screen"),
        text=str(raw.get("text") or "")[:4000],
        text_source=str(raw.get("text_source") or "fixture"),
        focused=bool(raw.get("focused", True)),
        screen_changed=bool(raw.get("screen_changed", True)),
        visual_required=bool(raw.get("visual_required", False)),
        frame_id=int(raw["frame_id"]) if raw.get("frame_id") is not None else None,
        browser_url=str(raw.get("browser_url")) if raw.get("browser_url") else None,
        image_path=str(raw.get("image_path") or raw.get("visual_ref")) if raw.get("image_path") or raw.get("visual_ref") else None,
    )
