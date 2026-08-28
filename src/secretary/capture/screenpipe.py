from __future__ import annotations

import json
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ScreenpipeError(RuntimeError):
    pass


class ScreenpipeCaptureProvider:
    """Small authenticated adapter; the rest of the app only sees capture items."""

    def __init__(self, base_url: str = "http://127.0.0.1:3030", api_key: str | None = None, timeout: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.last_error: str | None = None
        self.last_error_kind: str | None = None

    def _request(self, path: str, authenticated: bool = True) -> object:
        headers = {"X-Screenpipe-Client": "api"}
        if authenticated:
            if not self.api_key:
                self.last_error_kind = "not_configured"
                self.last_error = "Screenpipe API key is not configured"
                raise ScreenpipeError("Screenpipe API key is not configured")
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(f"{self.base_url}{path}", headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except HTTPError as exc:
            self.last_error_kind = "http_error"
            self.last_error = f"HTTP {exc.code}"
            raise ScreenpipeError("Screenpipe request failed: HTTP error") from exc
        except (URLError, TimeoutError, OSError) as exc:
            self.last_error_kind = "connection_refused" if isinstance(exc, (URLError, ConnectionRefusedError)) else "transport_error"
            self.last_error = "Screenpipe endpoint is not reachable" if self.last_error_kind == "connection_refused" else "Screenpipe transport error"
            raise ScreenpipeError("Screenpipe request failed: transport error") from exc
        try:
            result = json.loads(body.decode("utf-8"))
            self.last_error = None
            self.last_error_kind = None
            return result
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.last_error_kind = "invalid_response"
            self.last_error = "Screenpipe returned invalid JSON"
            raise ScreenpipeError("Screenpipe returned invalid JSON") from exc

    def health(self) -> bool:
        try:
            response = self._request("/health", authenticated=False)
            if isinstance(response, Mapping):
                healthy = str(response.get("status", "")).lower() in {"healthy", "ok", "running"}
                if not healthy:
                    self.last_error_kind = "unhealthy"
                    self.last_error = "Screenpipe health is not healthy"
                return healthy
            return bool(response)
        except ScreenpipeError as exc:
            if self.last_error is None:
                self.last_error = str(exc)
            return False

    def authenticated_search(self, limit: int = 1) -> bool:
        try:
            query = urlencode({
                "limit": max(1, min(20, limit)),
                "start_time": "30m ago",
                "content_type": "all",
                "fields": "type,content.app_name,content.timestamp,content.frame_id",
            })
            response = self._request(f"/search?{query}")
            return isinstance(response, Mapping) and isinstance(response.get("data", []), list)
        except ScreenpipeError as exc:
            if self.last_error is None:
                self.last_error = str(exc)
            return False

    def audio_disabled(self) -> bool:
        try:
            response = self._request("/health", authenticated=False)
            if isinstance(response, Mapping):
                audio = response.get("audio_status")
                return str(audio).lower() in {"disabled", "off", "inactive", "stopped"}
        except ScreenpipeError as exc:
            if self.last_error is None:
                self.last_error = str(exc)
        return False

    def ready(self) -> bool:
        """Health plus an authenticated, bounded search probe."""
        return self.health() and self.authenticated_search(limit=1)

    def poll(self) -> list[Mapping[str, object]]:
        query = urlencode({
            "limit": 20,
            "start_time": "30s ago",
            "end_time": "now",
            "content_type": "all",
            "fields": (
                "type,content.app_name,content.window_name,content.text,content.event_source,"
                "content.text_source,content.focused,content.screen_changed,content.visual_required,"
                "content.frame_id,content.timestamp,content.browser_url,content.file_path"
            ),
            "max_content_length": 4000,
        })
        try:
            response = self._request(f"/search?{query}")
            self.last_error = None
            data = response.get("data", []) if isinstance(response, Mapping) else []
            return [item for item in data if isinstance(item, Mapping)]
        except ScreenpipeError as exc:
            self.last_error = str(exc)
            return []
