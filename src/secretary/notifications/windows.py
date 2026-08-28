from __future__ import annotations


class WindowsNotificationError(RuntimeError):
    pass


class WindowsNotificationProvider:
    """Optional user-space Toast provider; never changes registry, services, or policy."""

    def notify(self, title: str, body: str) -> None:
        try:
            from winotify import Notification  # type: ignore[import-not-found]
        except ImportError as exc:
            raise WindowsNotificationError("install the optional 'windows' extra for live Toast notifications") from exc
        try:
            toast = Notification(app_id="Ambient Secretary", title=title, msg=body)
            toast.show()
        except Exception as exc:
            raise WindowsNotificationError("Windows Toast provider failed") from exc

