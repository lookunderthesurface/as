from __future__ import annotations

from collections.abc import Callable


class TrayUnavailable(RuntimeError):
    pass


class TrayApplication:
    """Optional tray shell. Core processing remains usable without a desktop UI."""

    def __init__(self, on_pause: Callable[[], None], on_resume: Callable[[], None], on_status: Callable[[], str], on_quit: Callable[[], None]) -> None:
        self.on_pause = on_pause
        self.on_resume = on_resume
        self.on_status = on_status
        self.on_quit = on_quit

    def run(self) -> None:
        try:
            import pystray  # type: ignore[import-not-found]
            from PIL import Image, ImageDraw  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TrayUnavailable("install the optional 'windows' extra for the system tray") from exc
        image = Image.new("RGB", (64, 64), (28, 96, 80))
        draw = ImageDraw.Draw(image)
        draw.ellipse((16, 16, 48, 48), fill=(230, 245, 235))
        menu = pystray.Menu(
            pystray.MenuItem("Pause", lambda icon, item: self.on_pause()),
            pystray.MenuItem("Resume", lambda icon, item: self.on_resume()),
            pystray.MenuItem("Status", self._show_status),
            pystray.MenuItem("Quit", lambda icon, item: (self.on_quit(), icon.stop())),
        )
        pystray.Icon("ambient-secretary", image, "Ambient Secretary", menu).run()

    def _show_status(self, icon, item) -> None:
        status = self.on_status()
        notify = getattr(icon, "notify", None)
        if callable(notify):
            try:
                notify(status, "Ambient Secretary")
            except Exception:
                # Status remains available from the callback even on tray
                # backends that do not implement notifications.
                pass
