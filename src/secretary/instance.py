from __future__ import annotations

import os
from pathlib import Path


class InstanceLock:
    """Small user-scoped, cross-platform single-instance lock."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle = None
        self._locked = False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError):
            handle.close()
            return False
        self._handle = handle
        self._locked = True
        return True

    def release(self) -> None:
        if not self._locked or self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except (OSError, IOError):
            pass
        finally:
            self._handle.close()
            self._handle = None
            self._locked = False

    def __enter__(self) -> "InstanceLock":
        if not self.acquire():
            raise RuntimeError("another Secretary instance is already running")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()

