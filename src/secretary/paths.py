from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    """Portable locations for source checkouts and frozen desktop builds."""

    project_root: Path
    source_root: Path
    runtime_root: Path
    data_dir: Path
    database_path: Path
    log_directory: Path
    prompt_directory: Path

    @classmethod
    def from_environment(cls, project_root: Path | None = None) -> "AppPaths":
        source_root = Path(project_root) if project_root is not None else Path(
            getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2])
        )
        explicit_data = os.getenv("SECRETARY_DATA_DIR")
        if explicit_data:
            runtime_root = Path(explicit_data)
        elif project_root is None and os.getenv("LOCALAPPDATA"):
            runtime_root = Path(os.environ["LOCALAPPDATA"]) / "AmbientSecretary"
        else:
            runtime_root = source_root
        data_dir = runtime_root / "data" if runtime_root == source_root else runtime_root
        return cls(
            project_root=source_root,
            source_root=source_root,
            runtime_root=runtime_root,
            data_dir=data_dir,
            database_path=data_dir / "state.db",
            log_directory=(runtime_root / "logs") if runtime_root != source_root else source_root / "logs",
            prompt_directory=source_root / "prompts",
        )

    def ensure(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_directory.mkdir(parents=True, exist_ok=True)

