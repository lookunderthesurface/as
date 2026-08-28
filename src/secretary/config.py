from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCREENPIPE_VERSION = "0.4.41"
DEFAULT_EXCLUDED_APPS = ("1Password", "KeePass")


def build_screenpipe_command(excluded_apps: Sequence[str] = DEFAULT_EXCLUDED_APPS) -> tuple[str, ...]:
    """Build the pinned managed command from flags verified by ``record --help``."""
    command = [
        "npx.cmd",
        "-y",
        f"screenpipe@{SCREENPIPE_VERSION}",
        "record",
        "--disable-audio",
        "--disable-clipboard-capture",
    ]
    excluded = tuple(item.strip() for item in excluded_apps if item.strip())
    for app in excluded:
        command.extend(("--ignored-windows", app))
    return tuple(command)


def _csv(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    return values or default


@dataclass
class SecretaryConfig:
    project_root: Path = PROJECT_ROOT
    screenpipe_base_url: str = "http://127.0.0.1:3030"
    screenpipe_api_key: str | None = None
    capture_provider: str = "screenpipe"
    screenpipe_mode: str = "managed"
    inference_provider: str = "mock"
    cloud_provider: str = "mock"
    inference_min_interval_seconds: float = 10.0
    inference_max_pending_requests: int = 1
    inference_stale_request_seconds: float = 30.0
    inference_vision_cooldown_seconds: float = 30.0
    inference_max_text_chars: int = 6000
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_text_model: str = "qwen3-vl:2b-instruct-q4_K_M"
    ollama_vision_model: str = "qwen3-vl:2b-instruct-q4_K_M"
    ollama_timeout_seconds: float = 120.0
    ollama_keep_alive: str = "30m"
    ollama_temperature: float = 0.0
    ollama_think: bool | None = False
    vision_max_long_edge: int = 1280
    vision_jpeg_quality: int = 85
    screenpipe_ready_timeout: float = 30.0
    screenpipe_supervision_interval: float = 10.0
    screenpipe_command: tuple[str, ...] = build_screenpipe_command()
    database_path: Path = PROJECT_ROOT / "data" / "state.db"
    log_directory: Path = PROJECT_ROOT / "logs"
    excluded_apps: tuple[str, ...] = DEFAULT_EXCLUDED_APPS
    max_notifications_per_hour: int = 2
    watch_expiration_minutes: int = 20
    send_screenshots_to_cloud: bool = False
    notification_cooldown_seconds: float = 300.0
    watch_max_active_hypotheses: int = 3
    model_remember_min_confidence: float = 0.75
    model_remember_min_importance: float = 0.50
    model_watch_min_confidence: float = 0.65
    model_watch_min_importance: float = 0.50
    model_investigate_min_confidence: float = 0.70
    model_investigate_min_importance: float = 0.60
    model_notify_min_confidence: float = 0.80
    model_notify_min_importance: float = 0.75
    model_notify_min_interrupt_score: float = 0.70
    model_notify_min_watch_evidence: int = 2
    shadow_mode: bool = False

    @classmethod
    def from_environment(cls, project_root: Path | None = None) -> "SecretaryConfig":
        root = project_root or PROJECT_ROOT
        database = Path(os.getenv("SECRETARY_DB_PATH", str(root / "data" / "state.db")))
        log_directory = Path(os.getenv("SECRETARY_LOG_DIR", str(root / "logs")))
        try:
            max_notifications = max(1, int(os.getenv("SECRETARY_MAX_NOTIFICATIONS_PER_HOUR", "2")))
        except ValueError:
            max_notifications = 2
        try:
            expiry = max(1, int(os.getenv("SECRETARY_WATCH_EXPIRATION_MINUTES", "20")))
        except ValueError:
            expiry = 20
        try:
            notification_cooldown = max(0.0, float(os.getenv("SECRETARY_NOTIFICATION_COOLDOWN_SECONDS", "300")))
        except ValueError:
            notification_cooldown = 300.0
        try:
            watch_max_hypotheses = max(1, int(os.getenv("SECRETARY_MAX_ACTIVE_HYPOTHESES", "3")))
        except ValueError:
            watch_max_hypotheses = 3
        try:
            ready_timeout = max(1.0, float(os.getenv("SCREENPIPE_READY_TIMEOUT", "30")))
        except ValueError:
            ready_timeout = 30.0
        try:
            supervision_interval = max(5.0, float(os.getenv("SCREENPIPE_SUPERVISION_INTERVAL", "10")))
        except ValueError:
            supervision_interval = 10.0
        try:
            inference_min_interval = max(0.0, float(os.getenv("INFERENCE_MIN_INTERVAL_SECONDS", "10")))
        except ValueError:
            inference_min_interval = 10.0
        try:
            inference_max_pending = max(1, int(os.getenv("INFERENCE_MAX_PENDING_REQUESTS", "1")))
        except ValueError:
            inference_max_pending = 1
        try:
            inference_stale = max(0.0, float(os.getenv("INFERENCE_STALE_REQUEST_SECONDS", "30")))
        except ValueError:
            inference_stale = 30.0
        try:
            vision_cooldown = max(0.0, float(os.getenv("VISION_COOLDOWN_SECONDS", "30")))
        except ValueError:
            vision_cooldown = 30.0
        try:
            max_text_chars = max(500, int(os.getenv("INFERENCE_MAX_TEXT_CHARS", "6000")))
        except ValueError:
            max_text_chars = 6000
        try:
            ollama_timeout = max(1.0, float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")))
        except ValueError:
            ollama_timeout = 120.0
        try:
            ollama_temperature = min(2.0, max(0.0, float(os.getenv("OLLAMA_TEMPERATURE", "0"))))
        except ValueError:
            ollama_temperature = 0.0
        think_value = os.getenv("OLLAMA_THINK")
        if think_value is None or not think_value.strip() or think_value.strip().lower() in {"auto", "default"}:
            ollama_think = False
        else:
            ollama_think = think_value.strip().lower() in {"1", "true", "yes", "on"}
        try:
            vision_max_edge = max(1, int(os.getenv("VISION_MAX_LONG_EDGE", "1280")))
        except ValueError:
            vision_max_edge = 1280
        try:
            vision_quality = min(95, max(1, int(os.getenv("VISION_JPEG_QUALITY", "85"))))
        except ValueError:
            vision_quality = 85
        def _threshold(name: str, default: float) -> float:
            try:
                return min(1.0, max(0.0, float(os.getenv(name, str(default)))))
            except ValueError:
                return default
        try:
            notify_watch_evidence = max(1, int(os.getenv("POLICY_NOTIFY_MIN_WATCH_EVIDENCE", "2")))
        except ValueError:
            notify_watch_evidence = 2
        excluded_apps = _csv(os.getenv("SECRETARY_EXCLUDED_APPS"), DEFAULT_EXCLUDED_APPS)
        return cls(
            project_root=root,
            screenpipe_base_url=os.getenv("SCREENPIPE_BASE_URL", "http://127.0.0.1:3030").rstrip("/"),
            screenpipe_api_key=os.getenv("SCREENPIPE_API_KEY") or os.getenv("SCREENPIPE_LOCAL_API_KEY"),
            capture_provider=os.getenv("CAPTURE_PROVIDER", "screenpipe").lower(),
            screenpipe_mode=os.getenv("SCREENPIPE_MODE", "managed").lower(),
            inference_provider=os.getenv("INFERENCE_PROVIDER", "mock").lower(),
            cloud_provider=os.getenv("CLOUD_PROVIDER", "mock").lower(),
            inference_min_interval_seconds=inference_min_interval,
            inference_max_pending_requests=inference_max_pending,
            inference_stale_request_seconds=inference_stale,
            inference_vision_cooldown_seconds=vision_cooldown,
            inference_max_text_chars=max_text_chars,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            ollama_text_model=os.getenv("OLLAMA_TEXT_MODEL", "qwen3-vl:2b-instruct-q4_K_M"),
            ollama_vision_model=os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:2b-instruct-q4_K_M"),
            ollama_timeout_seconds=ollama_timeout,
            ollama_keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
            ollama_temperature=ollama_temperature,
            ollama_think=ollama_think,
            vision_max_long_edge=vision_max_edge,
            vision_jpeg_quality=vision_quality,
            screenpipe_ready_timeout=ready_timeout,
            screenpipe_supervision_interval=supervision_interval,
            screenpipe_command=build_screenpipe_command(excluded_apps),
            database_path=database,
            log_directory=log_directory,
            excluded_apps=excluded_apps,
            max_notifications_per_hour=max_notifications,
            notification_cooldown_seconds=notification_cooldown,
            watch_expiration_minutes=expiry,
            watch_max_active_hypotheses=watch_max_hypotheses,
            model_remember_min_confidence=_threshold("POLICY_REMEMBER_MIN_CONFIDENCE", 0.75),
            model_remember_min_importance=_threshold("POLICY_REMEMBER_MIN_IMPORTANCE", 0.50),
            model_watch_min_confidence=_threshold("POLICY_WATCH_MIN_CONFIDENCE", 0.65),
            model_watch_min_importance=_threshold("POLICY_WATCH_MIN_IMPORTANCE", 0.50),
            model_investigate_min_confidence=_threshold("POLICY_INVESTIGATE_MIN_CONFIDENCE", 0.70),
            model_investigate_min_importance=_threshold("POLICY_INVESTIGATE_MIN_IMPORTANCE", 0.60),
            model_notify_min_confidence=_threshold("POLICY_NOTIFY_MIN_CONFIDENCE", 0.80),
            model_notify_min_importance=_threshold("POLICY_NOTIFY_MIN_IMPORTANCE", 0.75),
            model_notify_min_interrupt_score=_threshold("POLICY_NOTIFY_MIN_INTERRUPT_SCORE", 0.70),
            model_notify_min_watch_evidence=notify_watch_evidence,
            shadow_mode=os.getenv("SECRETARY_SHADOW_MODE", "false").strip().lower() in {"1", "true", "yes", "on"},
            send_screenshots_to_cloud=False,
        )


def ensure_project_dirs(config: SecretaryConfig) -> None:
    config.database_path.parent.mkdir(parents=True, exist_ok=True)
    config.log_directory.mkdir(parents=True, exist_ok=True)


def resolve_launcher(command: Sequence[str]) -> str | None:
    """Resolve a launcher without changing PATH or installing anything."""
    if not command:
        return None
    executable = command[0]
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    if os.name == "nt" and executable.casefold() == "npx.cmd":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        candidate = program_files / "nodejs" / "npx.cmd"
        if candidate.is_file():
            return str(candidate)
    return None
