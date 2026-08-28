from __future__ import annotations

import logging
import hashlib
from pathlib import Path


def build_logger(log_directory: Path) -> logging.Logger:
    log_directory.mkdir(parents=True, exist_ok=True)
    logger_name = "ambient-secretary-" + hashlib.sha1(str(log_directory.resolve()).encode("utf-8")).hexdigest()[:12]
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.FileHandler(log_directory / "secretary.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
