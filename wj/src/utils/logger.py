"""로깅 및 실험 디렉터리 관리."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import PROJECT_ROOT


def create_experiment_dir(root: Path, experiment_name: str) -> Path:
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = root / experiment_name / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def setup_logger(
    name: str,
    *,
    log_dir: Optional[Path] = None,
    level: str = "INFO",
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / f"{name}.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


__all__ = ["setup_logger", "create_experiment_dir"]

