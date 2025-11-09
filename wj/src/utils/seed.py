"""랜덤 시드 고정 유틸."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """torch, numpy, random 시드 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_from_env(env_var: str = "SEED", default: int = 42) -> int:
    """환경 변수에서 시드 값을 읽어 설정."""
    value = int(os.getenv(env_var, default))
    set_seed(value)
    return value


__all__ = ["set_seed", "seed_from_env"]

