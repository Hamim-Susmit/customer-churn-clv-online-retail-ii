"""Utility helpers for reproducibility and filesystem setup."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(paths: Iterable[Path]) -> None:
    """Create directories if they do not exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
