"""Configuration constants for the churn + value proxy project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "online_retail_II.xlsx"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RANDOM_STATE = 42
CHURN_THRESHOLDS_DAYS = (60, 90, 120)
DEFAULT_CHURN_THRESHOLD = 90
DEFAULT_PREDICTION_WINDOW_DAYS = 90


@dataclass(frozen=True)
class TimeSplitConfig:
    """Configuration for time-aware splitting."""

    cutoff_days_before_snapshot: int = 120
    prediction_window_days: int = DEFAULT_PREDICTION_WINDOW_DAYS


TIME_SPLIT = TimeSplitConfig()
