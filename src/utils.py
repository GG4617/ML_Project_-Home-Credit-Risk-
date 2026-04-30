"""
utils.py — Shared constants, logger factory, and memory helpers.
"""

import gc
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"
FIGURES_DIR  = REPORTS_DIR / "figures"

for _d in [DATA_RAW, DATA_PROC, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED    = 42
N_FOLDS = 5

# ── Ensemble blend weights (LightGBM / XGBoost) ───────────────────────────────
WEIGHT_LGB = 0.7
WEIGHT_XGB = 0.3

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0"]

PLOT_STYLE = {
    "figure.dpi":        120,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.size":         11,
}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(h)
    logger.setLevel(level)
    return logger


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to smallest safe dtype."""
    logger = get_logger(__name__)
    start_mb = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype
        if col_type == object:
            continue
        c_min, c_max = df[col].min(), df[col].max()
        if str(col_type)[:3] == "int":
            for dtype in [np.int8, np.int16, np.int32, np.int64]:
                if c_min > np.iinfo(dtype).min and c_max < np.iinfo(dtype).max:
                    df[col] = df[col].astype(dtype)
                    break
        else:
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)

    end_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        logger.info("Memory %.1f MB → %.1f MB (%.1f%% saved)",
                    start_mb, end_mb, 100 * (start_mb - end_mb) / (start_mb + 1e-9))
    gc.collect()
    return df
