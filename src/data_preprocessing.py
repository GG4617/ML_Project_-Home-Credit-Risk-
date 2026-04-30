"""
data_preprocessing.py — Load raw CSVs, reduce memory, save as parquet.

Usage:
    python src/data_preprocessing.py
"""

from pathlib import Path

import pandas as pd

from utils import DATA_RAW, DATA_PROC, get_logger, reduce_mem_usage

logger = get_logger(__name__)

RAW_FILES = {
    "app_train":    "application_train.csv",
    "app_test":     "application_test.csv",
    "bureau":       "bureau.csv",
    "bureau_bal":   "bureau_balance.csv",
    "previous":     "previous_application.csv",
    "installments": "installments_payments.csv",
    "pos_cash":     "POS_CASH_balance.csv",
    "credit_card":  "credit_card_balance.csv",
}


def load_all_raw(data_dir: Path = DATA_RAW) -> dict[str, pd.DataFrame]:
    """Load every raw CSV, reduce memory, return name→DataFrame dict."""
    dfs: dict[str, pd.DataFrame] = {}
    for key, filename in RAW_FILES.items():
        path = data_dir / filename
        if not path.exists():
            logger.warning("Missing file — skipping: %s", path)
            continue
        logger.info("Loading %s …", filename)
        df = pd.read_csv(path)
        df = reduce_mem_usage(df)
        logger.info("  shape: %s", df.shape)
        dfs[key] = df
    logger.info("Loaded %d / %d files.", len(dfs), len(RAW_FILES))
    return dfs


def save_processed(dfs: dict[str, pd.DataFrame], out_dir: Path = DATA_PROC) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in dfs.items():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved %s → %s", name, path)


def load_processed(out_dir: Path = DATA_PROC) -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    for key in RAW_FILES:
        path = out_dir / f"{key}.parquet"
        if path.exists():
            dfs[key] = pd.read_parquet(path)
            logger.info("Loaded %s  shape=%s", key, dfs[key].shape)
        else:
            logger.warning("Processed file not found: %s", path)
    return dfs


if __name__ == "__main__":
    logger.info("=== Data preprocessing start ===")
    save_processed(load_all_raw())
    logger.info("=== Done ===")
