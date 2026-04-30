"""
submission.py — Generate the final Kaggle submission CSV.

Usage:
    python src/submission.py
"""

import pickle

import pandas as pd

from utils import DATA_PROC, MODELS_DIR, REPORTS_DIR, get_logger

logger = get_logger(__name__)


def generate_submission(
    results: dict,
    test_df: pd.DataFrame,
    out_path=REPORTS_DIR / "submission_ensemble.csv",
    use: str = "ensemble",           # "ensemble" | "lgb" | "xgb"
) -> pd.DataFrame:
    """
    Build and save the submission CSV.

    Parameters
    ----------
    use : which predictions to submit — "ensemble", "lgb", or "xgb"
    """
    preds_map = {
        "ensemble": results["final_preds"],
        "lgb":      results["test_lgb"],
        "xgb":      results["test_xgb"],
    }
    assert use in preds_map, f"use must be one of {list(preds_map)}"
    preds = preds_map[use]

    submission = pd.DataFrame({
        "SK_ID_CURR": test_df["SK_ID_CURR"].values,
        "TARGET":     preds,
    })

    assert submission["TARGET"].between(0, 1).all(), "Probabilities out of [0, 1]!"
    assert submission["SK_ID_CURR"].nunique() == len(submission), "Duplicate IDs found!"

    submission.to_csv(out_path, index=False)
    logger.info("Submission saved → %s  shape=%s", out_path, submission.shape)
    logger.info("Prediction stats:\n%s", submission["TARGET"].describe().to_string())
    return submission


if __name__ == "__main__":
    test_df = pd.read_parquet(DATA_PROC / "test_features.parquet")

    with open(MODELS_DIR / "training_results.pkl", "rb") as f:
        results = pickle.load(f)

    generate_submission(results, test_df)
