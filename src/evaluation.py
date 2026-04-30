"""
evaluation.py — ROC curve, threshold optimisation, prediction distribution plots.

Usage:
    python src/evaluation.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    f1_score, classification_report,
)

from utils import DATA_PROC, FIGURES_DIR, MODELS_DIR, PALETTE, PLOT_STYLE, get_logger

logger = get_logger(__name__)
plt.rcParams.update(PLOT_STYLE)


# ─────────────────────────────────────────────────────────────────────────────
# ROC curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true: np.ndarray,
    oof_preds: dict[str, np.ndarray],
    out_path=FIGURES_DIR / "roc_curve.png",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, preds) in enumerate(oof_preds.items()):
        auc = roc_auc_score(y_true, preds)
        fpr, tpr, _ = roc_curve(y_true, preds)
        ax.plot(fpr, tpr, lw=2, label=f"{name}  AUC={auc:.4f}", color=PALETTE[i % len(PALETTE)])
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve (OOF)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved ROC curve → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold optimisation
# ─────────────────────────────────────────────────────────────────────────────

def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Return the threshold that maximises F1 on OOF predictions."""
    thresholds = np.linspace(0.1, 0.9, 50)
    scores     = [f1_score(y_true, (y_prob > t).astype(int), zero_division=0)
                  for t in thresholds]
    best_t = float(thresholds[int(np.argmax(scores))])
    logger.info("Best F1 threshold: %.4f  (F1=%.4f)", best_t, max(scores))
    return best_t


def plot_threshold_optimisation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    best_thresh: float,
    out_path=FIGURES_DIR / "threshold_optimisation.png",
) -> None:
    thresholds = np.linspace(0.1, 0.9, 50)
    scores     = [f1_score(y_true, (y_prob > t).astype(int), zero_division=0)
                  for t in thresholds]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, scores, color=PALETTE[0], lw=2)
    ax.axvline(best_thresh, color=PALETTE[2], linestyle="--",
               label=f"Best threshold = {best_thresh:.3f}")
    ax.set(xlabel="Threshold", ylabel="F1 Score", title="Threshold Optimisation")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved threshold plot → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction distribution comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_prediction_distributions(
    preds: dict[str, np.ndarray],
    out_path=FIGURES_DIR / "prediction_distributions.png",
) -> None:
    """Overlay histograms of test-set predictions for LGB, XGB, and ensemble."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (name, arr) in enumerate(preds.items()):
        ax.hist(arr, bins=60, alpha=0.5, label=name, color=PALETTE[i % len(PALETTE)])
    ax.set(xlabel="Predicted Probability", ylabel="Frequency",
           title="Prediction Distribution Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved prediction distribution plot → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Classification report
# ─────────────────────────────────────────────────────────────────────────────

def print_classification_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> None:
    y_pred = (y_prob > threshold).astype(int)
    logger.info(
        "Classification report (threshold=%.4f):\n%s",
        threshold,
        classification_report(y_true, y_pred, target_names=["No Default", "Default"]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(results: dict, y: np.ndarray) -> None:
    """Generate all evaluation artefacts from a `train_all()` results dict."""
    oof_lgb  = results["oof_lgb"]
    test_lgb = results["test_lgb"]
    test_xgb = results["test_xgb"]
    final    = results["final_preds"]

    # ROC curve on OOF
    plot_roc_curve(y, {"LightGBM OOF": oof_lgb})

    # Threshold
    best_t = find_best_threshold(y, oof_lgb)
    plot_threshold_optimisation(y, oof_lgb, best_t)
    print_classification_report(y, oof_lgb, best_t)

    # Prediction distributions
    plot_prediction_distributions({
        "LightGBM": test_lgb,
        "XGBoost":  test_xgb,
        "Ensemble": final,
    })

    logger.info("=== Prediction Stats ===")
    logger.info("  Min : %.4f", final.min())
    logger.info("  Max : %.4f", final.max())
    logger.info("  Mean: %.4f", final.mean())


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pickle
    train_df = pd.read_parquet(DATA_PROC / "train_features.parquet")
    TARGET, ID = "TARGET", "SK_ID_CURR"
    y = train_df[TARGET].values

    with open(MODELS_DIR / "training_results.pkl", "rb") as f:
        results = pickle.load(f)

    evaluate(results, y)
