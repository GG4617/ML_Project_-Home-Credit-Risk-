"""
model_training.py — Baseline LR/RF, LightGBM OOF CV, XGBoost, weighted ensemble.

Usage:
    python src/model_training.py
"""

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from utils import DATA_PROC, MODELS_DIR, SEED, N_FOLDS, WEIGHT_LGB, WEIGHT_XGB, get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Default hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_LGB_PARAMS: dict[str, Any] = {
    "n_estimators":  1000,
    "learning_rate": 0.02,
    "max_depth":     8,
    "num_leaves":    64,
    "random_state":  SEED,
    "n_jobs":        -1,
    "verbosity":     -1,
}

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators":  600,
    "learning_rate": 0.05,
    "max_depth":     6,
    "eval_metric":   "auc",
    "use_label_encoder": False,
    "random_state":  SEED,
    "n_jobs":        -1,
    "verbosity":     0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Sklearn pipeline builders
# ─────────────────────────────────────────────────────────────────────────────

def build_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=SEED,
        )),
    ])


def build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced", random_state=SEED, n_jobs=-1,
        )),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Quick validation-split baseline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_baseline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    name: str = "Model",
    test_size: float = 0.2,
) -> float:
    """Train/val split → return val AUC."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    t0 = time.time()
    pipeline.fit(X_tr, y_tr)
    auc = roc_auc_score(y_val, pipeline.predict_proba(X_val)[:, 1])
    logger.info("%s — Val AUC: %.4f  (%.1fs)", name, auc, time.time() - t0)
    return auc


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM OOF cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def train_lgbm_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    lgb_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Stratified K-Fold OOF training for LightGBM.

    Returns
    -------
    oof_preds   : shape (n_train,)
    test_preds  : shape (n_test,) — mean across folds
    fold_scores : per-fold AUC list
    """
    params = {**DEFAULT_LGB_PARAMS, **(lgb_params or {})}
    params.setdefault("scale_pos_weight", float((y == 0).sum() / (y == 1).sum()))

    skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores: list[float] = []

    logger.info("LightGBM — %d-fold OOF CV", N_FOLDS)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        model = LGBMClassifier(**params)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        oof_preds[val_idx]  = model.predict_proba(X.iloc[val_idx])[:, 1]
        test_preds         += model.predict_proba(X_test)[:, 1] / N_FOLDS
        score = roc_auc_score(y[val_idx], oof_preds[val_idx])
        fold_scores.append(score)
        logger.info("  Fold %d: AUC=%.5f", fold, score)

    overall = roc_auc_score(y, oof_preds)
    logger.info("LightGBM OOF AUC: %.5f  (mean=%.5f ± %.5f)",
                overall, np.mean(fold_scores), np.std(fold_scores))
    return oof_preds, test_preds, fold_scores


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost (full-data training for blending)
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    xgb_params: dict | None = None,
) -> np.ndarray:
    """Train XGBoost on the full training set and return test predictions."""
    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
    params.setdefault("scale_pos_weight", float((y == 0).sum() / (y == 1).sum()))

    logger.info("Training XGBoost …")
    t0    = time.time()
    model = XGBClassifier(**params)
    model.fit(X, y)
    preds = model.predict_proba(X_test)[:, 1]
    logger.info("XGBoost done  (%.1fs)", time.time() - t0)
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# Weighted ensemble
# ─────────────────────────────────────────────────────────────────────────────

def blend_predictions(
    lgb_preds: np.ndarray,
    xgb_preds: np.ndarray,
    weight_lgb: float = WEIGHT_LGB,
    weight_xgb: float = WEIGHT_XGB,
) -> np.ndarray:
    """Weighted average of LightGBM and XGBoost test predictions."""
    assert abs(weight_lgb + weight_xgb - 1.0) < 1e-6, "Weights must sum to 1."
    final = weight_lgb * lgb_preds + weight_xgb * xgb_preds
    logger.info("Ensemble — LGB×%.2f + XGB×%.2f  |  pred mean=%.4f",
                weight_lgb, weight_xgb, final.mean())
    return final


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_artifact(obj: Any, name: str, out_dir: Path = MODELS_DIR) -> Path:
    path = out_dir / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved artefact → %s", path)
    return path


def load_artifact(name: str, model_dir: Path = MODELS_DIR) -> Any:
    path = model_dir / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrated training run
# ─────────────────────────────────────────────────────────────────────────────

def train_all(
    X_train: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
) -> dict:
    """
    Full pipeline:
      1. Baseline LR + RF (val-split AUC only)
      2. LightGBM OOF CV
      3. XGBoost full-data
      4. Weighted ensemble

    Returns a results dict.
    """
    # ── Baselines ─────────────────────────────────────────────────────────────
    lr_auc = evaluate_baseline(build_lr_pipeline(), X_train, y, "LogisticRegression")
    rf_auc = evaluate_baseline(build_rf_pipeline(), X_train, y, "RandomForest")

    # ── LightGBM ──────────────────────────────────────────────────────────────
    oof_lgb, test_lgb, lgb_folds = train_lgbm_oof(X_train, y, X_test)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    test_xgb = train_xgboost(X_train, y, X_test)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    final_preds = blend_predictions(test_lgb, test_xgb)

    results = {
        "oof_lgb":     oof_lgb,
        "test_lgb":    test_lgb,
        "test_xgb":    test_xgb,
        "final_preds": final_preds,
        "blend_weights": {"lgb": WEIGHT_LGB, "xgb": WEIGHT_XGB},
        "auc_scores": {
            "LogisticRegression": lr_auc,
            "RandomForest":       rf_auc,
            "LightGBM_OOF":       roc_auc_score(y, oof_lgb),
        },
    }

    save_artifact(results, "training_results")

    logger.info("=== AUC Summary ===")
    for name, auc in results["auc_scores"].items():
        logger.info("  %-22s %.5f", name, auc)

    return results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_df = pd.read_parquet(DATA_PROC / "train_features.parquet")
    test_df  = pd.read_parquet(DATA_PROC / "test_features.parquet")

    TARGET, ID = "TARGET", "SK_ID_CURR"
    feature_cols = [c for c in train_df.columns if c not in [TARGET, ID]]

    X_train = train_df[feature_cols]
    y       = train_df[TARGET].values
    X_test  = test_df[feature_cols]

    train_all(X_train, y, X_test)
