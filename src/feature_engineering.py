"""
feature_engineering.py — All feature engineering and table aggregations.

Each function is pure (DataFrame in → DataFrame out).
`build_feature_matrix()` orchestrates the full pipeline.

Usage:
    python src/feature_engineering.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import DATA_PROC, get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Application table
# ─────────────────────────────────────────────────────────────────────────────

def engineer_application(df: pd.DataFrame) -> pd.DataFrame:
    """Clean & create ratio / aggregate features from application_{train|test}."""
    df = df.copy()

    # Fix DAYS_EMPLOYED sentinel
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"]      = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Age & employment in years
    df["AGE_YEARS"]      = (-df["DAYS_BIRTH"])             / 365.25
    df["EMPLOYED_YEARS"] = np.abs(df["DAYS_EMPLOYED"])     / 365.25

    # Financial ratios
    df["CREDIT_INCOME_RATIO"]  = df["AMT_CREDIT"]  / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_TERM"]          = df["AMT_ANNUITY"] / (df["AMT_CREDIT"]       + 1)
    df["INCOME_PER_PERSON"]    = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)

    # External source aggregates
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCE_STD"]  = df[ext_cols].std(axis=1)

    # Document flags total
    doc_cols = [c for c in df.columns if "FLAG_DOC" in c]
    df["DOC_COUNT"] = df[doc_cols].sum(axis=1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bureau aggregation
# ─────────────────────────────────────────────────────────────────────────────

def agg_bureau(bureau_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau table to one row per SK_ID_CURR."""
    bureau_df = bureau_df.copy()
    bureau_df["CREDIT_ACTIVE_BIN"] = (bureau_df["CREDIT_ACTIVE"] == "Active").astype(int)

    agg = bureau_df.groupby("SK_ID_CURR").agg(
        BUREAU_DAYS_CREDIT_MEAN   =("DAYS_CREDIT",        "mean"),
        BUREAU_DAYS_CREDIT_MAX    =("DAYS_CREDIT",        "max"),
        BUREAU_AMT_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM",     "mean"),
        BUREAU_AMT_CREDIT_SUM_SUM =("AMT_CREDIT_SUM",     "sum"),
        BUREAU_ACTIVE_MEAN        =("CREDIT_ACTIVE_BIN",  "mean"),
    ).reset_index()

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 3. Previous applications
# ─────────────────────────────────────────────────────────────────────────────

def agg_previous(prev_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate previous_application to one row per SK_ID_CURR."""
    prev_df = prev_df.copy()
    prev_df["APPROVED"] = (prev_df["NAME_CONTRACT_STATUS"] == "Approved").astype(int)

    agg = prev_df.groupby("SK_ID_CURR").agg(
        PREV_APP_MEAN     =("AMT_APPLICATION", "mean"),
        PREV_CREDIT_MEAN  =("AMT_CREDIT",      "mean"),
        PREV_APPROVAL_RATE=("APPROVED",         "mean"),
    ).reset_index()

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 4. Installments payments
# ─────────────────────────────────────────────────────────────────────────────

def agg_installments(inst_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate installments_payments to one row per SK_ID_CURR."""
    inst_df = inst_df.copy()
    inst_df["PAYMENT_DELAY"] = inst_df["DAYS_ENTRY_PAYMENT"] - inst_df["DAYS_INSTALMENT"]
    inst_df["PAYMENT_RATIO"] = inst_df["AMT_PAYMENT"] / (inst_df["AMT_INSTALMENT"] + 1)

    agg = inst_df.groupby("SK_ID_CURR").agg(
        INSTAL_PAYMENT_DELAY_MEAN=("PAYMENT_DELAY", "mean"),
        INSTAL_PAYMENT_DELAY_MAX =("PAYMENT_DELAY", "max"),
        INSTAL_PAYMENT_RATIO_MEAN=("PAYMENT_RATIO", "mean"),
        INSTAL_AMT_PAYMENT_SUM   =("AMT_PAYMENT",   "sum"),
    ).reset_index()

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 5. POS CASH balance
# ─────────────────────────────────────────────────────────────────────────────

def agg_pos_cash(pos_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate POS_CASH_balance to one row per SK_ID_CURR."""
    pos_df = pos_df.copy()
    pos_df["DPD_FLAG"] = (pos_df["SK_DPD"] > 0).astype(int)

    agg = pos_df.groupby("SK_ID_CURR").agg(
        POS_SK_DPD_MEAN  =("SK_DPD",    "mean"),
        POS_DPD_FLAG_MEAN=("DPD_FLAG",  "mean"),
    ).reset_index()

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 6. Credit card balance
# ─────────────────────────────────────────────────────────────────────────────

def agg_credit_card(cc_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credit_card_balance to one row per SK_ID_CURR."""
    cc_df = cc_df.copy()
    cc_df["UTIL_RATIO"] = cc_df["AMT_BALANCE"] / (cc_df["AMT_CREDIT_LIMIT_ACTUAL"] + 1)

    agg = cc_df.groupby("SK_ID_CURR").agg(
        CC_UTIL_RATIO_MEAN =("UTIL_RATIO",   "mean"),
        CC_AMT_BALANCE_MEAN=("AMT_BALANCE",  "mean"),
    ).reset_index()

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 7. Merge helper
# ─────────────────────────────────────────────────────────────────────────────

def merge_all(
    base_df: pd.DataFrame,
    tables: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Left-merge a list of (name, aggregated_df) tables onto base_df."""
    df = base_df.copy()
    for name, tbl in tables:
        before = df.shape[1]
        df = df.merge(tbl, on="SK_ID_CURR", how="left")
        logger.info("Merged %-14s +%d cols → %d total", name, df.shape[1] - before, df.shape[1])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. Categorical encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_categoricals(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label-encode object columns, fitting on the combined train+test pool."""
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        if col not in X_test.columns:
            continue
        combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col]  = le.transform(X_test[col].astype(str))
    logger.info("Encoded %d categorical columns.", len(cat_cols))
    return X_train, X_test


# ─────────────────────────────────────────────────────────────────────────────
# 9. Master builder
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    dfs: dict,
    save: bool = True,
    out_dir=DATA_PROC,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Full pipeline: engineer → aggregate → merge → encode → fillna(0).

    Returns
    -------
    X_train, X_test : feature DataFrames (TARGET and SK_ID_CURR removed)
    feature_cols    : list of feature column names
    """
    logger.info("Building feature matrix …")
    TARGET, ID = "TARGET", "SK_ID_CURR"

    # 1. Application features
    train = engineer_application(dfs["app_train"])
    test  = engineer_application(dfs["app_test"])

    # 2. Side-table aggregations
    agg_tables = [
        ("bureau",       agg_bureau(dfs["bureau"])),
        ("previous",     agg_previous(dfs["previous"])),
        ("installments", agg_installments(dfs["installments"])),
        ("pos_cash",     agg_pos_cash(dfs["pos_cash"])),
        ("credit_card",  agg_credit_card(dfs["credit_card"])),
    ]

    logger.info("Merging train …")
    train = merge_all(train, agg_tables)
    logger.info("Merging test …")
    test  = merge_all(test,  agg_tables)

    # 3. Split off target / ID
    y_series = train[TARGET].copy()
    X_train  = train.drop(columns=[TARGET, ID])
    X_test   = test.drop(columns=[ID], errors="ignore")

    # 4. Encode + align
    X_train, X_test = encode_categoricals(X_train, X_test)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # 5. Fill missing with 0 (Ganesh's original approach)
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    feature_cols = list(X_train.columns)
    logger.info("Feature matrix: train=%s  test=%s  features=%d",
                X_train.shape, X_test.shape, len(feature_cols))

    if save:
        # Reattach for parquet storage
        train_out = X_train.copy()
        train_out[TARGET] = y_series.values
        train_out[ID]     = dfs["app_train"][ID].values
        test_out  = X_test.copy()
        test_out[ID]      = dfs["app_test"][ID].values

        train_out.to_parquet(out_dir / "train_features.parquet", index=False)
        test_out.to_parquet(out_dir  / "test_features.parquet",  index=False)
        logger.info("Feature matrices saved to %s", out_dir)

    return X_train, X_test, feature_cols


if __name__ == "__main__":
    from data_preprocessing import load_processed
    dfs = load_processed()
    build_feature_matrix(dfs)
