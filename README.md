# Home Credit Default Risk — Ganesh ML Project

Production-ready refactor of the Home Credit Default Risk notebook into a clean,
modular Python project, ready for VS Code and GitHub.

---

## Project structure

```
ganesh-ml-project/
├── data/
│   ├── raw/               ← Place Kaggle CSV files here (git-ignored)
│   └── processed/         ← Parquet checkpoints (auto-generated)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── utils.py                  # Paths, logger, memory helper, constants
│   ├── data_preprocessing.py     # Load CSVs → parquet
│   ├── feature_engineering.py    # All feature transforms + merge pipeline
│   ├── model_training.py         # LR/RF baselines, LightGBM OOF, XGBoost, ensemble
│   ├── evaluation.py             # ROC, threshold optimisation, distribution plots
│   └── submission.py             # Generate submission_ensemble.csv
│
├── models/
│   └── training_results.pkl      # Serialised results dict
│
├── reports/
│   ├── figures/                  # Auto-generated PNG plots
│   └── submission_ensemble.csv   # Final Kaggle submission
│
├── app/
│   └── app.py                    # Optional FastAPI serving layer
│
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Set up environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add data

Download all CSV files from
[Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) into `data/raw/`:

```
application_train.csv    application_test.csv
bureau.csv               bureau_balance.csv
previous_application.csv installments_payments.csv
POS_CASH_balance.csv     credit_card_balance.csv
```

### 3. Run the pipeline

```bash
cd src

# Step 1 — Load raw CSVs + reduce memory → parquet
python data_preprocessing.py

# Step 2 — Feature engineering + merge
python feature_engineering.py

# Step 3 — Train LR, RF, LightGBM (OOF CV), XGBoost, ensemble
python model_training.py

# Step 4 — Evaluation plots + classification report
python evaluation.py

# Step 5 — Generate submission CSV
python submission.py
```

---

## Module overview

| Module | Responsibility |
|---|---|
| `utils.py` | Project paths, `SEED`, `N_FOLDS`, blend weights, logger, `reduce_mem_usage` |
| `data_preprocessing.py` | Load 8 CSVs, downcast dtypes, persist as parquet |
| `feature_engineering.py` | Application ratios, 5 side-table aggregations, label encoding, `fillna(0)`, merge |
| `model_training.py` | `evaluate_baseline` (LR/RF), `train_lgbm_oof`, `train_xgboost`, `blend_predictions`, `train_all` |
| `evaluation.py` | ROC curve, F1 threshold search, prediction distribution histogram |
| `submission.py` | Validates and writes `submission_ensemble.csv`; supports swapping model source |

---

## Key design decisions

| Decision | Rationale |
|---|---|
| **LightGBM OOF + XGBoost full-data blend** | Matches original notebook: LGB gets proper OOF estimates; XGB complements with a different tree algorithm |
| **`fillna(0)`** | Preserves Ganesh's original missing-value strategy |
| **`WEIGHT_LGB=0.7, WEIGHT_XGB=0.3`** | Configurable in `utils.py`; reflects original notebook weights |
| **Parquet checkpoints** | Avoids re-running slow upstream steps when iterating on models |
| **Pure functions** | Every transform is side-effect-free and independently testable |

---

## Changing blend weights

Edit `utils.py`:

```python
WEIGHT_LGB = 0.7   # ← adjust
WEIGHT_XGB = 0.3   # ← must sum to 1.0
```

## Choosing which predictions to submit

In `submission.py` (or when calling `generate_submission()`):

```python
generate_submission(results, test_df, use="ensemble")  # default
generate_submission(results, test_df, use="lgb")        # LightGBM only
generate_submission(results, test_df, use="xgb")        # XGBoost only
```
