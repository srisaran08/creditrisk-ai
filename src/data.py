"""
src/data.py — Data layer for CreditRisk AI.

Responsibilities
----------------
1. Load the raw UCI Statlog German Credit file (1,000 historical loans).
2. Define the feature schema: which columns the model may use, which are
   *excluded for compliance* (protected attributes), and which are kept only
   for the fairness audit.
3. Feature engineering (monthly repayment burden).
4. Build the scikit-learn preprocessing transformer (log-transform + scaling
   for numeric fields, one-hot encoding for categorical fields).
5. Leakage-safe stratified train / test split.

Nothing in this module looks at the target when building features, so the
whole preprocessor can live *inside* the model Pipeline and be re-fitted on
every cross-validation fold (no train/test leakage).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "german_credit.csv"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
TARGET_RAW = "credit_risk"      # 1 = good loan, 0 = bad loan (raw file)
TARGET = "default"              # 1 = defaulted / bad loan (what we predict)

# Numeric features. Several are ordinal bands coded 1-4 in the source data.
NUMERIC_FEATURES = [
    "duration",            # loan tenure in months
    "amount",              # loan amount (Deutsche Mark, original units)
    "installment_rate",    # instalment as % of disposable income, band 1 (low) - 4 (high)
    "present_residence",   # years at current address, band 1-4
    "age",                 # years
    "number_credits",      # existing credits at this bank, band 1-4
    "people_liable",       # dependants the applicant supports, 1-2
    "monthly_burden",      # ENGINEERED: amount / duration
]
# Right-skewed money fields get log1p before scaling (helps the linear model,
# harmless for trees because the transform is monotonic).
SKEWED_FEATURES = ["amount", "monthly_burden"]

CATEGORICAL_FEATURES = [
    "status",                   # checking-account balance band
    "credit_history",
    "purpose",
    "savings",
    "employment_duration",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
]

# Protected / sensitive attributes: NEVER used by the model (fair-lending
# practice — sex, marital status and nationality are not legitimate credit
# criteria). Kept in the frame only so the fairness audit can test outcomes.
EXCLUDED_PROTECTED = ["personal_status_sex", "foreign_worker"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Human-readable labels used by the web app and reason codes.
FEATURE_LABELS = {
    "duration": "Loan tenure (months)",
    "amount": "Loan amount",
    "installment_rate": "Instalment burden (% of income band)",
    "present_residence": "Years at current address (band)",
    "age": "Applicant age",
    "number_credits": "Existing credits at bank",
    "people_liable": "Number of dependants",
    "monthly_burden": "Monthly repayment size",
    "status": "Checking-account balance",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "savings": "Savings balance",
    "employment_duration": "Employment tenure",
    "other_debtors": "Guarantor / co-applicant",
    "property": "Collateral / property",
    "other_installment_plans": "Other instalment plans",
    "housing": "Housing",
    "job": "Job type",
    "telephone": "Registered telephone",
}


# ---------------------------------------------------------------------------
# Loading & feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Pure row-wise maths -> safe to apply anywhere."""
    out = df.copy()
    out["monthly_burden"] = out["amount"] / out["duration"].clip(lower=1)
    return out


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV, validate it, create the `default` target."""
    df = pd.read_csv(path)
    expected = set(NUMERIC_FEATURES) - {"monthly_burden"} | set(CATEGORICAL_FEATURES) \
        | set(EXCLUDED_PROTECTED) | {TARGET_RAW}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing columns: {sorted(missing)}")
    if df.isna().any().any():
        raise ValueError("Unexpected missing values in source data.")

    df = engineer_features(df)
    df[TARGET] = (df[TARGET_RAW] == 0).astype(int)   # bad loan -> 1
    df = df.drop(columns=[TARGET_RAW])
    for col in CATEGORICAL_FEATURES + EXCLUDED_PROTECTED:
        df[col] = df[col].astype(str)
    return df


def audit_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Groups used ONLY for the fairness audit (never fed to the model)."""
    out = pd.DataFrame(index=df.index)
    out["sex"] = np.where(df["personal_status_sex"].str.startswith("female"), "female", "male")
    out["age_band"] = pd.cut(df["age"], bins=[0, 25, 35, 50, 120],
                             labels=["<=25", "26-35", "36-50", "50+"]).astype(str)
    out["foreign_worker"] = df["foreign_worker"]
    return out


def split(df: pd.DataFrame, test_size: float = 0.2):
    """Stratified hold-out split. The test set is touched exactly once,
    at the very end, after model selection and threshold tuning."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Numeric: log1p on skewed money fields -> StandardScaler.
    Categorical: one-hot, unseen categories ignored at inference time."""
    skewed = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    plain_numeric = [c for c in NUMERIC_FEATURES if c not in SKEWED_FEATURES]
    return ColumnTransformer(
        transformers=[
            ("skew", skewed, SKEWED_FEATURES),
            ("num", StandardScaler(), plain_numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def feature_groups(preprocessor: ColumnTransformer) -> dict[str, list[int]]:
    """Map each ORIGINAL feature to the column indices it occupies after
    preprocessing (one-hot creates several). Used to roll SHAP values of the
    dummies back up to one business-readable number per feature — valid
    because SHAP values are additive."""
    names = list(preprocessor.get_feature_names_out())
    groups: dict[str, list[int]] = {f: [] for f in FEATURES}
    for i, name in enumerate(names):
        if name in groups:                       # numeric, kept its name
            groups[name].append(i)
            continue
        for cat in CATEGORICAL_FEATURES:         # one-hot: "<col>_<level>"
            if name.startswith(cat + "_"):
                groups[cat].append(i)
                break
    return groups


def category_options(df: pd.DataFrame) -> dict[str, list[str]]:
    """Allowed levels for each categorical field (drives the app's dropdowns)."""
    return {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL_FEATURES}


def numeric_ranges(df: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    """(min, max, median) for each raw numeric input (drives the app's sliders)."""
    raw = [c for c in NUMERIC_FEATURES if c != "monthly_burden"]
    return {c: (float(df[c].min()), float(df[c].max()), float(df[c].median())) for c in raw}
