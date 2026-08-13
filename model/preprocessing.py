"""
Shared feature-engineering layer for the Telco churn study.

Both `train_models.py` and the Streamlit app import from here, which guarantees
that a CSV uploaded at inference time travels through exactly the same
transformations as the training data. Every saved model is a full
`Pipeline(preprocessor -> estimator)`, so the app never has to encode anything
by hand.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COLUMNS = ["customerID"]

# TotalCharges arrives as text because 11 rows hold a single space character.
NUMERIC_LIKE = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def load_raw(path: str) -> pd.DataFrame:
    """Read the CSV without letting pandas guess types on dirty columns."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise whitespace and repair the numeric columns.

    Applied identically at train time and at upload time. Returns a copy;
    the caller's frame is never mutated.
    """
    out = df.copy()

    # "Yes" and "Yes " must not become two separate one-hot categories.
    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype).startswith("str"):
            out[col] = out[col].astype(str).str.strip()

    for col in NUMERIC_LIKE:
        if col in out.columns:
            # Blanks become NaN here and are median-imputed inside the pipeline.
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def split_features_and_target(df: pd.DataFrame):
    """Return (X, y) with y encoded as 1 = churned, 0 = retained."""
    frame = clean_frame(df)

    if TARGET not in frame.columns:
        raise ValueError(
            f"Column '{TARGET}' not found. The uploaded CSV must keep the "
            f"label column so metrics can be scored against it."
        )

    y = (
        frame[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0, "1": 1, "0": 0})
    )

    if y.isna().any():
        bad = frame.loc[y.isna(), TARGET].unique()[:5]
        raise ValueError(f"Unrecognised values in '{TARGET}': {list(bad)}")

    drop = [c for c in ID_COLUMNS + [TARGET] if c in frame.columns]
    X = frame.drop(columns=drop)

    return X, y.astype(int)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Numeric -> median impute + scale. Categorical -> mode impute + one-hot.

    Scaling is not cosmetic: kNN measures Euclidean distance, and an unscaled
    TotalCharges (0-8,700) would drown out tenure (0-72) and every binary
    indicator. Dense output is requested so GaussianNB can consume the matrix.
    """
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_branch = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_branch = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_branch, numeric_cols),
            ("cat", categorical_branch, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
