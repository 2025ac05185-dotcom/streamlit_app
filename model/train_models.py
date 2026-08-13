"""
Fit five classifiers on the Telco churn dataset and persist them.

Run from the repository root:

    python model/train_models.py

Outputs written into model/ :
    <name>.joblib   one fitted Pipeline per algorithm
    metrics.json    machine-readable results, consumed by the Streamlit app
    metrics.md      the same table in Markdown, pasted into the README

Also writes test_data.csv at the repository root - the held-out split in its
original raw form, which is the file meant to be uploaded to the deployed app.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocessing import build_preprocessor, load_raw, split_features_and_target

SEED = 42
TEST_FRACTION = 0.25

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"

# Roughly 3 in 4 customers stay, so every algorithm that exposes a class weight
# gets one. Without it the models drift toward predicting "No" for everyone,
# which scores ~73% accuracy while catching nobody worth retaining.
MODELS = {
    "logistic_regression": (
        "Logistic Regression",
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    ),
    "decision_tree": (
        "Decision Tree",
        DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=25,
            class_weight="balanced",
            random_state=SEED,
        ),
    ),
    "knn": (
        "kNN",
        KNeighborsClassifier(n_neighbors=25, weights="distance"),
    ),
    "naive_bayes": (
        "Naive Bayes",
        GaussianNB(),
    ),
    "random_forest": (
        "Random Forest (Ensemble)",
        RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
    ),
}


def score(y_true, y_pred, y_prob) -> dict:
    """The six metrics the assignment asks for, in the order it asks for them."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main() -> None:
    raw = load_raw(str(DATA_FILE))
    print(f"Loaded {len(raw):,} rows x {raw.shape[1]} columns from {DATA_FILE.name}")

    # Split the *raw* frame first so test_data.csv can be written back out in
    # the same shape a grader would download from Kaggle.
    _, y_all = split_features_and_target(raw)
    raw_train, raw_test = train_test_split(
        raw, test_size=TEST_FRACTION, stratify=y_all, random_state=SEED
    )

    raw_test.to_csv(TEST_CSV, index=False)
    print(f"Held-out split written to {TEST_CSV.name}: {len(raw_test):,} rows")

    X_train, y_train = split_features_and_target(raw_train)
    X_test, y_test = split_features_and_target(raw_test)
    print(f"Train churn rate {y_train.mean():.4f} | test churn rate {y_test.mean():.4f}")

    encoded_width = build_preprocessor(X_train).fit_transform(X_train).shape[1]
    print(f"{X_train.shape[1]} raw features -> {encoded_width} encoded columns")

    results = {}
    for key, (label, estimator) in MODELS.items():
        pipe = Pipeline(
            [("prep", build_preprocessor(X_train)), ("clf", estimator)]
        ).fit(X_train, y_train)

        # Threshold explicitly at 0.50 rather than calling .predict(). For
        # distance-weighted kNN the two can disagree on ties, and the app's
        # threshold slider works off probabilities - this keeps the numbers
        # reported here identical to what the dashboard shows at 0.50.
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)
        results[key] = {"label": label, **score(y_test, y_pred, y_prob)}

        # compress=3 keeps the forest well under GitHub's comfortable file size.
        joblib.dump(pipe, MODEL_DIR / f"{key}.joblib", compress=3)
        row = results[key]
        print(
            f"{label:26} acc={row['Accuracy']:.4f} auc={row['AUC']:.4f} "
            f"f1={row['F1']:.4f} mcc={row['MCC']:.4f}"
        )

    (MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    table = pd.DataFrame(results).T.set_index("label")
    table.index.name = "ML Model Name"
    (MODEL_DIR / "metrics.md").write_text(
        table.astype(float).round(4).to_markdown()
    )
    print("\n" + table.astype(float).round(4).to_string())


if __name__ == "__main__":
    main()
