"""Train and evaluate a transparent baseline for the German Credit dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabularcreditscoring.data import load_german_credit


def build_baseline(features: pd.DataFrame) -> Pipeline:
    """Build a preprocessing + logistic-regression pipeline.

    All learned preprocessing stays inside the pipeline, preventing data
    leakage when fitting only on the training split.
    """
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=2_000),
            ),
        ]
    )


def evaluate_baseline(
    features: pd.DataFrame,
    labels: pd.Series,
    output_dir: Path,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit on a stratified training split, evaluate once, and save artifacts."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=random_state,
        stratify=labels,
    )
    model = build_baseline(x_train)
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype("int8")
    metrics: dict[str, Any] = {
        "dataset": "OpenML credit-g (German Credit)",
        "target": "bad_credit (1 = bad, 0 = good)",
        "random_state": random_state,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "classification_report_at_0_5": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "logistic_baseline.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts"), help="Artifact output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Train/test split seed")
    args = parser.parse_args()

    features, labels = load_german_credit()
    metrics = evaluate_baseline(features, labels, args.output_dir, random_state=args.seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
