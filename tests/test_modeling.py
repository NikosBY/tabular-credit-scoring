from pathlib import Path

import pandas as pd

from tabularcreditscoring.modeling import build_baseline, evaluate_baseline


def sample_credit_data() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        {
            "duration": [6, 12, 18, 24, 9, 30, 15, 21, 10, 36, 8, 28],
            "amount": [
                1_000,
                2_000,
                3_000,
                4_000,
                1_500,
                5_000,
                2_500,
                3_500,
                1_200,
                6_000,
                1_800,
                4_500,
            ],
            "purpose": ["radio", "car", "furniture", "car"] * 3,
        }
    )
    labels = pd.Series([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], name="bad_credit")
    return features, labels


def test_pipeline_fits_and_handles_unseen_categories() -> None:
    features, labels = sample_credit_data()
    model = build_baseline(features)
    model.fit(features.iloc[:-2], labels.iloc[:-2])

    new_case = pd.DataFrame(
        {"duration": [17], "amount": [2_700], "purpose": ["new"]}
    )
    scores = model.predict_proba(new_case)

    assert scores.shape == (1, 2)
    assert 0 <= scores[0, 1] <= 1


def test_evaluation_saves_model_and_metrics(tmp_path: Path) -> None:
    features, labels = sample_credit_data()

    metrics = evaluate_baseline(features, labels, tmp_path)

    assert 0 <= metrics["roc_auc"] <= 1
    assert (tmp_path / "logistic_baseline.joblib").is_file()
    assert (tmp_path / "metrics.json").is_file()
