# German Credit Risk Baseline

A first end-to-end baseline for the UCI German Credit benchmark, distributed
through [OpenML's `credit-g` dataset](https://www.openml.org/search?type=data&status=active&id=31).
It predicts the dataset label `bad_credit` (`1` = bad, `0` = good) using a
scikit-learn pipeline with median imputation and scaling for numeric features,
most-frequent imputation and one-hot encoding for categorical features, and a
class-weighted logistic-regression classifier.

> **Educational portfolio project only.** This historical benchmark is small
> and has known limitations; the model is not suitable for real lending or
> other consequential decisions.

## Run it

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run credit-baseline --output-dir artifacts
```

The first run downloads `credit-g` from OpenML; scikit-learn caches it locally.
The command creates:

- `artifacts/logistic_baseline.joblib` — fitted preprocessing/model pipeline
- `artifacts/metrics.json` — held-out ROC-AUC, average precision, Brier score,
  and classification report at a 0.5 probability threshold

The split is stratified and reproducible (`--seed 42` by default). Imputation,
scaling, and encoding are fitted only on the training split to avoid leakage.

## Development

```bash
uv sync --dev
uv run ruff check .
uv run pytest -v
```

Tests use a small synthetic dataset and do not require network access. CI runs
lint and tests on each push and pull request.

## Next steps

- Add a cross-validated comparison against tree-based models.
- Study threshold selection, probability calibration, and subgroup metrics.
- Add SHAP explanations and a local prediction API after the evaluation
  baseline is established.
