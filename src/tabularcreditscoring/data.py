"""Loading and validating the German Credit benchmark dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_openml


def load_german_credit() -> tuple[pd.DataFrame, pd.Series]:
    """Download German Credit (OpenML ``credit-g``) and encode bad credit as 1.

    The first call downloads the dataset from OpenML; scikit-learn caches it for
    subsequent calls. Features are returned unchanged so preprocessing can be
    learned from each training fold rather than from the full dataset.
    """
    dataset = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    features = dataset.data
    labels = dataset.target.astype("string").str.lower().map({"good": 0, "bad": 1})

    if labels.isna().any():
        unknown = sorted(dataset.target[labels.isna()].astype(str).unique())
        raise ValueError(f"Unexpected target labels in credit-g: {unknown}")

    return features, labels.astype("int8").rename("bad_credit")
