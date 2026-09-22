"""Multi-label target handling for arXiv categories."""

import json

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def load_label_space(stats_path: str, min_frequency: int) -> list[str]:
    """Return the sorted list of category codes we will predict.

    The label space is derived from dataset-wide category frequencies
    measured in Phase 4, not from the training split. This is deliberate:
    the label space is a declared vocabulary (the arXiv taxonomy, minus
    codes too rare to learn), not something inferred from the relationship
    between features and targets, so fixing it up front is not leakage.
    """
    with open(stats_path) as f:
        stats = json.load(f)

    frequencies = stats["category_frequency"]
    return sorted(code for code, count in frequencies.items() if count >= min_frequency)


def build_label_matrix(categories: pd.Series, label_space: list[str]):
    """Convert the space-separated `categories` column to a binary matrix.

    Returns (Y, mlb, keep_mask). Papers whose labels are *all* outside the
    label space end up with no labels at all; `keep_mask` marks them False
    so callers can drop them, since a row with an all-zero target teaches
    the model nothing useful here.
    """
    allowed = set(label_space)

    filtered = categories.str.split().apply(
        lambda codes: [code for code in codes if code in allowed]
    )
    keep_mask = filtered.apply(len) > 0

    # `classes=` fixes the column order to our declared label space, so the
    # binariser never infers a vocabulary from whatever data it is given.
    mlb = MultiLabelBinarizer(classes=label_space)
    Y = mlb.fit_transform(filtered[keep_mask])

    return Y, mlb, keep_mask