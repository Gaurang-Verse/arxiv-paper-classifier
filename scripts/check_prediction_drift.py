"""Prediction distribution drift check (PSI).

Compares the distribution of *predicted* categories between a reference batch
and a current batch using the Population Stability Index.

Design notes (a first version of this script got both of these wrong and
reported PSI=1.31 on same-distribution data -- see docs/monitoring.md):
- Compare predictions to predictions. Comparing model predictions against raw
  training label frequencies confuses model bias (e.g. under-predicting rare
  categories) with data drift.
- Bin coarsely. With 172 categories and a few hundred papers, most categories
  get zero counts, which inflates PSI. We keep the top-K categories from the
  reference batch and pool the rest into "other".

HONEST LIMITATION: there is no production traffic. This script runs a null
test (two disjoint validation batches, same distribution, PSI should be low)
and a positive control (a deliberately shifted batch, PSI should be high)
to show the check behaves sensibly. It does not show real drift detection.

Run: python scripts/check_prediction_drift.py
"""

import numpy as np
import yaml

from arxiv_classifier.inference import Predictor
from arxiv_classifier.pipeline import prepare_dataset

BATCH_SIZE = 300
TOP_K = 15


def predicted_counts(predictor, texts):
    counts = {}
    for text in texts:
        for label in predictor.predict(text):
            counts[label] = counts.get(label, 0) + 1
    return counts


def binned_distribution(counts, bins):
    total = sum(counts.values()) or 1
    dist = [counts.get(b, 0) / total for b in bins]
    other = sum(v for k, v in counts.items() if k not in bins) / total
    return np.array(dist + [other])


def psi(expected, actual, epsilon=1e-4):
    expected = np.clip(expected, epsilon, None)
    actual = np.clip(actual, epsilon, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


if __name__ == "__main__":
    with open("configs/transformer_full.yaml") as f:
        config = yaml.safe_load(f)

    predictor = Predictor(
        model_dir="data/processed/transformer_full_checkpoints/final",
        eda_stats_path=config["eda_stats_path"],
        min_label_frequency=config["min_label_frequency"],
    )
    data = prepare_dataset(config)
    val_idx = data.val_idx
    texts = data.text_full

    reference_idx = val_idx[:BATCH_SIZE]
    same_dist_idx = val_idx[BATCH_SIZE:2 * BATCH_SIZE]

    # Positive control: papers whose TRUE labels include a math.* category,
    # drawn from validation rows not used in the two batches above.
    math_cols = [i for i, lbl in enumerate(data.label_space) if lbl.startswith("math.")]
    pool = val_idx[2 * BATCH_SIZE:]
    has_math = data.Y_full[pool][:, math_cols].sum(axis=1) > 0
    shifted_idx = pool[has_math][:BATCH_SIZE]

    print("Predicting reference batch...")
    ref_counts = predicted_counts(predictor, texts.iloc[reference_idx])
    bins = sorted(ref_counts, key=ref_counts.get, reverse=True)[:TOP_K]
    ref_dist = binned_distribution(ref_counts, bins)

    print("Predicting same-distribution batch (null test)...")
    same_dist = binned_distribution(predicted_counts(predictor, texts.iloc[same_dist_idx]), bins)

    print("Predicting shifted batch (positive control: math papers only)...")
    shifted_dist = binned_distribution(predicted_counts(predictor, texts.iloc[shifted_idx]), bins)

    print(f"\nBatch size: {BATCH_SIZE}, bins: top {TOP_K} reference categories + 'other'")
    print(f"PSI, null test (same distribution):      {psi(ref_dist, same_dist):.4f}")
    print(f"PSI, positive control (math-only batch): {psi(ref_dist, shifted_dist):.4f}")
    print("Rule of thumb: <0.1 no significant shift, 0.1-0.25 moderate, >0.25 significant")