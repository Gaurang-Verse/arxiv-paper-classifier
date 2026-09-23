"""Train and evaluate the TF-IDF + Logistic Regression baseline.

Run: python scripts/train_baseline.py
"""
import json

import numpy as np
import pandas as pd
import yaml

from arxiv_classifier.data.splitting import split_indices
from arxiv_classifier.features.labels import build_label_matrix, load_label_space
from arxiv_classifier.features.text import combine_title_abstract
from arxiv_classifier.models.baseline import build_model, build_vectorizer, evaluate
from arxiv_classifier.tracking import log_run

with open("configs/baseline.yaml") as f:
    config = yaml.safe_load(f)

print("Loading sample...")
df = pd.read_parquet(config["sample_path"])

label_space = load_label_space(config["eda_stats_path"], config["min_label_frequency"])
Y_full, mlb, keep_mask = build_label_matrix(df["categories"], label_space)
text_full = combine_title_abstract(df[keep_mask])

# Reset to a clean 0..n-1 index now that dropped rows are gone, so the
# split indices below line up correctly with both X and Y.
text_full = text_full.reset_index(drop=True)
n_rows = len(text_full)
print(f"Rows after label filtering: {n_rows}")

train_idx, val_idx, test_idx = split_indices(
    n_rows, config["train_fraction"], config["val_fraction"], config["random_seed"]
)
print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

# Sanity check: label distribution should be roughly similar across splits,
# not wildly different (which would indicate a bug in the split logic).
train_label_rate = Y_full[train_idx].mean()
val_label_rate = Y_full[val_idx].mean()
test_label_rate = Y_full[test_idx].mean()
print(f"Mean label rate — train: {train_label_rate:.5f}  val: {val_label_rate:.5f}  test: {test_label_rate:.5f}")

print("\nFitting TF-IDF vectorizer on training text...")
vectorizer = build_vectorizer(config["tfidf"])
X_train = vectorizer.fit_transform(text_full.iloc[train_idx])
X_val = vectorizer.transform(text_full.iloc[val_idx])
X_test = vectorizer.transform(text_full.iloc[test_idx])
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

print("\nTraining OneVsRest Logistic Regression (this will take a while)...")
model = build_model(config["model"])
model.fit(X_train, Y_full[train_idx])

print("\nEvaluating on validation set...")
val_probs = model.predict_proba(X_val)
val_preds = (val_probs >= config["prediction_threshold"]).astype(int)
val_results = evaluate(Y_full[val_idx], val_preds, label_space)

print(f"Validation micro-F1: {val_results['micro_f1']:.4f}")
print(f"Validation macro-F1: {val_results['macro_f1']:.4f}")

output = {
    "config": config,
    "n_train": len(train_idx),
    "n_val": len(val_idx),
    "n_test": len(test_idx),
    "vocabulary_size": len(vectorizer.vocabulary_),
    "label_rates": {
        "train": float(train_label_rate),
        "val": float(val_label_rate),
        "test": float(test_label_rate),
    },
    "validation_micro_f1": val_results["micro_f1"],
    "validation_macro_f1": val_results["macro_f1"],
    "validation_per_label_report": val_results["per_label_report"],
}

with open("data/processed/baseline_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved results to data/processed/baseline_results.json")


log_run(
    run_name=f"baseline_tfidf_logreg_{len(train_idx)}rows",
    data=output,
    artifact_paths=["data/processed/baseline_results.json"],
    tags={"model_type": "baseline", "backfilled": "false"},
)