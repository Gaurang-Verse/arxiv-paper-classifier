"""Sanity-check text and label preprocessing before training anything.

Prints label-space size, rows dropped for having no in-scope labels,
label-matrix shape, and a sample input text.
Run: python scripts/check_preprocessing.py
"""

import json

import pandas as pd
import yaml

from arxiv_classifier.features.labels import build_label_matrix, load_label_space
from arxiv_classifier.features.text import combine_title_abstract

with open("configs/baseline.yaml") as f:
    config = yaml.safe_load(f)

df = pd.read_parquet(config["sample_path"])
print(f"Loaded sample: {df.shape}")

label_space = load_label_space(config["eda_stats_path"], config["min_label_frequency"])
print(f"Label space size: {len(label_space)}")
with open(config["eda_stats_path"]) as f:
    n_observed = len(json.load(f)["category_frequency"])
print(f"Excluded from label space: {n_observed} - {len(label_space)} = {n_observed - len(label_space)}")

Y, mlb, keep_mask = build_label_matrix(df["categories"], label_space)
print(f"\nPapers dropped (no labels left after filtering): {(~keep_mask).sum()}")
print(f"Label matrix shape: {Y.shape}")
print(f"Total label assignments: {Y.sum()}")
print(f"Mean labels per paper: {Y.sum() / Y.shape[0]:.3f}")

text = combine_title_abstract(df[keep_mask])
print(f"\nText series length: {len(text)} (should match label matrix rows)")
print(f"Mean text length (chars): {text.str.len().mean():.1f}")
print(f"\nFirst example:\n{text.iloc[0][:300]}...")