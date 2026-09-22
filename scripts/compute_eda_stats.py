"""Stream the raw arXiv file once and compute lightweight summary statistics
for EDA, saved to data/processed/eda_stats.json.

We compute stats via a single streaming pass (not by loading the full
DataFrame into memory) because the raw file is 4.6GB and contains nested
list columns that would multiply memory usage significantly if loaded whole.
"""

import json
from collections import Counter

import pandas as pd

RAW_PATH = "data/raw/arxiv-metadata-oai-snapshot.json"
OUTPUT_PATH = "data/processed/eda_stats.json"
CHUNKSIZE = 100_000

# Word-count histogram buckets for title/abstract length distributions.
WORD_BUCKETS = [0, 5, 10, 20, 40, 80, 160, 320, 640, float("inf")]


def bucket_label(n_words: int) -> str:
    for i in range(len(WORD_BUCKETS) - 1):
        if WORD_BUCKETS[i] <= n_words < WORD_BUCKETS[i + 1]:
            lo, hi = WORD_BUCKETS[i], WORD_BUCKETS[i + 1]
            return f"{int(lo)}-{int(hi)}" if hi != float("inf") else f"{int(lo)}+"
    return "unknown"


def compute_stats(path: str, chunksize: int) -> dict:
    category_counts = Counter()       # individual category label frequency
    labels_per_paper_counts = Counter()  # how many labels does a paper have
    title_length_buckets = Counter()
    abstract_length_buckets = Counter()

    total_rows = 0
    title_word_sum = 0
    abstract_word_sum = 0

    reader = pd.read_json(path, lines=True, chunksize=chunksize, dtype={"id": str})

    for chunk in reader:
        total_rows += len(chunk)

        cats_split = chunk["categories"].str.split()
        labels_per_paper_counts.update(cats_split.apply(len))
        for cats in cats_split:
            category_counts.update(cats)

        title_words = chunk["title"].fillna("").str.split().apply(len)
        abstract_words = chunk["abstract"].fillna("").str.split().apply(len)

        title_word_sum += int(title_words.sum())
        abstract_word_sum += int(abstract_words.sum())

        title_length_buckets.update(title_words.apply(bucket_label))
        abstract_length_buckets.update(abstract_words.apply(bucket_label))

    return {
        "total_rows": total_rows,
        "category_frequency": dict(category_counts.most_common()),
        "labels_per_paper_distribution": {str(k): v for k, v in sorted(labels_per_paper_counts.items())},
        "title_word_count_mean": title_word_sum / total_rows,
        "abstract_word_count_mean": abstract_word_sum / total_rows,
        "title_length_buckets": dict(title_length_buckets),
        "abstract_length_buckets": dict(abstract_length_buckets),
    }


if __name__ == "__main__":
    import os

    os.makedirs("data/processed", exist_ok=True)
    stats = compute_stats(RAW_PATH, CHUNKSIZE)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved EDA stats to {OUTPUT_PATH}")
    print(f"Total rows: {stats['total_rows']}")
    print(f"Distinct categories seen: {len(stats['category_frequency'])}")
    print(f"Top 10 categories: {list(stats['category_frequency'].items())[:10]}")