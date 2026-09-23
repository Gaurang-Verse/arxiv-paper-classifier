"""End-to-end data preparation on a tiny synthetic dataset written to tmp_path."""

import json

import pandas as pd
import pytest

from arxiv_classifier.pipeline import prepare_dataset


@pytest.fixture
def tiny_dataset(tmp_path):
    df = pd.DataFrame({
        "title": [f"Paper {i}" for i in range(20)],
        "abstract": [f"Abstract text number {i}." for i in range(20)],
        "categories": (["cs.LG cs.AI"] * 10) + (["stat.ML"] * 10),
    })
    sample_path = tmp_path / "sample.parquet"
    df.to_parquet(sample_path)

    stats_path = tmp_path / "eda_stats.json"
    stats_path.write_text(json.dumps({
        "category_frequency": {"cs.LG": 10, "cs.AI": 10, "stat.ML": 10}
    }))

    return {
        "sample_path": str(sample_path),
        "eda_stats_path": str(stats_path),
        "min_label_frequency": 5,
        "train_fraction": 0.6,
        "val_fraction": 0.2,
        "random_seed": 42,
    }


def test_prepare_dataset_shapes_are_consistent(tiny_dataset):
    data = prepare_dataset(tiny_dataset)
    assert len(data.text_full) == 20
    assert data.Y_full.shape == (20, 3)
    assert len(data.label_space) == 3
    assert len(data.train_idx) + len(data.val_idx) + len(data.test_idx) == 20


def test_prepare_dataset_splits_do_not_overlap(tiny_dataset):
    data = prepare_dataset(tiny_dataset)
    train_set, val_set, test_set = set(data.train_idx), set(data.val_idx), set(data.test_idx)
    assert train_set & val_set == set()
    assert train_set & test_set == set()
    assert val_set & test_set == set()


def test_prepare_dataset_is_deterministic_given_same_config(tiny_dataset):
    data_a = prepare_dataset(tiny_dataset)
    data_b = prepare_dataset(tiny_dataset)
    assert data_a.train_idx.tolist() == data_b.train_idx.tolist()