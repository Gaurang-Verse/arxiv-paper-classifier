"""The split must be a disjoint, complete, seed-deterministic partition."""

import numpy as np

from arxiv_classifier.data.splitting import split_indices


def test_split_sizes_match_fractions():
    train_idx, val_idx, test_idx = split_indices(
        n_rows=1000, train_fraction=0.8, val_fraction=0.1, seed=42
    )
    assert len(train_idx) == 800
    assert len(val_idx) == 100
    assert len(test_idx) == 100


def test_splits_do_not_overlap():
    train_idx, val_idx, test_idx = split_indices(
        n_rows=1000, train_fraction=0.8, val_fraction=0.1, seed=42
    )
    train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)
    assert train_set & val_set == set()
    assert train_set & test_set == set()
    assert val_set & test_set == set()


def test_splits_cover_every_row_exactly_once():
    n_rows = 1000
    train_idx, val_idx, test_idx = split_indices(
        n_rows=n_rows, train_fraction=0.8, val_fraction=0.1, seed=42
    )
    covered = np.concatenate([train_idx, val_idx, test_idx])
    assert sorted(covered.tolist()) == list(range(n_rows))


def test_same_seed_gives_identical_splits():
    a = split_indices(n_rows=500, train_fraction=0.7, val_fraction=0.15, seed=7)
    b = split_indices(n_rows=500, train_fraction=0.7, val_fraction=0.15, seed=7)
    for part_a, part_b in zip(a, b):
        assert np.array_equal(part_a, part_b)


def test_different_seed_gives_different_splits():
    train_a, _, _ = split_indices(n_rows=500, train_fraction=0.7, val_fraction=0.15, seed=1)
    train_b, _, _ = split_indices(n_rows=500, train_fraction=0.7, val_fraction=0.15, seed=2)
    assert not np.array_equal(train_a, train_b)