"""Train/validation/test splitting for the baseline dataset."""

import numpy as np


def split_indices(
    n_rows: int, train_fraction: float, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) as a random row-index partition.

    A true stratified split for multi-label data is nontrivial (a row can
    belong to several label groups at once), so we use a random split here
    and verify afterward, with real numbers, that the label distribution
    looks reasonably similar across splits rather than assuming it does.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_rows)

    n_train = int(n_rows * train_fraction)
    n_val = int(n_rows * val_fraction)

    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train : n_train + n_val]
    test_idx = shuffled[n_train + n_val :]

    return train_idx, val_idx, test_idx