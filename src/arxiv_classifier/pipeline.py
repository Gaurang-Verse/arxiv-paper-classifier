"""Shared data preparation pipeline used by both baseline and transformer training scripts.

Centralizing this logic guarantees the two models are compared on identical
train/val/test splits and identical label preprocessing -- if this logic lived
separately in each script, the two could silently drift apart.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from arxiv_classifier.data.splitting import split_indices
from arxiv_classifier.features.labels import build_label_matrix, load_label_space
from arxiv_classifier.features.text import combine_title_abstract


@dataclass
class PreparedDataset:
    text_full: pd.Series
    Y_full: np.ndarray
    label_space: list
    mlb: object
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def prepare_dataset(config: dict) -> PreparedDataset:
    """Load the sampled parquet file, build the label matrix, and split into
    train/val/test using the config's split fractions and random seed.

    This is the single source of truth for what "the training data" means
    across every model in this project -- baseline and transformer both call
    this, so their comparisons are guaranteed apples-to-apples.
    """
    df = pd.read_parquet(config["sample_path"])
    label_space = load_label_space(config["eda_stats_path"], config["min_label_frequency"])
    Y_full, mlb, keep_mask = build_label_matrix(df["categories"], label_space)
    text_full = combine_title_abstract(df[keep_mask]).reset_index(drop=True)
    n_rows = len(text_full)

    train_idx, val_idx, test_idx = split_indices(
        n_rows, config["train_fraction"], config["val_fraction"], config["random_seed"]
    )

    return PreparedDataset(
        text_full=text_full,
        Y_full=Y_full,
        label_space=label_space,
        mlb=mlb,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )