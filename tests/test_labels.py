import json

import pandas as pd

from arxiv_classifier.features.labels import build_label_matrix, load_label_space


def test_load_label_space_filters_by_min_frequency(tmp_path):
    stats_path = tmp_path / "eda_stats.json"
    stats_path.write_text(json.dumps({
        "category_frequency": {
            "cs.LG": 100,
            "cs.AI": 60,
            "stat.ML": 49,   # below threshold, excluded
            "q-bio.NC": 50,  # exactly at threshold, included
        }
    }))

    result = load_label_space(str(stats_path), min_frequency=50)

    assert result == ["cs.AI", "cs.LG", "q-bio.NC"]  # sorted, stat.ML excluded


def test_build_label_matrix_filters_and_orders_correctly():
    categories = pd.Series([
        "cs.LG cs.AI",
        "stat.ML",
        "unknown.cat",         # entirely outside label space -> dropped
        "cs.LG unknown.cat",   # partially outside -> keeps only cs.LG
    ])
    label_space = ["cs.AI", "cs.LG", "stat.ML"]

    Y, mlb, keep_mask = build_label_matrix(categories, label_space)

    assert keep_mask.tolist() == [True, True, False, True]
    assert Y.shape == (3, 3)
    # Column order follows label_space: [cs.AI, cs.LG, stat.ML]
    assert Y[0].tolist() == [1, 1, 0]  # cs.LG cs.AI
    assert Y[1].tolist() == [0, 0, 1]  # stat.ML
    assert Y[2].tolist() == [0, 1, 0]  # cs.LG only (unknown.cat dropped)


def test_build_label_matrix_empty_label_space_keeps_nothing():
    categories = pd.Series(["cs.LG", "cs.AI"])
    Y, mlb, keep_mask = build_label_matrix(categories, label_space=["stat.ML"])
    assert keep_mask.tolist() == [False, False]
    assert Y.shape == (0, 1)