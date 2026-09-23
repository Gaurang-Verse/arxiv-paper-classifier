import numpy as np

from arxiv_classifier.models.baseline import build_model, build_vectorizer, evaluate


def test_build_vectorizer_applies_config():
    config = {"max_features": 100, "ngram_range": [1, 2], "min_df": 2, "sublinear_tf": True}
    vectorizer = build_vectorizer(config)
    assert vectorizer.max_features == 100
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.min_df == 2
    assert vectorizer.sublinear_tf is True


def test_build_model_applies_config():
    config = {"C": 2.0, "max_iter": 500}
    model = build_model(config)
    assert model.estimator.C == 2.0
    assert model.estimator.max_iter == 500


def test_evaluate_perfect_predictions():
    y_true = np.array([[1, 0], [0, 1], [1, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 1]])
    result = evaluate(y_true, y_pred, label_names=["a", "b"])
    assert result["micro_f1"] == 1.0
    assert result["macro_f1"] == 1.0


def test_evaluate_all_wrong_predictions():
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[0, 1], [1, 0]])
    result = evaluate(y_true, y_pred, label_names=["a", "b"])
    assert result["micro_f1"] == 0.0
    assert result["macro_f1"] == 0.0


def test_evaluate_returns_per_label_report_with_both_labels():
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1]])
    result = evaluate(y_true, y_pred, label_names=["a", "b"])
    assert "a" in result["per_label_report"]
    assert "b" in result["per_label_report"]