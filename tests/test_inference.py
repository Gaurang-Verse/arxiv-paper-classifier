import torch

from arxiv_classifier.inference import Predictor


class _StubOutput:
    def __init__(self, logits):
        self.logits = logits


class _StubModel:
    def __init__(self, logits):
        self._logits = logits

    def __call__(self, **kwargs):
        return _StubOutput(self._logits)

class _StubBatchEncoding(dict):
    def to(self, device):
        return self


class _StubTokenizer:
    def __call__(self, text, **kwargs):
        return _StubBatchEncoding({"input_ids": torch.zeros((1, 4), dtype=torch.long)})


def _make_predictor(logits_values, label_space, threshold=0.2):
    predictor = object.__new__(Predictor)
    predictor.label_space = label_space
    predictor.max_length = 256
    predictor.threshold = threshold
    predictor.device = "cpu"
    predictor.tokenizer = _StubTokenizer()
    predictor.model = _StubModel(torch.tensor([logits_values]))
    return predictor


def test_predict_proba_returns_one_probability_per_label():
    label_space = ["cs.AI", "cs.LG", "stat.ML"]
    predictor = _make_predictor([2.0, -1.0, 0.0], label_space)
    probs = predictor.predict_proba("some text")
    assert set(probs.keys()) == set(label_space)
    assert probs["cs.AI"] > 0.5
    assert probs["cs.LG"] < 0.5


def test_predict_applies_threshold_and_sorts_by_probability():
    label_space = ["cs.AI", "cs.LG", "stat.ML"]
    predictor = _make_predictor([2.0, -1.0, 0.5], label_space, threshold=0.4)
    predictions = predictor.predict("some text")
    assert predictions[0] == "cs.AI"
    assert "cs.LG" not in predictions
    assert "stat.ML" in predictions


def test_predict_custom_threshold_overrides_default():
    label_space = ["cs.AI", "cs.LG"]
    predictor = _make_predictor([2.0, -1.0], label_space, threshold=0.9)
    assert predictor.predict("text") == []
    assert predictor.predict("text", threshold=0.1) == ["cs.AI", "cs.LG"]