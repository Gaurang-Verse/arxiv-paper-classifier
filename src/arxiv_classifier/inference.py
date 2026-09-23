"""Inference interface for the trained arXiv category classifier.

Loads the fine-tuned DistilBERT model once and serves predictions. The
FastAPI app calls this directly. Keeping it separate from the web layer means
it can be tested on its own and reused from a notebook or batch script.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from arxiv_classifier.features.labels import load_label_space


class Predictor:
    """Wraps tokenizer + model + label space behind a text-in, labels-out API.

    The label space is rebuilt from the same EDA stats and frequency cutoff
    used at training time, and checked against the model's output size, so a
    mismatched model/config pair fails at startup instead of silently
    returning the wrong category names.
    """

    def __init__(
        self,
        model_dir: str,
        eda_stats_path: str,
        min_label_frequency: int = 50,
        max_length: int = 256,
        threshold: float = 0.20,
        device: str | None = None,
    ):
        self.label_space = load_label_space(eda_stats_path, min_label_frequency)
        self.max_length = max_length
        self.threshold = threshold

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        if self.model.config.num_labels != len(self.label_space):
            raise ValueError(
                f"Model was trained with {self.model.config.num_labels} labels but "
                f"the loaded label space has {len(self.label_space)} entries -- "
                "these must match or predictions will be silently misaligned."
            )

    def predict_proba(self, text: str) -> dict[str, float]:
        """Return the raw sigmoid probability for every category."""
        inputs = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.sigmoid(logits).cpu().numpy()[0]
        return dict(zip(self.label_space, probs.tolist()))

    def predict(self, text: str, threshold: float | None = None) -> list[str]:
        """Return predicted category codes above the threshold, sorted by
        probability descending."""
        threshold = self.threshold if threshold is None else threshold
        probs = self.predict_proba(text)
        predicted = [(label, p) for label, p in probs.items() if p >= threshold]
        predicted.sort(key=lambda pair: pair[1], reverse=True)
        return [label for label, _ in predicted]