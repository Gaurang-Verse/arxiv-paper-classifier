"""DistilBERT-based multi-label classifier setup."""

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def build_model(model_name: str, num_labels: int):
    # problem_type="multi_label_classification" makes the model use
    # BCEWithLogitsLoss internally (independent sigmoid per label) instead
    # of the softmax/cross-entropy used for ordinary single-label
    # classification — this is the actual mechanism that makes multi-label
    # prediction work here.
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )