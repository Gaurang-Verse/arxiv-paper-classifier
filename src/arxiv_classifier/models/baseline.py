"""TF-IDF + OneVsRest Logistic Regression baseline for multi-label
arXiv category classification.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.multiclass import OneVsRestClassifier


def build_vectorizer(config: dict) -> TfidfVectorizer:
    """TF-IDF over word 1-2 grams. Fit it on the training split only."""
    return TfidfVectorizer(
        max_features=config["max_features"],
        ngram_range=tuple(config["ngram_range"]),
        min_df=config["min_df"],
        sublinear_tf=config["sublinear_tf"],
    )


def build_model(config: dict) -> OneVsRestClassifier:
    """One independent logistic regression per label (multi-label via OvR)."""
    base = LogisticRegression(C=config["C"], max_iter=config["max_iter"])
    return OneVsRestClassifier(base)


def evaluate(y_true, y_pred, label_names: list[str]) -> dict:
    """Compute micro/macro F1 and a full per-label report.

    Micro-F1 reflects overall correctness weighted by label frequency.
    Macro-F1 treats every label equally, which is what actually exposes
    how badly we do on the long-tail categories (docs/eda_findings.md) - a
    single averaged number would hide that entirely.
    """
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_label_report": report,
    }