# Baseline Model Results

TF-IDF + OneVsRest Logistic Regression, trained on a 150,000-paper random
subsample (seed 42) of the full arXiv metadata dataset. Measured on the
validation split (15,000 papers), 2026-09-22.

## Configuration

- TF-IDF: 50,000 max features, 1-2 grams, min_df=3, sublinear TF scaling.
- Model: Logistic Regression (C=1.0), one classifier per label
  (`OneVsRestClassifier`), prediction threshold 0.5.
- Label space: 172 categories (176 minus the 4 excluded for having fewer
  than 50 dataset-wide examples — see `docs/eda_findings.md`).
- Split: 80/10/10 train/val/test, random with seed 42. Mean label rate was
  0.01001 (train), 0.01008 (val), 0.01001 (test) — close enough to confirm
  the split is not meaningfully skewed.

## Results

| Metric | Value |
|---|---|
| Micro-F1 | 0.471 |
| Macro-F1 | 0.210 |

The ~2.2x gap between micro and macro F1 is the long-tail label problem
identified during EDA showing up directly in model performance:
micro-F1 is dominated by the model's reasonable performance on frequent
categories, while macro-F1 (equal weight per label) is dragged down by
categories with too few training examples to learn from.

## Per-label breakdown

Best-performing categories (all high-frequency, well-represented in
training data):

| Category | F1 | Val support |
|---|---|---|
| cs.CV | 0.802 | 900 |
| hep-ph | 0.731 | 987 |
| quant-ph | 0.721 | 896 |
| cond-mat.supr-con | 0.709 | 244 |
| cs.CL | 0.681 | 472 |

**66 of 172 categories (38%) scored F1 = 0.0** on validation. Their
validation-set support is concentrated at the low end (mostly under 70
examples, several under 10) — directly confirming that categories near
the EDA rarity cutoff are, in practice, unlearnable by this baseline
with 150K training rows. This is a real, measured result: the model isn't
broken, the categories are genuinely too sparse in this training sample.

## What happened next

This baseline (macro-F1 0.210) became the number to beat. A DistilBERT model
fine-tuned on the same 120,000 training rows beat it on both metrics:
validation micro-F1 0.593 and macro-F1 0.323 at a tuned threshold of 0.20.
The rare-category problem is still there, just smaller. See
`docs/transformer_results.md`.
