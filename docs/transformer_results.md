# Transformer Results (DistilBERT)

This document records the measured results of fine-tuning `distilbert-base-uncased`
for multi-label arXiv category classification, and compares them honestly against
the TF-IDF + Logistic Regression baseline (see `docs/baseline_results.md`).

All numbers below are taken directly from `data/processed/transformer_results.json`
(20K-row local run) and `data/processed/transformer_full_results.json` (120K-row
Colab run). Nothing here is estimated or projected — where a number isn't measured,
it isn't stated.

## Setup common to all runs

- Model: `distilbert-base-uncased`, `AutoModelForSequenceClassification` with
  `problem_type="multi_label_classification"` (independent sigmoid per label,
  BCEWithLogitsLoss)
- 172-category label space (same as baseline; categories with <50 occurrences
  in the full 150K sample excluded — see `docs/eda_findings.md`)
- max_length=256, 3 epochs, learning_rate=2e-5, weight_decay=0.01
- Identical train/val/test split (seed=42, 80/10/10) as the baseline, so the
  validation set is the same rows in every experiment in this document
- Evaluated at the default 0.5 sigmoid threshold, then swept across
  [0.05, 0.10, ..., 0.50] on the same validation set (no retraining) to check
  threshold sensitivity

## Run 1 — 20,000-row training subsample (local, Apple M4, MPS backend)

Training data was subsampled from the 120,000-row training split down to 20,000
rows (seed=123) to keep local training time manageable (~2 hours on MPS).

| Threshold | micro-F1 | macro-F1 |
|---|---|---|
| 0.05 | 0.3143 | 0.1204 |
| 0.10 | 0.3957 | 0.1030 |
| **0.15** | **0.4144** | 0.0909 |
| 0.20 | 0.3955 | 0.0751 |
| 0.25 | 0.3578 | 0.0572 |
| 0.30 | 0.3216 | 0.0441 |
| 0.50 (default) | 0.2254 | 0.0258 |

Best micro-F1 at threshold 0.15 (0.4144). Best macro-F1 at threshold 0.05 (0.1204).

At the untuned default threshold of 0.5, this run looked like a clear regression
against the baseline (micro-F1 0.225 vs. baseline 0.471). The threshold sweep
showed that was real but partly an artifact: with only 20,000 training examples
across 172 labels (many low-frequency), the model's predicted probabilities were
systematically conservative, so few labels crossed 0.5. Lowering the threshold
recovered a large share of the gap, but even at its best threshold this run still
underperformed the baseline on both metrics — consistent with 20K examples being
insufficient training signal per label for a 172-way multi-label head, compared
to the baseline's 120,000.

## Run 2 — Full 120,000-row training split (Google Colab, T4 GPU, fp16)

Same config, but with `train_subsample_size: null` (uses the entire training
split — identical 120,000 rows the baseline trained on, removing the
data-volume difference between the two models entirely).

| Threshold | micro-F1 | macro-F1 |
|---|---|---|
| 0.05 | 0.4446 | 0.3020 |
| **0.10** | 0.5381 | **0.3345** |
| 0.15 | 0.5767 | 0.3328 |
| 0.20 | 0.5934 | 0.3234 |
| **0.25** | **0.6007** | 0.3082 |
| 0.30 | 0.6006 | 0.2947 |
| 0.35 | 0.5921 | 0.2774 |
| 0.40 | 0.5813 | 0.2624 |
| 0.45 | 0.5668 | 0.2459 |
| 0.50 (default) | 0.5475 | 0.2276 |

Best micro-F1 at threshold 0.25 (0.6007). Best macro-F1 at threshold 0.10 (0.3345).

## Comparison summary

| Model | n_train | micro-F1 | macro-F1 |
|---|---|---|---|
| Baseline (TF-IDF + LogReg) | 120,000 | 0.471 | 0.210 |
| DistilBERT, 20K, @0.5 | 20,000 | 0.225 | 0.026 |
| DistilBERT, 20K, best threshold | 20,000 | 0.414 (@0.15) | 0.120 (@0.05) |
| DistilBERT, 120K, @0.5 (default) | 120,000 | 0.548 | 0.228 |
| **DistilBERT, 120K, best threshold** | **120,000** | **0.601 (@0.25)** | **0.335 (@0.10)** |

## Interpretation

Two separate factors were confounded in the initial 20K run, and both mattered:

1. **Threshold miscalibration.** A fixed 0.5 cutoff on independent per-label
   sigmoids is not automatically well-calibrated for a 172-way multi-label
   problem with a long-tailed label distribution. Sweeping the threshold on
   the *same trained model* (no retraining) recovered a substantial share of
   performance in both runs — most dramatically in the 20K run, where macro-F1
   improved roughly 4.6x (0.026 → 0.120) just by lowering the cutoff.

2. **Training data volume.** Even after correcting the threshold, the 20K-row
   model still underperformed the baseline. Matching the baseline's full
   120,000-row training set closed that remaining gap and then some: the
   full-data transformer beats the baseline on both metrics, at the default
   threshold and more so at a tuned one.

This was a genuine, measured negative result at first (20K run) that inverted
once the confounds were identified and corrected — not a result chosen after
the fact to look better. Both the 20K and 120K results are kept in this
document and in the repository's `data/processed/` results files, rather than
only keeping the final favorable one.

## Threshold selection

Micro-F1 and macro-F1 are optimized at different thresholds (0.25 vs. 0.10),
reflecting the usual micro/macro tension in long-tailed multi-label problems:
a higher threshold favors precision on frequent categories (driving micro-F1,
which is dominated by high-support labels), while a lower threshold helps
recall on rare categories (driving macro-F1, which weights all 172 categories
equally regardless of frequency).

**Threshold used going forward: 0.20.** This is a compromise position — not
the argmax of either single metric — chosen because it sits in the region
where both metrics are still near their peaks (micro-F1 0.593 vs. peak 0.601;
macro-F1 0.323 vs. peak 0.335) rather than optimizing one at meaningful cost
to the other. This is a documented engineering choice, not a measured result;
a different downstream use case (e.g. one that specifically cares about
correctly identifying rare subject categories) might reasonably prefer 0.10
instead.

## Model selection for later phases

The full-120K-data DistilBERT model (threshold 0.20) is used as the production
model for inference, the API, and monitoring in later phases, since it
outperforms the baseline on both metrics under an identical training-data
budget. The baseline model and its results remain in the repository as the
reference point this decision is based on.

## Limitations

- Single training run per configuration (no repeated-seed variance estimate).
  Model weight initialization and data loader shuffling are not seeded, so
  a repeat run would produce a similar but not identical result.
- No hyperparameter search (learning rate, epoch count, max_length were fixed
  based on common DistilBERT fine-tuning defaults, not tuned for this task).
- Only `distilbert-base-uncased` was tried; no comparison to other transformer
  architectures or sizes.
- The test set has not been touched in any experiment in this document — all
  numbers above are validation-set metrics, consistent with the project's
  leakage-prevention rule of only evaluating on test once, at the end.