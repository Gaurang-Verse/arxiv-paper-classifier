# EDA Findings — arXiv Metadata

Computed via `scripts/compute_eda_stats.py` on the full raw file
(2,895,350 rows), 2026-09-22. Full results in `data/processed/eda_stats.json`.

## Category distribution

- 176 distinct category labels observed.
- Most common: `cs.LG` (243,308 papers, ~8.4% of the dataset).
- Median category frequency: 14,244 papers.
- Long tail: 15 categories have fewer than 1,000 papers; 4 have fewer than
  100 (`atom-ph`: 123, `acc-phys`: 49, `plasm-ph`: 38, `ao-sci`: 17,
  `bayes-an`: 16 — note the last few resemble deprecated pre-2007 arXiv
  category codes; this is a plausible explanation based on the naming
  pattern, not independently confirmed against arXiv's historical taxonomy
  records).

**Decision:** categories with fewer than 50 papers will be excluded from
the model's target label set (affects `acc-phys`: 49, `plasm-ph`: 38,
`ao-sci`: 17, `bayes-an`: 16 — 4 categories, ~120 combined category-label
occurrences out of 2,895,350 papers, negligible). Verified against the
actual preprocessing code output (`scripts/check_preprocessing.py`):
label space size 172 (176 − 4), matching this list exactly. A model cannot
learn a reliable decision boundary from 16–49 examples, and including them
would only add noise to multi-label evaluation metrics. Papers keep their
other valid labels; they simply won't be evaluated on these four rare tags.

## Labels per paper

- 52.4% of papers have exactly 1 label; 47.6% have 2 or more (up to 13).
- **Confirms this is a genuine multi-label problem**, not a degenerate
  case that would have been better modeled as ordinary multi-class.

## Text length

- Mean title length: 9.83 words. 97%+ of titles fall between 5-20 words;
  negligible outliers up to 40-80 words.
- Mean abstract length: 144.19 words. ~83% of abstracts fall in the
  80-320 word range; only 1,473 of 2,895,350 (0.05%) exceed 320 words.

**Decision:** a standard transformer max-sequence-length (e.g. 512 tokens)
will cover the overwhelming majority of abstracts without truncation. No
elaborate truncation/chunking strategy needed; accept truncation on the
negligible long-tail of outlier abstracts.