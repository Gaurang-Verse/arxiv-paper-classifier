# Data Validation Findings — arXiv Metadata

Measured on the full raw file (2,895,350 rows) on 2026-09-22, via
`src/arxiv_classifier/data/validation.py`.

## Results

- Total rows: 2,895,350
- Unique IDs: 2,895,324 (26 duplicate IDs found — see decision below)
- Empty title or abstract: 0 rows
- Malformed category strings: 0 (after fixing an overly strict regex during
  development — see note below)

## Null rates by column (measured, not estimated)

| Column | Null % |
|---|---|
| report-no | 93.5% |
| journal-ref | 68.2% |
| doi | 55.9% |
| comments | 26.9% |
| license | 15.6% |
| submitter | 0.5% |
| id, authors, title, categories, abstract, versions, update_date, authors_parsed | 0% |

## Preprocessing decisions (based on the above)

- **Dropped for modeling:** `report-no`, `journal-ref`, `doi`, `comments`,
  `submitter`, `license` — either too sparse to be useful, or not a
  legitimate predictive input (e.g. `submitter` identifies a person, not a
  paper property, and using it risks the model learning an author-identity
  shortcut rather than genuine text signal).
- **Input features:** `title`, `abstract` (concatenated as model input text).
- **Target:** `categories` (split on whitespace into a list of labels).
- **Kept as metadata only (not model input):** `authors`, `authors_parsed`,
  `versions`, `update_date`.
- **Duplicate IDs (26 rows):** will be de-duplicated, keeping the first
  occurrence, during the preprocessing step (Phase 4/5) — not yet
  implemented as of this document.

## Known limitation in the validation script's development

An earlier version of the category-format regex incorrectly flagged 624,245
rows (21.6%) as "malformed" due to an overly strict pattern that didn't
account for hyphenated, variable-length arXiv subcategory codes (e.g.
`cond-mat.mes-hall`, `physics.gen-ph`). Manual inspection of the flagged
samples confirmed they were valid categories, not data errors. The regex was
corrected and the validation re-run, producing the 0 result above. Noted
here for transparency about the development process.