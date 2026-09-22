# Dataset: arXiv Paper Metadata

- **Source:** Kaggle — "arXiv Dataset" (Cornell-University/arxiv), mirroring arXiv's official metadata.
- **URL:** https://www.kaggle.com/datasets/Cornell-University/arxiv
- **License:** Not yet confirmed. Kaggle's dataset page crashes on load (a bug in their own page rendering, unrelated to our access). To be confirmed via `kaggle auth login` + `kaggle datasets metadata`, or by checking the page directly once Kaggle fixes the bug. Raw data is not committed to this repository regardless.
- **File used:** `arxiv-metadata-oai-snapshot.json`
- **Location (local, gitignored):** `data/raw/arxiv-metadata-oai-snapshot.json`
- **Format:** Newline-delimited JSON (one JSON object per line), not a single JSON array.
- **Verified size:** 4.6 GB, 2,895,350 records (measured 2026-09-22).
- **Fields:** id, submitter, authors, authors_parsed, title, abstract, comments, journal-ref, doi, report-no, categories, license, versions, update_date.
- **Target variable:** `categories` — space-separated arXiv category codes (multi-label).
- **Input features:** `title`, `abstract` (text).
- **Fields likely dropped in preprocessing (to be confirmed with actual null-rate analysis in Phase 3):** `journal-ref`, `doi`, `report-no`, `comments`, `submitter`, per-paper `license`.