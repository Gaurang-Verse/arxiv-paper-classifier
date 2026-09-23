# Dataset: arXiv Paper Metadata

- **Source:** the "arXiv Dataset" on Kaggle (`Cornell-University/arxiv`), a
  mirror of arXiv's official metadata that arXiv itself links to from its
  bulk-data page.
- **URL:** https://www.kaggle.com/datasets/Cornell-University/arxiv
- **License:** CC0 1.0 Universal (Public Domain Dedication). arXiv's license
  page states that "A Creative Commons CC0 1.0 Universal Public Domain
  Dedication will apply to all metadata"
  (https://info.arxiv.org/help/license/index.html). This project only uses
  metadata (titles, abstracts, category codes), never the papers' full text,
  which carries separate per-paper licenses. Raw data is still not committed
  here because of its size.
- **File used:** `arxiv-metadata-oai-snapshot.json`
- **Location (local, gitignored):** `data/raw/arxiv-metadata-oai-snapshot.json`
- **Format:** newline-delimited JSON (one object per line), not a single JSON array.
- **Verified size:** 4.6 GB, 2,895,350 records (measured 2026-09-22). The
  sampling code checks this row count and refuses to run if the file has
  changed, because a different snapshot would give a different sample.
- **Fields:** id, submitter, authors, authors_parsed, title, abstract, comments,
  journal-ref, doi, report-no, categories, license, versions, update_date.
- **Target:** `categories`, space-separated arXiv category codes (multi-label).
- **Input:** `title` + `abstract`, concatenated.
- **Dropped:** see the null-rate table and reasoning in `docs/data_validation.md`.

## Getting the file

```bash
pip install kaggle                     # needs a Kaggle API token in ~/.kaggle/
kaggle datasets download -d Cornell-University/arxiv -p data/raw --unzip
```

Kaggle updates this snapshot over time. A newer file will have a different row
count and `scripts/create_sample.py` will stop with an error instead of quietly
drawing a different sample. Update `expected_total_rows` in `configs/data.yaml`
only if you actually intend to work with a new snapshot.
