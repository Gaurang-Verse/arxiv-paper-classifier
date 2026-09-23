"""Validation checks for the raw arXiv metadata file.

Runs in chunks rather than loading the full file into memory, since the
raw file is several GB. Reports issues; does not modify data.
"""

import re

import pandas as pd

# arXiv category codes look like "cs.LG", "cond-mat.mes-hall", or a bare
# top-level code like "hep-ph". Main groups and subcategories can both be
# hyphenated and are variable length. This is a structural format check,
# not a lookup against arXiv's full official taxonomy (which changes over
# time) — it catches obviously corrupted values, not every invalid code.
CATEGORY_PATTERN = re.compile(r"^[a-z]+(-[a-z]+)*(\.[A-Za-z]+(-[A-Za-z]+)*)?$")


def validate_raw_file(path: str, chunksize: int = 100_000) -> dict:
    """Stream the raw JSON-lines file in chunks and collect validation stats."""
    total_rows = 0
    seen_ids = set()
    duplicate_ids = 0
    null_counts = {}
    empty_title_or_abstract = 0
    malformed_category_rows = 0

    reader = pd.read_json(path, lines=True, chunksize=chunksize, dtype={"id": str})

    for chunk in reader:
        total_rows += len(chunk)

        # ID uniqueness
        for pid in chunk["id"]:
            if pid in seen_ids:
                duplicate_ids += 1
            else:
                seen_ids.add(pid)

        # Null counts, accumulated across chunks
        chunk_nulls = chunk.isnull().sum()
        for col, count in chunk_nulls.items():
            null_counts[col] = null_counts.get(col, 0) + int(count)

        # Empty title/abstract after stripping whitespace
        empty_mask = (
            chunk["title"].fillna("").str.strip().eq("")
            | chunk["abstract"].fillna("").str.strip().eq("")
        )
        empty_title_or_abstract += int(empty_mask.sum())

        # Category format check
        def has_malformed_category(cat_string: str) -> bool:
            if not isinstance(cat_string, str) or not cat_string.strip():
                return True
            return not all(CATEGORY_PATTERN.match(c) for c in cat_string.split())

        malformed_category_rows += int(chunk["categories"].apply(has_malformed_category).sum())

    return {
        "total_rows": total_rows,
        "unique_ids": len(seen_ids),
        "duplicate_ids": duplicate_ids,
        "null_counts": null_counts,
        "empty_title_or_abstract": empty_title_or_abstract,
        "malformed_category_rows": malformed_category_rows,
    }


if __name__ == "__main__":
    import json

    results = validate_raw_file("data/raw/arxiv-metadata-oai-snapshot.json")
    print(json.dumps(results, indent=2))