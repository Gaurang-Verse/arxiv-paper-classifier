"""Diagnostic kept for the record: confirm paper IDs load as strings.

Without dtype={"id": str}, pandas reads IDs like 0704.0001 as floats and
silently drops the leading zero (704.0001). Every read_json call in the
project passes the dtype for this reason.
"""

import pandas as pd

df = pd.read_json(
    "data/raw/arxiv-metadata-oai-snapshot.json",
    lines=True,
    nrows=5000,
    dtype={"id": str},
)

print("id dtype:", df["id"].dtype)
print("Sample ids:", df["id"].head(5).tolist())