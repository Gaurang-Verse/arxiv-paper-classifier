import pandas as pd

from arxiv_classifier.data.validation import CATEGORY_PATTERN

df = pd.read_json(
    "data/raw/arxiv-metadata-oai-snapshot.json",
    lines=True,
    nrows=200_000,
    dtype={"id": str},
)


def has_malformed_category(cat_string):
    if not isinstance(cat_string, str) or not cat_string.strip():
        return True
    return not all(CATEGORY_PATTERN.match(c) for c in cat_string.split())


flagged = df[df["categories"].apply(has_malformed_category)]
print("Flagged count in this sample:", len(flagged))
print("\nDistinct flagged category strings (first 30):")
for cat in flagged["categories"].drop_duplicates().head(30):
    print(" -", cat)