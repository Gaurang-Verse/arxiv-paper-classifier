import pandas as pd

df = pd.read_json(
    "data/raw/arxiv-metadata-oai-snapshot.json",
    lines=True,
    nrows=5000,
    dtype={"id": str},
)

print("id dtype:", df["id"].dtype)
print("Sample ids:", df["id"].head(5).tolist())