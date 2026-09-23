"""Draw the reproducible 150,000-paper sample used by every model.

Reads configs/data.yaml and writes data/processed/sample_150k.parquet.
Run: python scripts/create_sample.py
"""

import os

import yaml

from arxiv_classifier.data.sampling import sample_rows

with open("configs/data.yaml") as f:
    config = yaml.safe_load(f)

df = sample_rows(
    path=config["raw_path"],
    expected_total_rows=config["expected_total_rows"],
    sample_size=config["sample_size"],
    seed=config["random_seed"],
    chunksize=config["chunksize"],
    keep_columns=config["keep_columns"],
)

os.makedirs(os.path.dirname(config["sample_output_path"]), exist_ok=True)
df.to_parquet(config["sample_output_path"], index=False)

print(f"Saved {len(df)} rows to {config['sample_output_path']}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 ids: {df['id'].head(3).tolist()}")
print(f"Sample categories: {df['categories'].head(3).tolist()}")