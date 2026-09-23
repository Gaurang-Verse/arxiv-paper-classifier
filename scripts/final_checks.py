"""Last round of data checks before calling the project done.

1. Duplicate paper IDs inside the 150K sample. The raw file has 26 duplicate
   IDs (docs/data_validation.md) and no de-duplication step was added, so
   this measures whether any of them actually reached the sample.
2. Exact-duplicate input texts shared between train and val/test. This is a
   direct leakage path: the model could be scored on text it trained on.
3. How often max_length=256 truncates the title + abstract input.

Run: python scripts/final_checks.py
"""

import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer

from arxiv_classifier.pipeline import prepare_dataset

TOKENIZER_DIR = "data/processed/transformer_full_checkpoints/final"
N_LENGTH_SAMPLE = 5000

with open("configs/transformer_full.yaml") as f:
    config = yaml.safe_load(f)

# 1. Duplicate IDs in the sample
sample = pd.read_parquet(config["sample_path"])
dup_ids = int(sample["id"].duplicated().sum())
print(f"[1] Duplicate IDs in the {len(sample):,}-row sample: {dup_ids}")

# 2. Exact text overlap between train and val/test
data = prepare_dataset(config)
text = data.text_full
train_texts = set(text.iloc[data.train_idx])
val_overlap = int(text.iloc[data.val_idx].isin(train_texts).sum())
test_overlap = int(text.iloc[data.test_idx].isin(train_texts).sum())
print(f"[2] Val texts also in train:  {val_overlap} / {len(data.val_idx):,}")
print(f"    Test texts also in train: {test_overlap} / {len(data.test_idx):,}")

# 3. Token-length distribution on a random sample of inputs
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
rng = np.random.default_rng(0)
idx = rng.choice(len(text), size=N_LENGTH_SAMPLE, replace=False)
lengths = np.array([len(tokenizer(t)["input_ids"]) for t in text.iloc[idx]])
max_len = config["max_length"]
print(f"[3] Token length over {N_LENGTH_SAMPLE:,} random inputs: "
      f"median={int(np.median(lengths))}  p90={int(np.percentile(lengths, 90))}  "
      f"max={int(lengths.max())}")
print(f"    Truncated at max_length={max_len}: {(lengths > max_len).mean():.1%}")
