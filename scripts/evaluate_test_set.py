"""One-time final evaluation of the selected model on the held-out TEST set.

The test split has not been used by any earlier experiment: every model,
threshold, and config choice was made on validation. This script reports
test metrics for the final model at the threshold chosen on validation.

Run it once. Changing anything because of its output and re-running would
turn the test set into a second validation set.

Run: python scripts/evaluate_test_set.py
"""

import json

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from arxiv_classifier.pipeline import prepare_dataset

MODEL_DIR = "data/processed/transformer_full_checkpoints/final"
CHOSEN_THRESHOLD = 0.20  # selected on validation, see docs/transformer_results.md
BATCH_SIZE = 64

with open("configs/transformer_full.yaml") as f:
    config = yaml.safe_load(f)

data = prepare_dataset(config)
test_idx = data.test_idx
texts = data.text_full.iloc[test_idx].tolist()
y_true = data.Y_full[test_idx]

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
assert model.config.num_labels == y_true.shape[1], "label space / model mismatch"

# Dynamic padding per batch (padding=True) is faster than padding everything
# to max_length, and gives the same outputs because padded positions are masked.
batches = []
with torch.no_grad():
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        enc = tokenizer(
            batch, truncation=True, max_length=config["max_length"],
            padding=True, return_tensors="pt",
        ).to(device)
        batches.append(torch.sigmoid(model(**enc).logits).cpu().numpy())
        if (start // BATCH_SIZE) % 20 == 0:
            print(f"{start + len(batch)}/{len(texts)}")
probs = np.vstack(batches)

# The 0.5 row is reported only for transparency. The headline number is the
# threshold that was fixed on validation before this script was ever run.
results = {"n_test": len(test_idx), "device_used": device}
for name, threshold in [("chosen_threshold", CHOSEN_THRESHOLD), ("default_threshold", 0.5)]:
    preds = (probs >= threshold).astype(int)
    micro = f1_score(y_true, preds, average="micro", zero_division=0)
    macro = f1_score(y_true, preds, average="macro", zero_division=0)
    results[name] = {"threshold": threshold, "micro_f1": micro, "macro_f1": macro}
    print(f"TEST  threshold={threshold:.2f}  micro_f1={micro:.4f}  macro_f1={macro:.4f}")

with open("data/processed/test_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved results to data/processed/test_results.json")
