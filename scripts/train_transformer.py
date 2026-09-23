import json
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score
from arxiv_classifier.tracking import log_run
from arxiv_classifier.data.splitting import split_indices
from arxiv_classifier.data.torch_dataset import ArxivTextDataset
from arxiv_classifier.features.labels import build_label_matrix, load_label_space
from arxiv_classifier.features.text import combine_title_abstract
from arxiv_classifier.models.transformer_model import build_model, build_tokenizer

config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/transformer.yaml"
print(f"Using config: {config_path}")

with open(config_path) as f:
    config = yaml.safe_load(f)

df = pd.read_parquet(config["sample_path"])
label_space = load_label_space(config["eda_stats_path"], config["min_label_frequency"])
Y_full, mlb, keep_mask = build_label_matrix(df["categories"], label_space)
text_full = combine_title_abstract(df[keep_mask]).reset_index(drop=True)
n_rows = len(text_full)

train_idx, val_idx, test_idx = split_indices(
    n_rows, config["train_fraction"], config["val_fraction"], config["random_seed"]
)

subsample_size = config.get("train_subsample_size")
if subsample_size in (None, "all"):
    print(f"No subsampling: using full training split ({len(train_idx)} rows)")
else:
    sub_rng = np.random.default_rng(config["train_subsample_seed"])
    train_idx = sub_rng.choice(train_idx, size=subsample_size, replace=False)
    print(f"Subsampled training split to {len(train_idx)} rows")

tokenizer = build_tokenizer(config["model_name"])
model = build_model(config["model_name"], num_labels=len(label_space))

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
model.to(device)
use_fp16 = device == "cuda"
print(f"device={device}  fp16={use_fp16}")

train_dataset = ArxivTextDataset(text_full.iloc[train_idx], Y_full[train_idx], tokenizer, config["max_length"])
val_dataset = ArxivTextDataset(text_full.iloc[val_idx], Y_full[val_idx], tokenizer, config["max_length"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= config["prediction_threshold"]).astype(int)
    return {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


training_args = TrainingArguments(
    output_dir=config["output_dir"],
    num_train_epochs=config["training"]["num_train_epochs"],
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
    learning_rate=float(config["training"]["learning_rate"]),
    weight_decay=config["training"]["weight_decay"],
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    report_to="none",
    fp16=use_fp16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

eval_history = [log for log in trainer.state.log_history if "eval_loss" in log]
eval_results = eval_history[-1] if eval_history else {}

# --- Save the final model once (not per-epoch checkpoints) ---
final_model_dir = f"{config['output_dir']}/final"
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print(f"Saved final model to {final_model_dir}")

# --- Threshold sweep diagnostic ---
predict_output = trainer.predict(val_dataset)
val_logits = predict_output.predictions
val_labels = predict_output.label_ids
val_probs = 1 / (1 + np.exp(-val_logits))

threshold_sweep = []
for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    preds = (val_probs >= threshold).astype(int)
    micro = f1_score(val_labels, preds, average="micro", zero_division=0)
    macro = f1_score(val_labels, preds, average="macro", zero_division=0)
    threshold_sweep.append({"threshold": threshold, "micro_f1": micro, "macro_f1": macro})
    print(f"threshold={threshold:.2f}  micro_f1={micro:.4f}  macro_f1={macro:.4f}")

best_by_micro = max(threshold_sweep, key=lambda r: r["micro_f1"])
best_by_macro = max(threshold_sweep, key=lambda r: r["macro_f1"])

output = {
    "config_path": config_path,
    "config": config,
    "n_train": len(train_idx),
    "n_val": len(val_idx),
    "device_used": device,
    "fp16_used": use_fp16,
    "final_eval_results_at_default_threshold": eval_results,
    "threshold_sweep": threshold_sweep,
    "best_threshold_by_micro_f1": best_by_micro,
    "best_threshold_by_macro_f1": best_by_macro,
}
with open(config["results_path"], "w") as f:
    json.dump(output, f, indent=2, default=str)

print("Saved results to", config["results_path"])

log_run(
    run_name=f"distilbert_{len(train_idx)}rows",
    data=output,
    artifact_paths=[config["results_path"]],
    tags={"model_type": "distilbert", "config_path": config_path, "backfilled": "false"},
)