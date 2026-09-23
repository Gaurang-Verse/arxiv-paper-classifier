"""Summarize per-label baseline performance: best/worst categories and F1=0 labels.

Run: python scripts/inspect_baseline_results.py
"""

import json

with open("data/processed/baseline_results.json") as f:
    results = json.load(f)

report = results["validation_per_label_report"]

# Filter out the aggregate keys (micro avg, macro avg, weighted avg, samples avg)
per_label = {
    k: v for k, v in report.items()
    if k not in ("micro avg", "macro avg", "weighted avg", "samples avg")
}

# Sort by F1 score
sorted_labels = sorted(per_label.items(), key=lambda x: x[1]["f1-score"], reverse=True)

print("Top 15 best-performing categories:")
for label, metrics in sorted_labels[:15]:
    print(f"  {label:20s} F1={metrics['f1-score']:.3f}  support={int(metrics['support'])}")

print("\nBottom 15 worst-performing categories:")
for label, metrics in sorted_labels[-15:]:
    print(f"  {label:20s} F1={metrics['f1-score']:.3f}  support={int(metrics['support'])}")

zero_f1 = [l for l, m in per_label.items() if m["f1-score"] == 0.0]
print(f"\nCategories with F1 = 0.0: {len(zero_f1)} out of {len(per_label)}")
print(f"Their support (val set example counts): {[int(per_label[l]['support']) for l in zero_f1]}")