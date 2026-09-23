import json

from arxiv_classifier.tracking import log_run


def backfill(json_path, run_name, model_type):
    with open(json_path) as f:
        data = json.load(f)
    log_run(
        run_name=run_name,
        data=data,
        artifact_paths=[json_path],
        tags={"backfilled": "true", "model_type": model_type, "source_file": json_path},
    )


if __name__ == "__main__":
    backfill("data/processed/baseline_results.json", "baseline_tfidf_logreg", "baseline")
    backfill("data/processed/transformer_results.json", "distilbert_20k_local", "distilbert")
    backfill("data/processed/transformer_full_results.json", "distilbert_120k_colab", "distilbert")