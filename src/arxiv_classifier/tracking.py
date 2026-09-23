"""Thin MLflow helper so training scripts can log a run in one call.

Each training script already builds a results dict (config + metrics) that it
saves as JSON. `log_run` takes that same dict, flattens it, and sends it to
MLflow: numbers become metrics, everything else becomes params, and the JSON
file itself is attached as an artifact.

Tracking goes to a local SQLite file (mlflow.db). MLflow 3.x no longer accepts
the old plain-folder ./mlruns store. View runs with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import mlflow


def _flatten(d: dict, parent_key: str = "") -> dict:
    """Flatten nested dicts into dotted keys, e.g. training.learning_rate.

    Lists are skipped (per-label reports, threshold sweeps). They don't fit
    MLflow's one-value-per-key model, and they are still available in full in
    the JSON artifact.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten(v, new_key))
        elif isinstance(v, list):
            continue
        else:
            items[new_key] = v
    return items


def _split(flat: dict) -> tuple[dict, dict]:
    """Numbers go to metrics and everything else to params.

    Bools are checked first because bool is a subclass of int in Python and
    would otherwise be logged as a 0/1 metric.
    """
    params, metrics = {}, {}
    for k, v in flat.items():
        if isinstance(v, bool):
            params[k] = v
        elif isinstance(v, (int, float)):
            metrics[k] = v
        else:
            params[k] = v
    return params, metrics


def log_run(
    run_name: str,
    data: dict,
    artifact_paths: list[str] | None = None,
    tags: dict | None = None,
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "arxiv-classifier",
) -> None:
    """Log one training/evaluation run to MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    params, metrics = _split(_flatten(data))

    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(tags)
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        for path in artifact_paths or []:
            mlflow.log_artifact(path)

    print(f"Logged MLflow run '{run_name}': {len(params)} params, {len(metrics)} metrics")
