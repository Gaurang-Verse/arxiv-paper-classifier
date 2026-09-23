import mlflow


def _flatten(d, parent_key=""):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten(v, new_key))
        elif isinstance(v, list):
            continue  # skip large nested lists (per-label reports, threshold sweeps)
        else:
            items[new_key] = v
    return items


def _split(flat):
    params, metrics = {}, {}
    for k, v in flat.items():
        if isinstance(v, bool):
            params[k] = v
        elif isinstance(v, (int, float)):
            metrics[k] = v
        else:
            params[k] = v
    return params, metrics

def log_run(run_name, data, artifact_paths=None, tags=None,
            tracking_uri="sqlite:///mlflow.db", experiment_name="arxiv-classifier"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    flat = _flatten(data)
    params, metrics = _split(flat)
    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(tags)
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        if artifact_paths:
            for path in artifact_paths:
                mlflow.log_artifact(path)
    print(f"Logged MLflow run '{run_name}': {len(params)} params, {len(metrics)} metrics")