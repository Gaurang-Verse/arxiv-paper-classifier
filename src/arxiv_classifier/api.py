"""FastAPI application serving the arXiv category classifier.

Endpoints:
    GET  /health   liveness check
    POST /predict  title + abstract -> predicted categories and probabilities
    GET  /metrics  Prometheus metrics (see docs/monitoring.md)

The model is loaded once at startup in the lifespan hook and shared across
requests. Loading a 250MB model per request would dominate latency.
"""

import json
import time
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from arxiv_classifier.inference import Predictor

# Paths are relative to the repo root (or /app in the Docker image).
with open("configs/api.yaml") as f:
    API_CONFIG = yaml.safe_load(f)

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total", "Total number of /predict requests received"
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Time spent computing a single /predict response"
)
PREDICTED_CATEGORY_COUNT = Histogram(
    "predicted_category_count", "Number of categories returned per prediction",
    buckets=[0, 1, 2, 3, 4, 5, 10, 20],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model before the server accepts traffic; release it on shutdown."""
    app.state.predictor = Predictor(
        model_dir=API_CONFIG["model_dir"],
        eda_stats_path=API_CONFIG["eda_stats_path"],
        min_label_frequency=API_CONFIG["min_label_frequency"],
        max_length=API_CONFIG["max_length"],
        threshold=API_CONFIG["threshold"],
    )
    yield
    app.state.predictor = None


app = FastAPI(
    title="arXiv Paper Classifier API",
    description="Multi-label classification of paper title+abstract into arXiv subject categories.",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    title: str = Field(..., min_length=1, description="Paper title")
    abstract: str = Field(..., min_length=1, description="Paper abstract")
    threshold: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Override the default prediction threshold (0-1). Uses the server default if omitted.",
    )


class PredictResponse(BaseModel):
    predicted_categories: list[str]
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    predictor: Predictor = app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    text = f"{request.title} {request.abstract}"
    probabilities = predictor.predict_proba(text)
    threshold = request.threshold if request.threshold is not None else predictor.threshold
    predicted = [label for label, prob in probabilities.items() if prob >= threshold]
    predicted.sort(key=lambda label: probabilities[label], reverse=True)
    latency = time.perf_counter() - start

    # Request text is intentionally not logged: abstracts may be unpublished work.
    PREDICTION_REQUESTS.inc()
    PREDICTION_LATENCY.observe(latency)
    PREDICTED_CATEGORY_COUNT.observe(len(predicted))

    print(json.dumps({
        "event": "prediction",
        "latency_ms": round(latency * 1000, 2),
        "threshold_used": threshold,
        "num_predicted_categories": len(predicted),
        "top_category": predicted[0] if predicted else None,
    }))

    return PredictResponse(predicted_categories=predicted, probabilities=probabilities)