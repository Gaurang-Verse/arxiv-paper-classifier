"""FastAPI application serving the arXiv category classifier.

The model is loaded once at startup (via the lifespan context manager) and
reused across requests -- loading it per-request would be far too slow and
wasteful, so this is a deliberate choice, not an accident.
"""

from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from arxiv_classifier.inference import Predictor

with open("configs/api.yaml") as f:
    API_CONFIG = yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
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


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    predictor: Predictor = app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = f"{request.title} {request.abstract}"
    probabilities = predictor.predict_proba(text)
    threshold = request.threshold if request.threshold is not None else predictor.threshold
    predicted = [label for label, prob in probabilities.items() if prob >= threshold]
    predicted.sort(key=lambda label: probabilities[label], reverse=True)

    return PredictResponse(predicted_categories=predicted, probabilities=probabilities)