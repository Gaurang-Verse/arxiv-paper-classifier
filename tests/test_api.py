import pytest
from fastapi.testclient import TestClient

import arxiv_classifier.api as api_module


class _FakePredictor:
    def __init__(self, **kwargs):
        self.threshold = kwargs.get("threshold", 0.2)

    def predict_proba(self, text):
        return {"cs.AI": 0.9, "cs.LG": 0.6, "stat.ML": 0.1}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_module, "Predictor", _FakePredictor)
    with TestClient(api_module.app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_categories_above_threshold(client):
    response = client.post("/predict", json={
        "title": "Some Paper",
        "abstract": "Some abstract text.",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_categories"] == ["cs.AI", "cs.LG"]
    assert data["probabilities"]["cs.AI"] == 0.9


def test_predict_with_custom_threshold(client):
    response = client.post("/predict", json={
        "title": "Some Paper",
        "abstract": "Some abstract text.",
        "threshold": 0.05,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_categories"] == ["cs.AI", "cs.LG", "stat.ML"]


def test_predict_requires_title_and_abstract(client):
    response = client.post("/predict", json={"title": "Only title"})
    assert response.status_code == 422


def test_predict_rejects_empty_title(client):
    response = client.post("/predict", json={"title": "", "abstract": "text"})
    assert response.status_code == 422