"""Tests for API endpoints."""

import pytest


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["n_features"] == 795
    assert data["threshold"] == 0.11


def test_predict_single(client, sample_features):
    response = client.post("/predict", json={"features": sample_features})
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0
    assert data["prediction"] in (0, 1)
    assert data["threshold"] == 0.11


def test_predict_batch(client, sample_mlflow_request):
    response = client.post("/predict/batch", json=sample_mlflow_request)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) >= 1
    for pred in data["predictions"]:
        assert 0.0 <= pred["probability"] <= 1.0
        assert pred["prediction"] in (0, 1)


def test_predict_empty_features(client):
    """All features as None — model should still return a score."""
    from app.features import FEATURE_NAMES

    features = {name: None for name in FEATURE_NAMES}
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_missing_body(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_openapi_docs(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "paths" in response.json()
