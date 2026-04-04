"""Tests for input validation and edge cases."""

from app.features import FEATURE_NAMES


def test_partial_features(client):
    """Only a subset of features provided — rest become NaN."""
    features = {
        "AMT_INCOME_TOTAL": 100000.0,
        "AMT_CREDIT": 500000.0,
    }
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0


def test_extra_features_ignored(client):
    """Extra features not in the model should be silently ignored."""
    features = {name: 0.0 for name in FEATURE_NAMES[:10]}
    features["UNKNOWN_FEATURE_XYZ"] = 999.0
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200


def test_nan_values_explicit(client):
    """Explicit None values should be handled as NaN."""
    features = {FEATURE_NAMES[0]: None, FEATURE_NAMES[1]: None}
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200


def test_batch_empty_data(client):
    """Empty data array should return empty predictions."""
    response = client.post(
        "/predict/batch",
        json={"dataframe_split": {"columns": FEATURE_NAMES, "data": []}},
    )
    assert response.status_code == 200
    assert response.json()["predictions"] == []


def test_batch_single_row(client):
    """Single row in batch format."""
    row = [0.0] * len(FEATURE_NAMES)
    response = client.post(
        "/predict/batch",
        json={"dataframe_split": {"columns": FEATURE_NAMES, "data": [row]}},
    )
    assert response.status_code == 200
    preds = response.json()["predictions"]
    assert len(preds) == 1
    assert 0.0 <= preds[0]["probability"] <= 1.0


def test_feature_count():
    """Verify we have exactly 795 features."""
    assert len(FEATURE_NAMES) == 795
    assert len(set(FEATURE_NAMES)) == 795  # no duplicates
