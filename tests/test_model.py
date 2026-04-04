"""Tests for model loading and prediction logic."""

import numpy as np

from app.model import load_model, predict, predict_batch
from app.features import FEATURE_NAMES


def test_model_loads():
    model = load_model()
    assert model is not None


def test_model_num_features():
    model = load_model()
    assert model.num_feature() == len(FEATURE_NAMES)


def test_predict_returns_probability():
    features = {name: None for name in FEATURE_NAMES}
    result = predict(features)
    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_with_real_values():
    features = {name: None for name in FEATURE_NAMES}
    features["AMT_INCOME_TOTAL"] = 202500.0
    features["AMT_CREDIT"] = 406597.5
    features["EXT_SOURCE_2"] = 0.263
    result = predict(features)
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_batch_consistency():
    """Batch prediction should match individual predictions."""
    features = {name: 0.0 for name in FEATURE_NAMES}
    single = predict(features)
    batch = predict_batch([features])
    assert len(batch) == 1
    assert abs(single["probability"] - batch[0]["probability"]) < 1e-6


def test_predict_deterministic():
    """Same input should give same output."""
    features = {name: None for name in FEATURE_NAMES}
    features["AMT_INCOME_TOTAL"] = 100000.0
    r1 = predict(features)
    r2 = predict(features)
    assert r1["probability"] == r2["probability"]
