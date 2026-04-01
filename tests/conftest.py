"""Shared fixtures for tests."""

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.features import FEATURE_NAMES


def _sanitize_nans(obj):
    """Replace NaN/Infinity with None for JSON serialization."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, list):
        return [_sanitize_nans(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_nans(v) for k, v in obj.items()}
    return obj


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_features() -> dict[str, float | int | None]:
    """Minimal valid feature dict (all NaN except a few)."""
    features = {name: None for name in FEATURE_NAMES}
    features["AMT_INCOME_TOTAL"] = 202500.0
    features["AMT_CREDIT"] = 406597.5
    features["EXT_SOURCE_2"] = 0.263
    features["DAYS_BIRTH"] = -9461
    features["DAYS_EMPLOYED"] = -637.0
    return features


@pytest.fixture
def sample_mlflow_request() -> dict:
    """MLflow dataframe_split request from test fixture file."""
    test_file = Path(__file__).resolve().parent / "serving_test_request.json"
    if test_file.exists():
        data = json.loads(test_file.read_text())
        return _sanitize_nans(data)
    # Fallback: minimal request with 2 rows
    return {
        "dataframe_split": {
            "columns": FEATURE_NAMES,
            "data": [[0.0] * len(FEATURE_NAMES), [1.0] * len(FEATURE_NAMES)],
        }
    }
