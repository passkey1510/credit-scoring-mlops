"""LightGBM model loading and prediction."""

import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import MODEL_PATH, THRESHOLD
from .features import FEATURE_NAMES

_model: lgb.Booster | None = None


def load_model() -> lgb.Booster:
    """Load model once at startup."""
    global _model
    if _model is None:
        _model = lgb.Booster(model_file=str(MODEL_PATH))
    return _model


def _build_dataframe(records: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with correct dtypes from feature dicts."""
    # Start with NaN for all features, then fill known values
    df = pd.DataFrame(np.nan, index=range(len(records)), columns=FEATURE_NAMES)
    for i, rec in enumerate(records):
        for col, val in rec.items():
            if col in df.columns and val is not None:
                df.at[i, col] = val
    return df


def predict(data: dict[str, object]) -> dict:
    """Run prediction on a single client record.

    Args:
        data: dict with feature names as keys. Missing features become NaN.

    Returns:
        dict with probability, score (binary), and threshold.
    """
    model = load_model()
    df = _build_dataframe([data])
    probability = float(model.predict(df)[0])
    return {
        "probability": round(probability, 6),
        "prediction": int(probability >= THRESHOLD),
        "threshold": THRESHOLD,
    }


def predict_batch(records: list[dict[str, object]]) -> list[dict]:
    """Run prediction on multiple client records."""
    if not records:
        return []
    model = load_model()
    df = _build_dataframe(records)
    probabilities = model.predict(df)
    return [
        {
            "probability": round(float(p), 6),
            "prediction": int(p >= THRESHOLD),
            "threshold": THRESHOLD,
        }
        for p in probabilities
    ]
