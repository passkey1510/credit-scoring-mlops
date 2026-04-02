"""Structured JSON logging for predictions."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import LOG_PATH


def _ensure_log_dir():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_prediction(
    request_data: dict,
    result: dict,
    latency_ms: float,
    client_ip: str = "",
):
    """Append a JSON line to the prediction log."""
    _ensure_log_dir()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "client_ip": client_ip,
        "latency_ms": round(latency_ms, 2),
        "n_features_provided": len(request_data),
        "n_nan_features": sum(1 for v in request_data.values() if v is None),
        "probability": result["probability"],
        "prediction": result["prediction"],
        "threshold": result["threshold"],
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


class Timer:
    """Context manager to measure elapsed time in milliseconds."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
