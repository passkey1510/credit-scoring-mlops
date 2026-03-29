"""Configuration for the credit scoring API."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.lgb"
REFERENCE_DATA_PATH = BASE_DIR / "data" / "reference_data.parquet"
LOG_PATH = BASE_DIR / "logs" / "predictions.jsonl"

# Model
THRESHOLD = 0.11

# API
API_HOST = "0.0.0.0"
API_PORT = 8000

# Dashboard
DASHBOARD_PORT = 7860
