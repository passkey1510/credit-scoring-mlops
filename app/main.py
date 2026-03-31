"""FastAPI credit scoring API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from .config import THRESHOLD
from .features import FEATURE_NAMES
from .logging_config import Timer, log_prediction
from .model import load_model, predict, predict_batch
from .schemas import (
    BatchPredictionResponse,
    DataframeSplitRequest,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup."""
    load_model()
    yield


app = FastAPI(
    title="Credit Scoring API",
    description="LightGBM credit default prediction — MLOps Project 8",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    model = load_model()
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        n_features=len(FEATURE_NAMES),
        threshold=THRESHOLD,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_single(request: PredictionRequest, raw_request: Request):
    """Predict default probability for a single client."""
    with Timer() as t:
        result = predict(request.features)
    log_prediction(
        request_data=request.features,
        result=result,
        latency_ms=t.elapsed_ms,
        client_ip=raw_request.client.host if raw_request.client else "",
    )
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_mlflow(request: DataframeSplitRequest, raw_request: Request):
    """Predict using MLflow dataframe_split format (batch)."""
    ds = request.dataframe_split
    columns = ds.get("columns", FEATURE_NAMES)
    data_rows = ds.get("data", [])

    records = [dict(zip(columns, row)) for row in data_rows]

    with Timer() as t:
        results = predict_batch(records)

    for i, rec in enumerate(records):
        log_prediction(
            request_data=rec,
            result=results[i],
            latency_ms=t.elapsed_ms / len(records),
            client_ip=raw_request.client.host if raw_request.client else "",
        )

    return BatchPredictionResponse(
        predictions=[PredictionResponse(**r) for r in results]
    )
