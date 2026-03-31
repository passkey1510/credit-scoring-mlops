"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Single client prediction request — flat dict of features."""
    features: dict[str, float | int | None] = Field(
        ..., description="Feature name → value mapping. Missing features become NaN."
    )


class DataframeSplitRequest(BaseModel):
    """MLflow dataframe_split format."""
    dataframe_split: dict = Field(
        ..., description="MLflow format with 'columns', 'data', and optional 'index'."
    )


class PredictionResponse(BaseModel):
    """Single prediction result."""
    probability: float = Field(..., description="Default probability (0-1)")
    prediction: int = Field(..., description="Binary decision (0=approved, 1=default)")
    threshold: float = Field(..., description="Decision threshold used")


class BatchPredictionResponse(BaseModel):
    """Multiple prediction results."""
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    n_features: int
    threshold: float
