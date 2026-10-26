"""REST endpoints for the baseline and locally trained forecasting model."""
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.forecast import seasonal_naive_next_day
from app.model import MODEL_PATH, forecast_model
from app.schemas import ForecastRequest, ForecastResponse, MLForecastRequest

logger = logging.getLogger(__name__)
app = FastAPI(title="Greek Grid Load Forecast API", version="0.2.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: is the HTTP process responding?"""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    """Readiness: has a trained model artifact been packaged?"""
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="Trained model is unavailable")
    return {"status": "ready"}


@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest) -> ForecastResponse:
    """Return the seasonal-naive baseline (no trained artifact required)."""
    forecast = seasonal_naive_next_day(request.last_24_hours_mwh)
    return ForecastResponse(
        model_name="seasonal_naive_24h",
        horizon_hours=24,
        forecast_next_24_hours_mwh=forecast,
    )


@app.post("/predict/ml", response_model=ForecastResponse)
def predict_ml(request: MLForecastRequest) -> ForecastResponse:
    """Return 24 hourly forecasts using the trusted locally trained model."""
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="No trained model; run python -m scripts.train")
    predictions = forecast_model(
        request.last_168_hours_mwh,
        request.forecast_start_utc,
        model_path=MODEL_PATH,
    )
    logger.info("Served ML forecast for start_utc=%s", request.forecast_start_utc)
    return ForecastResponse(
        model_name="random_forest_day_ahead",
        horizon_hours=24,
        forecast_next_24_hours_mwh=predictions,
    )
