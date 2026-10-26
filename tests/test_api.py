"""HTTP and validation tests for the forecasting API."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_seasonal_naive_forecast() -> None:
    history = [1000.0 + hour * 10 for hour in range(24)]
    response = client.post("/predict", json={"last_24_hours_mwh": history})
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "seasonal_naive_24h"
    assert body["horizon_hours"] == 24
    assert body["forecast_next_24_hours_mwh"] == history


def test_predict_rejects_short_history() -> None:
    response = client.post("/predict", json={"last_24_hours_mwh": [1000.0] * 23})
    assert response.status_code == 422


def test_predict_rejects_negative_demand() -> None:
    response = client.post("/predict", json={"last_24_hours_mwh": [-1.0] + [1000.0] * 23})
    assert response.status_code == 422
