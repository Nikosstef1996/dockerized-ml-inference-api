"""Day 3: readiness, caching and trained-model API integration checks."""
import math

from fastapi.testclient import TestClient

from app import main
from app.model import _load_model

client = TestClient(main.app)


def test_ready_requires_model(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "absent.joblib")
    response = client.get("/ready")
    assert response.status_code == 503


def test_health_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "absent.joblib")
    assert client.get("/health").json() == {"status": "ok"}


def test_trained_model_served_via_api(tmp_path, monkeypatch):
    """Use real inference from a fitted artifact, not an endpoint mock."""
    # Train a compact, deterministic RF for this integration test only.
    from joblib import dump
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from app.features import FEATURE_NAMES

    x = np.tile(np.array([1200., 1300., 1100., 1500., 1., 0., 0., 1., 0.]), (40, 1))
    # Use DataFrame to preserve feature names used in production.
    import pandas as pd
    features = pd.DataFrame(x, columns=FEATURE_NAMES)
    target = pd.Series(np.linspace(1000, 1400, 40))
    model = RandomForestRegressor(n_estimators=5, random_state=42).fit(features, target)
    model_file = tmp_path / "rf.joblib"
    dump(model, model_file)
    monkeypatch.setattr(main, "MODEL_PATH", model_file)
    _load_model.cache_clear()
    try:
        assert client.get("/ready").status_code == 200
        payload = {
            "last_168_hours_mwh": [1100.0 + i % 24 * 5 for i in range(168)],
            "forecast_start_utc": "2026-10-27T00:00:00Z",
        }
        response = client.post("/predict/ml", json=payload)
        assert response.status_code == 200, response.text
        output = response.json()
        assert output["model_name"] == "random_forest_day_ahead"
        assert len(output["forecast_next_24_hours_mwh"]) == 24
        assert all(math.isfinite(value) and value >= 0 for value in output["forecast_next_24_hours_mwh"])
    finally:
        _load_model.cache_clear()
