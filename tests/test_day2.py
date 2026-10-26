"""Day-2 unit and integration checks."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.data import load_hourly_data
from app.features import FEATURE_NAMES, build_supervised, next_day_features
from app.main import app
from scripts.make_sample_data import generate_sample
from scripts.train import chronological_split

client = TestClient(app)


def test_sample_data_and_chronological_split(tmp_path):
    path = tmp_path / "hourly.csv"
    generate_sample(path, days=80)
    df = load_hourly_data(path)
    x, y = build_supervised(df)
    assert len(x) == len(df) - 168
    train, val, test = chronological_split(x, y)
    assert train[0].index[-1] < val[0].index[0] < val[0].index[-1] < test[0].index[0]
    assert len(val[0]) == 14 * 24 and len(test[0]) == 14 * 24


def test_hour_t_features_ignore_last_23_hours_and_target():
    dates = pd.date_range("2026-01-01", periods=300, freq="h", tz="UTC")
    df = pd.DataFrame({"demand_mwh": np.arange(300, dtype=float) + 1000}, index=dates)
    before, _ = build_supervised(df)
    altered = df.copy()
    # At target t=200, only observations through t-24 may be used.
    altered.iloc[177:201, 0] += 999999
    after, _ = build_supervised(altered)
    pd.testing.assert_series_equal(before.loc[dates[200]], after.loc[dates[200]])


def test_deployed_features_equal_training_features():
    dates = pd.date_range("2026-01-01", periods=250, freq="h", tz="UTC")
    values = np.arange(250, dtype=float) + 1000
    data = pd.DataFrame({"demand_mwh": values}, index=dates)
    x, _ = build_supervised(data)
    # At time t=168 all lags come from the preceding 168 values.
    forecast = next_day_features(values[:168].tolist(), dates[168])
    np.testing.assert_allclose(forecast.iloc[0][FEATURE_NAMES], x.loc[dates[168]][FEATURE_NAMES])
    np.testing.assert_allclose(forecast.iloc[23][FEATURE_NAMES], x.loc[dates[191]][FEATURE_NAMES])


def test_data_gap_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z"],
                  "demand_mwh": [1000, 1020]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="gaps"):
        load_hourly_data(path)


def test_ml_endpoint_requires_trained_model_or_predicts(monkeypatch, tmp_path):
    from app import main
    missing = tmp_path / "missing.joblib"
    monkeypatch.setattr(main, "MODEL_PATH", missing)
    payload = {"last_168_hours_mwh": [1000.0] * 168,
               "forecast_start_utc": "2026-10-27T00:00:00Z"}
    response = client.post("/predict/ml", json=payload)
    assert response.status_code == 503


def test_ml_endpoint_input_validation():
    payload = {"last_168_hours_mwh": [1000.0] * 167,
               "forecast_start_utc": "2026-10-27T00:00:00Z"}
    assert client.post("/predict/ml", json=payload).status_code == 422
    payload["last_168_hours_mwh"].append(1000.)
    payload["forecast_start_utc"] = "2026-10-27T00:15:00Z"
    assert client.post("/predict/ml", json=payload).status_code == 422
