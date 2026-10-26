"""Day-ahead features using only information available before forecast origin."""
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "lag_24", "lag_48", "lag_168", "rolling_24_to_lag_24",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "weekend",
]


def calendar_features(dt: pd.Timestamp) -> dict[str, float]:
    hour = dt.hour
    weekday = dt.dayofweek
    return {
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "weekday_sin": float(np.sin(2 * np.pi * weekday / 7)),
        "weekday_cos": float(np.cos(2 * np.pi * weekday / 7)),
        "weekend": float(weekday >= 5),
    }


def build_supervised(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Predict hour t from demand through t-24 at the latest.

    This supports a complete forecast day issued at time t0: all 24 target
    hours t0..t0+23 have their lag-24 and older observations available at t0.
    """
    y = data["demand_mwh"]
    feat = pd.DataFrame(index=data.index)
    feat["lag_24"] = y.shift(24)
    feat["lag_48"] = y.shift(48)
    feat["lag_168"] = y.shift(168)
    feat["rolling_24_to_lag_24"] = y.shift(24).rolling(24).mean()
    hour = data.index.hour.to_numpy()
    weekday = data.index.dayofweek.to_numpy()
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feat["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    feat["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    feat["weekend"] = (weekday >= 5).astype(float)
    valid = feat.notna().all(axis=1)
    return feat.loc[valid, FEATURE_NAMES], y.loc[valid]


def next_day_features(last_168_hours_mwh: list[float], forecast_start_utc: pd.Timestamp) -> pd.DataFrame:
    """Twenty forecast-hour feature rows from the last 168 observed hours."""
    history = np.asarray(last_168_hours_mwh, dtype=float)
    if len(history) != 168 or not np.isfinite(history).all() or (history <= 0).any():
        raise ValueError("Exactly 168 positive, finite hourly values are required")
    start = pd.Timestamp(forecast_start_utc)
    if start.tz is None:
        raise ValueError("forecast_start_utc must specify a timezone")
    start = start.tz_convert("UTC")
    if start.minute or start.second or start.microsecond or start.nanosecond:
        raise ValueError("forecast_start_utc must be aligned to the exact hour")
    rows = []
    for hour in range(24):
        lag24_ix = 168 - 24 + hour
        when = start + pd.Timedelta(hours=hour)
        row = {
            "lag_24": history[lag24_ix],
            "lag_48": history[168 - 48 + hour],
            "lag_168": history[hour],
            "rolling_24_to_lag_24": float(history[lag24_ix - 23:lag24_ix + 1].mean()),
            **calendar_features(when),
        }
        rows.append(row)
    return pd.DataFrame(rows, index=pd.date_range(start=start, periods=24, freq="h"))[FEATURE_NAMES]
