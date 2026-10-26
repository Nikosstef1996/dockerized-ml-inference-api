"""Read strict UTC hourly electricity-demand observations."""
from pathlib import Path

import numpy as np
import pandas as pd


def load_hourly_data(csv_path: str | Path) -> pd.DataFrame:
    """Return indexed data with no missing/duplicate hours.

    Contract: two columns: timestamp (ISO 8601 with UTC marker or offset)
    and demand_mwh (> 0). For clarity, supply UTC timestamps ending in Z.
    """
    data = pd.read_csv(csv_path)
    required = {"timestamp", "demand_mwh"}
    if not required.issubset(data.columns):
        raise ValueError("CSV requires columns: timestamp,demand_mwh")
    if data.empty:
        raise ValueError("Empty dataset")
    if not data["timestamp"].astype(str).str.contains(r"(?:Z|[+-]\d\d:\d\d)$", regex=True).all():
        raise ValueError("Timestamps must contain UTC Z or an explicit timezone offset")
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    data["demand_mwh"] = pd.to_numeric(data["demand_mwh"], errors="raise")
    values = data["demand_mwh"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Demand must be finite and positive")
    if data["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected")
    data = data.sort_values("timestamp").set_index("timestamp")
    if len(data) > 1 and not data.index.to_series().diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Expected contiguous 1-hour UTC observations; gaps or mixed resolutions detected")
    return data[["demand_mwh"]]
