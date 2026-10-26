"""Deterministic synthetic hourly load for a first local exercise; NOT actual Greece data."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def generate_sample(path: Path, days: int = 120) -> None:
    if days < 60:
        raise ValueError("Need at least 60 days for meaningful training/validation/test windows")
    rng = np.random.default_rng(2026)
    ts = pd.date_range("2025-01-01", periods=24 * days, freq="h", tz="UTC")
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    daily = 480 * np.sin(2 * np.pi * (hour - 7) / 24)
    evening = 220 * np.cos(4 * np.pi * (hour - 19) / 24)
    weekday = np.where(dow < 5, 340.0, -240.0)
    slow_cycle = 150 * np.sin(2 * np.pi * np.arange(len(ts)) / (24 * 45))
    noise = rng.normal(0, 55, len(ts))
    load = np.maximum(300, 3800 + daily + evening + weekday + slow_cycle + noise)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "demand_mwh": np.round(load, 2)}).to_csv(path, index=False)
    print(f"Generated {len(ts):,} synthetic hourly records -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "sample_hourly_demand.csv")
    parser.add_argument("--days", type=int, default=120)
    args = parser.parse_args()
    generate_sample(args.output, args.days)


if __name__ == "__main__":
    main()
