"""Print a JSON request for /predict/ml from the latest 168 hours in your CSV."""
import argparse
import json
from pathlib import Path
import pandas as pd
from app.data import load_hourly_data

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "sample_hourly_demand.csv")
    args = parser.parse_args()
    data = load_hourly_data(args.csv)
    if len(data) < 168:
        raise ValueError("Need >=168 hours")
    request = {
        "last_168_hours_mwh": data["demand_mwh"].iloc[-168:].tolist(),
        "forecast_start_utc": (data.index[-1] + pd.Timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
    }
    print(json.dumps(request, indent=2))


if __name__ == "__main__":
    main()
