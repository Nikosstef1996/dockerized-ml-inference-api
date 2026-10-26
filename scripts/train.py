"""Train and evaluate a day-ahead forecast without random-split leakage."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.data import load_hourly_data
from app.features import FEATURE_NAMES, build_supervised

ROOT = Path(__file__).resolve().parent.parent


def scores(y_true: pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def chronological_split(x: pd.DataFrame, y: pd.Series, holdout_days: int = 14):
    """Never shuffle time series. Last 14 days test, prior 14 validation."""
    n_holdout = 24 * holdout_days
    if len(x) < n_holdout * 2 + 24 * 28:
        raise ValueError("Need >= 28 training days plus validation/test holdouts, after lagging")
    train_end = len(x) - 2 * n_holdout
    val_end = len(x) - n_holdout
    return (
        (x.iloc[:train_end], y.iloc[:train_end]),
        (x.iloc[train_end:val_end], y.iloc[train_end:val_end]),
        (x.iloc[val_end:], y.iloc[val_end:]),
    )


def train(csv_path: Path, use_mlflow: bool = False) -> dict:
    data = load_hourly_data(csv_path)
    x, y = build_supervised(data)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = chronological_split(x, y)

    model = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model.fit(x_train, y_train)
    val_rf = scores(y_val, model.predict(x_val))
    val_baseline = scores(y_val, x_val["lag_24"])
    test_rf = scores(y_test, model.predict(x_test))
    test_baseline = scores(y_test, x_test["lag_24"])

    # Do not pick a model using test metrics! Validation selects the winner.
    selected = "random_forest" if val_rf["mae"] < val_baseline["mae"] else "seasonal_naive"
    result = {
        "dataset": csv_path.name,
        "note": "Sample CSV is synthetic; not actual Greek demand measurements" if csv_path.name.startswith("sample_") else "User-supplied dataset",
        "features": FEATURE_NAMES,
        "forecast_setup": "For hour t, newest target-dependent feature comes from t-24",
        "train_rows": len(x_train),
        "validation_rows": len(x_val),
        "test_rows": len(x_test),
        "training_window_utc": [str(x_train.index.min()), str(x_train.index.max())],
        "validation_window_utc": [str(x_val.index.min()), str(x_val.index.max())],
        "test_window_utc": [str(x_test.index.min()), str(x_test.index.max())],
        "validation": {"seasonal_naive": val_baseline, "random_forest": val_rf},
        "test": {"seasonal_naive": test_baseline, "random_forest": test_rf},
        "validation_selected_model": selected,
    }

    model_path = ROOT / "models" / "day_ahead_rf.joblib"
    report_path = ROOT / "reports" / "metrics.json"
    pred_path = ROOT / "reports" / "test_predictions.csv"
    model_path.parent.mkdir(exist_ok=True)
    report_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame({
        "timestamp": x_test.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "observed_mwh": y_test.to_numpy(),
        "baseline_mwh": x_test["lag_24"].to_numpy(),
        "rf_mwh": model.predict(x_test),
    }).to_csv(pred_path, index=False)

    if use_mlflow:
        try:
            import mlflow
        except ImportError as exc:
            raise SystemExit("MLflow not installed: python -m pip install -r requirements-mlflow.txt") from exc
        mlflow.set_tracking_uri((ROOT / "mlruns").resolve().as_uri())
        mlflow.set_experiment("greek-grid-load-forecast")
        with mlflow.start_run(run_name="day-ahead-random-forest"):
            mlflow.log_params({"n_estimators": 100, "min_samples_leaf": 2,
                               "seed": 42, "train_rows": len(x_train)})
            mlflow.log_metrics({
                "validation_rf_mae": val_rf["mae"],
                "validation_baseline_mae": val_baseline["mae"],
                "test_rf_mae": test_rf["mae"],
                "test_baseline_mae": test_baseline["mae"],
                "test_rf_rmse": test_rf["rmse"],
            })
            mlflow.log_artifact(str(report_path), artifact_path="reports")
            mlflow.log_artifact(str(pred_path), artifact_path="reports")
            mlflow.log_artifact(str(model_path), artifact_path="model")

    print(json.dumps({"validation": result["validation"],
                      "test": result["test"], "selected_on_validation": selected}, indent=2))
    print(f"Saved trained model: {model_path}")
    print(f"Saved evaluation report: {report_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "sample_hourly_demand.csv")
    parser.add_argument("--mlflow", action="store_true", help="Log run to local MLflow")
    args = parser.parse_args()
    train(args.csv, args.mlflow)


if __name__ == "__main__":
    main()
