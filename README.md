# Dockerized ML Inference API

A containerized machine learning inference service built with **Python, scikit-learn, FastAPI, Pydantic, Docker, pytest, and GitHub Actions**.

This project demonstrates how to take a trained forecasting model beyond a notebook: expose it through a validated REST API, package it in a Docker container, test it, and automate quality checks in continuous integration (CI).

> **Data disclaimer:** This repository uses **synthetic hourly electricity-demand data by default**. Its purpose is to demonstrate an ML-serving workflow, **not** to provide validated forecasts of Greece's actual electricity demand. Deployment is **local through Docker Compose**, not public cloud hosting.

## Features

- **Two forecasting approaches:** a seasonal-naive baseline and a trained Random Forest regressor.
- **Day-ahead inference:** 24 hourly predictions from validated historical-demand inputs.
- **REST API:** FastAPI endpoints documented automatically with Swagger UI.
- **Input validation:** Pydantic checks the number and validity of demand observations.
- **Model evaluation:** chronological train/validation/test splits, MAE and RMSE, with comparison against the baseline.
- **Docker deployment:** a self-contained demo image that builds a synthetic-data model.
- **Automated tests and CI:** pytest plus a GitHub Actions workflow for Python tests and Docker smoke tests.
- **Optional experiment tracking:** local MLflow integration.

## Architecture

```text
Client / Swagger UI / HTTP request
                |
                v
       FastAPI + Pydantic
                |
        +-------+--------+
        |                |
        v                v
  Seasonal-naive    Random Forest
    baseline        trained model
        |                |
        +-------+--------+
                |
                v
     24-hour JSON forecast

Docker Compose runs the API on localhost:8000.
```

## Quick start: Docker Desktop

**Prerequisite:** Docker Desktop running in Linux-container mode.

From the project directory, run these commands in **Windows Command Prompt (CMD)** or a terminal:

```bash
docker compose up --build -d
docker compose ps
```

The image creates sample synthetic data, trains the Random Forest during the build, and starts the FastAPI application. No AWS account, API key, or external dataset is required.

Once the container is healthy, open **http://127.0.0.1:8000/docs** to test the API using Swagger UI.

Check the service with:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

When finished:

```bash
docker compose down
```

**Note:** The API binds to `127.0.0.1`, so it is available on your own computer, not publicly on the internet.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Confirms the HTTP service is running. |
| `GET` | `/ready` | Confirms that the trained model file exists. |
| `POST` | `/predict` | Seasonal-naive baseline: accepts 24 hourly demand observations. |
| `POST` | `/predict/ml` | Random Forest: accepts 168 hourly observations and the UTC forecast start time. |
| `GET` | `/docs` | Interactive Swagger UI documentation. |

### Baseline prediction

The baseline predicts each hour using the corresponding hour from the previous day:

\[\hat y_t = y_{t-24}\]

Example request to `POST /predict`:

```json
{
  "last_24_hours_mwh": [
    1000, 1020, 1040, 1050, 1060, 1080,
    1100, 1200, 1300, 1350, 1340, 1280,
    1250, 1220, 1200, 1190, 1210, 1250,
    1320, 1380, 1330, 1250, 1170, 1100
  ]
}
```

### Random Forest prediction

`POST /predict/ml` accepts:

- `last_168_hours_mwh`: exactly 168 positive, hourly demand values, oldest first.
- `forecast_start_utc`: the start of the 24-hour forecast on an exact UTC hour, e.g. `2026-10-27T00:00:00Z`.

To generate a valid example request using the sample dataset **inside the running container**, run:

```bash
docker compose exec -T api python -m scripts.demo_request > request.json
```

Open `request.json`, copy its contents into the **Try it out** form for `/predict/ml` at `/docs`, and click **Execute**. The response includes `model_name`, `horizon_hours: 24`, and 24 predicted hourly demand values.

You can also send the generated request directly from CMD:

```bash
curl -X POST http://127.0.0.1:8000/predict/ml -H "Content-Type: application/json" --data-binary "@request.json"
```

## Model training and evaluation

The Random Forest uses lagged demand, rolling historical averages, and calendar features. The pipeline splits data **chronologically** to avoid train/test leakage. A daily seasonal-naive forecast (`y[t-24]`) serves as the benchmark.

Sample **synthetic-data** held-out test metrics from the included training run:

| Model | MAE (MWh) | RMSE (MWh) |
|---|---:|---:|
| Seasonal naive | 217.92 | 329.52 |
| Random Forest | 61.31 | 77.29 |

These results characterize only the synthetic sample generator. They **must not** be interpreted as real-world forecasting accuracy.

### Run training and tests without Docker

Using Python 3.12 on Windows:

```bat
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m scripts.make_sample_data
.venv\Scripts\python.exe -m scripts.train
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

The training script saves a model in `models/` and an evaluation report in `reports/`. Generated datasets and models do not need to be committed to Git.

For optional local MLflow tracking:

```bat
.venv\Scripts\python.exe -m pip install -r requirements-mlflow.txt
.venv\Scripts\python.exe -m scripts.train --mlflow
```

To use your own hourly data, provide a CSV containing `timestamp,demand_mwh` with consecutive timezone-aware UTC timestamps and positive demand values, then run:

```bat
.venv\Scripts\python.exe -m scripts.train --csv data/greece_hourly_demand.csv
```

Document the source, unit conventions, time zone, licensing, and evaluation procedure before reporting real-data performance.

## Project structure

```text
.
├── app/                   # API, schemas, features, model inference
├── scripts/               # Synthetic data, training, demo request
├── tests/                 # Automated tests
├── .github/workflows/ci.yml
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .gitignore
├── requirements.txt
├── requirements-runtime.txt
├── requirements-mlflow.txt
└── README.md
```

## Continuous integration

The GitHub Actions workflow in `.github/workflows/ci.yml` is configured to run on pushes and pull requests. It checks the Python code, executes tests, trains a synthetic-data model, builds the Docker image, and performs HTTP smoke tests.

A successful CI run confirms those checks passed; **it does not mean the application is publicly hosted**. See the repository's **Actions** tab for the actual workflow status.

## Scope and limitations

This is a **portfolio demonstration of local ML model serving**, not a hardened production forecasting platform:

- Default training/evaluation data is synthetic; real-world performance remains unverified.
- The API has no authentication and should not be exposed to the public internet as-is.
- `/ready` currently checks whether the model artifact exists, not whether a full prediction succeeds.
- The Docker build trains its own demo model for reproducibility; a production deployment should use separately trained, approved, versioned artifacts.
- Dependency ranges are bounded but not fully locked. More rigorous deployments should pin and audit versions.
- Only load trusted `.joblib` model artifacts, since pickle-based deserialization can execute code.

Potential next steps include real-data backtesting, a versioned model registry, stronger readiness checks, authentication, and monitoring.

## Technologies

Python · FastAPI · Pydantic · scikit-learn · pandas · NumPy · MLflow (optional) · pytest · Docker · Docker Compose · GitHub Actions
