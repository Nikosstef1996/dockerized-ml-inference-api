# Greek Grid Load Forecast API — Day 3: Docker + GitHub Actions

A **self-contained portfolio demonstration** of a day-ahead forecasting API with a seasonal-naive baseline, a Random Forest, chronological model evaluation, FastAPI, Docker, and continuous integration (CI).

> **Data disclaimer:** The default demo dataset is deterministic, **synthetic** hourly demand. The results do not measure performance on real Greek electricity demand. See the Day 2 section below for the real-data input contract.

## What is new on Day 3?

- A `Dockerfile` packages an API and its model, with a non-root runtime user and `/ready` health check.
- `compose.yaml` starts the Dockerized API locally, bound to `127.0.0.1` (not publicly exposed).
- `.dockerignore` keeps caches, local datasets, model files, secrets, and other artifacts out of Docker build context.
- `.github/workflows/ci.yml` runs tests and checks a self-contained Docker build on pushes and pull requests.
- `/ready` checks **readiness** (the model exists), separately from `/health` (the server is responsive).
- The model is cached per process for faster requests; **restart the service when replacing the file**.
- `tests/test_day3.py` checks readiness and an actual trained-model inference request.

## Windows: run with Docker Desktop (no AWS)

**Prerequisite:** Install Docker Desktop and enable its Linux container engine. Run these commands from the extracted project root in PowerShell.

```powershell
docker --version
docker compose version
docker compose up --build -d
docker compose ps
```

Wait for the container health check, then visit http://127.0.0.1:8000/docs .

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
```

Generate a JSON payload for trained ML inference using the same deterministic sample locally (see 'Run without Docker'), or directly create a 168-value request in PowerShell:

```powershell
$body = @{
  last_168_hours_mwh = @(1..168 | ForEach-Object { 1000.0 })
  forecast_start_utc = '2026-10-27T00:00:00Z'
} | ConvertTo-Json -Depth 4
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict/ml -Method Post -ContentType 'application/json' -Body $body
```

Stop the container with `docker compose down`.

**What's inside the image?** The Dockerfile generates synthetic sample data and trains a model as part of `docker build`. This is intentional for a credential-free demo; in a real production system, training should be handled separately and only approved, versioned models should be deployed.

## Run without Docker (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.make_sample_data
.\.venv\Scripts\python.exe -m scripts.train
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs .

## API endpoints

| Method | URL | Result |
|---|---|---|
| GET | `/health` | HTTP process liveness; always `{'status':'ok'}` when running |
| GET | `/ready` | 200 if trained artifact exists; 503 otherwise |
| POST | `/predict` | Seasonal-naive forecast, requiring 24 historical hourly values |
| POST | `/predict/ml` | Random Forest 24-hour forecast from 168 historical hours + a UTC forecast start |

## GitHub Actions / CI

The workflow triggers on every push and pull request. It performs (1) Python unit/integration tests, (2) synthetic training and local readiness check, and (3) Docker build and live HTTP smoke tests. No AWS credentials, Docker Hub account, or paid hosting required. A green CI badge indicates the checks passed on the repository's last workflow run; it is **not proof of external production deployment**.

**Publishing without overwriting your existing repo**: if `greek-grid-load-forecast` already exists, do not replace its contents blindly. Create a branch, copy the updated `app/`, `scripts/`, `tests/`, `Dockerfile`, `compose.yaml`, `.dockerignore`, `.github/workflows/ci.yml`, `requirements-runtime.txt` and any dependency changes; keep existing documentation and reconcile any conflicts before commit.

Sample git commands after reviewing changes:

```powershell
git checkout -b feature/day3-docker-ci
git status
git add app scripts tests Dockerfile compose.yaml .dockerignore .github/workflows/ci.yml requirements*.txt README.md .gitignore
git commit -m "Add Docker packaging and GitHub Actions CI"
git push -u origin feature/day3-docker-ci
```

Then use the **Actions** tab on GitHub to inspect both jobs and their logs. If a job fails, click that job to see the exact failing command.

## Model training and real-data extension

The project retains all Day 2 code: `scripts/make_sample_data.py`, `scripts/train.py`, optional MLflow (`requirements-mlflow.txt`), and the local model artifacts under `models/`. To switch to actual Greek hourly demand, supply `timestamp,demand_mwh` CSV rows with consecutive, timezone-aware UTC hourly timestamps and positive values, then run `python -m scripts.train --csv data/greece_hourly_demand.csv` locally. Keep large data and model artifacts outside Git and cite their provenance/licensing in your README.

Potential real source: ENTSO-E Transparency Platform https://transparency.entsoe.eu/ .

## Interview discussion questions

1. What's the difference between an image and a running container?
2. Why include a `Dockerfile` if Python and pip are installed locally?
3. What does `docker compose up --build -d` do?
4. Why does CI run tests on every pull request? What's the difference between CI and CD?
5. What is the difference between liveness (`/health`) and readiness (`/ready`)?
6. Why is generating a synthetic model during a Docker build good for a demo, but not ideal for production?
7. Why must you not load untrusted `.joblib` files?

## Known limitations / next improvements

- Synthetic data is used by default; real-world forecasting needs real data, provenance, and rolling-origin validation.
- Versions in requirements use bounded ranges; for stronger reproducibility generate and review a locked dependency file.
- Local dev API has no authentication: never expose it directly to the public internet.
- Readiness currently checks model-file existence, not deserialization and a real prediction; improve it before production.
- Model loads into each process on first inference. Replace artifact by restarting processes; use a versioned deployment strategy for a production model registry.
