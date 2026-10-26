# Demonstration Docker image: builds a synthetic-data model at image-build time.
# For production, train and approve a versioned artifact in a separate pipeline.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /srv/app

# Copy requirements separately to improve layer caching.
COPY requirements-runtime.txt .
RUN python -m pip install --no-cache-dir -r requirements-runtime.txt

COPY app/ ./app/
COPY scripts/ ./scripts/

# Self-contained demo: no external data, API keys, or prebuilt model required.
RUN python -m scripts.make_sample_data \
    && python -m scripts.train

# The app only needs to read its artifacts at runtime.
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
