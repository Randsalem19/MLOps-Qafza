# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# Keep the image small and predictable: no compiler toolchain left behind,
# no pip cache, no notebooks, no test files, no dev tooling.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROJECT_ROOT=/app

WORKDIR /app

# System deps only where actually needed to build wheels (scipy/scikit-learn
# occasionally need these); removed in the same layer to keep image size down.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential

# Only what the running service needs — notebooks/, tests/, .git, docs are
# excluded via .dockerignore.
COPY app/ ./app/
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# models/ and logs/ are created empty here; real model artifacts and the
# MLflow sqlite db are mounted at runtime (see docker-compose.yml) so the
# image itself never bakes in a specific model version.
RUN mkdir -p models logs

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
