"""FastAPI service entrypoint.

Run locally:
    uvicorn app.main:app --reload --port 8000

Routes:
    GET  /health            liveness/readiness + whether the model loaded
    GET  /model/info        model name, version, threshold, feature count
    POST /predict           score a single order
    POST /predict/batch     score up to 1000 orders, runs the (slower, more
                             thorough) Great Expectations suite instead of
                             the fast pandas checks — see validation.py
    GET  /metrics           Prometheus scrape endpoint
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from inference_service.config import get_config
from inference_service.logging_config import configure_logging
from inference_service.model_loader import get_model
from inference_service.monitoring import REQUEST_COUNTER, VALIDATION_FAILURES
from inference_service.predictor import ValidationError, predict_orders
from inference_service.schemas import (
    BatchOrderInput,
    BatchPredictionResult,
    HealthResponse,
    ModelInfoResponse,
    OrderInput,
    PredictionResult,
)

configure_logging()
logger = logging.getLogger(__name__)
config = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        loaded = get_model()
        logger.info("Model ready at startup (source=%s, version=%s)", loaded.source, loaded.model_version)
    except FileNotFoundError as exc:
        # Deliberately do not crash the process: /health will report the
        # problem clearly instead of the container looping on restart.
        logger.error("Model could not be loaded at startup: %s", exc)
    yield


app = FastAPI(
    title=config.service.api_title,
    version=config.service.api_version,
    description=config.service.description,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    REQUEST_COUNTER.labels(route=request.url.path, status=str(response.status_code)).inc()
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        extra={
            "extra_fields": {
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
            }
        },
    )
    return response


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    VALIDATION_FAILURES.inc()
    return JSONResponse(
        status_code=422, content={"detail": "Input failed validation", "failures": exc.failures}
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        loaded = get_model()
        return HealthResponse(status="ok", model_loaded=True, model_source=loaded.source)
    except Exception as exc:  # model missing, corrupted, etc.
        logger.error("Health check: model unavailable: %s", exc)
        return HealthResponse(status="degraded", model_loaded=False, model_source="unavailable")


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    loaded = get_model()
    return ModelInfoResponse(
        model_name=config.model.name,
        model_version=loaded.model_version,
        decision_threshold=loaded.decision_threshold,
        feature_count=len(loaded.feature_names) if loaded.feature_names else 0,
        primary_metric="PR-AUC",
        loaded_from=loaded.source,
    )


@app.post("/predict", response_model=PredictionResult)
def predict(order: OrderInput) -> PredictionResult:
    result = predict_orders([order.model_dump()], use_great_expectations=False)[0]
    return PredictionResult(**result.__dict__)


@app.post("/predict/batch", response_model=BatchPredictionResult)
def predict_batch(payload: BatchOrderInput) -> BatchPredictionResult:
    orders = [order.model_dump() for order in payload.orders]
    results = predict_orders(orders, use_great_expectations=True)
    return BatchPredictionResult(predictions=[PredictionResult(**r.__dict__) for r in results])


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
