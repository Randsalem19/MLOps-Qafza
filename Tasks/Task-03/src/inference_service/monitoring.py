"""Service metrics (Prometheus) and prediction logging (for drift tracking).

Every prediction is appended to a JSON-lines file with the input, output,
latency, and model version — exactly what you need later to join against the
real delivery outcome and measure drift/decay, per Task 3's monitoring
requirement.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import Lock

from prometheus_client import Counter, Histogram

from .config import get_config, resolve_path

logger = logging.getLogger(__name__)

REQUEST_COUNTER = Counter("inference_requests_total", "Total HTTP requests received", ["route", "status"])
PREDICTION_COUNTER = Counter("predictions_total", "Total predictions made", ["outcome"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Time to score one batch of orders")
VALIDATION_FAILURES = Counter("validation_failures_total", "Requests rejected by input validation")

_LOG_LOCK = Lock()


def log_prediction(order_input: dict, prediction_output: dict, latency_ms: float) -> None:
    config = get_config()
    log_path: Path = resolve_path(config.monitoring.prediction_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "logged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "latency_ms": round(latency_ms, 2),
        "input": order_input,
        "output": prediction_output,
    }
    try:
        with _LOG_LOCK:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
    except OSError as exc:
        logger.warning("Could not write prediction log: %s", exc)


def record_validation_failure() -> None:
    VALIDATION_FAILURES.inc()
