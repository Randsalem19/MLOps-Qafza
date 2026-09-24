"""Orchestrates one prediction request end to end.

validate -> build features -> transform (fitted preprocessor, never re-fit)
-> predict_proba -> apply the selected threshold -> log the outcome.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd

from . import features as feature_module
from .model_loader import LoadedModel, get_model
from .monitoring import PREDICTION_COUNTER, PREDICTION_LATENCY, log_prediction
from .validation import ValidationResult, validate_fast, validate_with_great_expectations

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    def __init__(self, failures: list[str]):
        self.failures = failures
        super().__init__("; ".join(failures))


@dataclass
class Prediction:
    order_id: str
    prediction: str
    probability_late: float
    decision_threshold: float
    model_name: str
    model_version: str


def _validate_or_raise(frame: pd.DataFrame, use_great_expectations: bool) -> None:
    result: ValidationResult = (
        validate_with_great_expectations(frame) if use_great_expectations else validate_fast(frame)
    )
    if not result.is_valid:
        raise ValidationError(result.failures)


def predict_orders(orders: list[dict], use_great_expectations: bool = False) -> list[Prediction]:
    start = time.perf_counter()
    order_ids = [order["order_id"] for order in orders]

    raw_frame = feature_module.orders_to_dataframe(orders)
    _validate_or_raise(raw_frame, use_great_expectations)

    engineered = feature_module.build_feature_frame(raw_frame)
    model_frame = feature_module.select_model_columns(engineered)

    loaded: LoadedModel = get_model()
    transformed = loaded.preprocessor.transform(model_frame)
    probabilities = loaded.model.predict_proba(transformed)[:, 1]

    results: list[Prediction] = []
    for order_id, probability in zip(order_ids, probabilities, strict=True):
        label = "late" if probability >= loaded.decision_threshold else "on_time"
        results.append(
            Prediction(
                order_id=order_id,
                prediction=label,
                probability_late=float(probability),
                decision_threshold=loaded.decision_threshold,
                model_name="order-delivery-classifier",
                model_version=loaded.model_version,
            )
        )

    latency_ms = (time.perf_counter() - start) * 1000
    PREDICTION_LATENCY.observe(latency_ms / 1000)
    PREDICTION_COUNTER.labels(outcome="success").inc(len(results))
    for order, result in zip(orders, results, strict=True):
        log_prediction(order, result.__dict__, latency_ms)

    logger.info(
        "Scored %d order(s) in %.1fms",
        len(results),
        latency_ms,
        extra={"extra_fields": {"latency_ms": latency_ms, "count": len(results)}},
    )
    return results
