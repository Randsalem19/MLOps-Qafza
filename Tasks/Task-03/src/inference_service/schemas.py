"""Pydantic models — the contract for every request and response.

Field names and types mirror exactly the raw columns Notebook 05 (feature
engineering) reads before building purchase_month / estimated_delivery_days /
etc. Do NOT add the derived columns here — the service computes them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OrderInput(BaseModel):
    """One order, in the same shape Notebook 01's join produces (pre
    feature-engineering). This is what a caller sends to /predict."""

    # --- identifiers (echoed back, not used by the model) ---
    order_id: str = Field(..., description="Any string the caller uses to identify the order.")

    # --- datetimes the service derives purchase_* / estimated_delivery_days from ---
    order_purchase_timestamp: datetime
    order_estimated_delivery_date: datetime

    # --- numeric features (see config.yaml -> features.numeric) ---
    customer_zip_code_prefix: float
    item_count: float
    product_count: float
    seller_count: float
    price_total: float
    price_mean: float
    freight_total: float
    freight_mean: float
    product_weight_mean: float
    product_length_mean: float
    product_height_mean: float
    product_width_mean: float
    product_photos_mean: float
    seller_lat_mean: float
    seller_lng_mean: float
    seller_state_nunique: float
    payment_count: float
    payment_value_total: float
    payment_installments_max: float
    payment_type_count: float
    customer_lat: float
    customer_lng: float
    customer_seller_distance_km: float

    # --- categorical features (see config.yaml -> features.categorical) ---
    customer_state: str
    primary_seller_state: str
    primary_category: str
    dominant_payment_type: str

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "order_id": "example-order-0001",
                "order_purchase_timestamp": "2018-05-14T10:32:00",
                "order_estimated_delivery_date": "2018-05-29T00:00:00",
                "customer_zip_code_prefix": 14409,
                "item_count": 1,
                "product_count": 1,
                "seller_count": 1,
                "price_total": 89.9,
                "price_mean": 89.9,
                "freight_total": 16.11,
                "freight_mean": 16.11,
                "product_weight_mean": 700.0,
                "product_length_mean": 25.0,
                "product_height_mean": 13.0,
                "product_width_mean": 20.0,
                "product_photos_mean": 2.0,
                "seller_lat_mean": -23.55,
                "seller_lng_mean": -46.63,
                "seller_state_nunique": 1,
                "payment_count": 1,
                "payment_value_total": 106.01,
                "payment_installments_max": 2,
                "payment_type_count": 1,
                "customer_lat": -21.17,
                "customer_lng": -47.81,
                "customer_seller_distance_km": 248.5,
                "customer_state": "SP",
                "primary_seller_state": "SP",
                "primary_category": "bed_bath_table",
                "dominant_payment_type": "credit_card",
            }
        },
    )


class BatchOrderInput(BaseModel):
    orders: list[OrderInput] = Field(..., min_length=1, max_length=1000)


class PredictionResult(BaseModel):
    order_id: str
    prediction: Literal["late", "on_time"]
    probability_late: float
    decision_threshold: float
    model_name: str
    model_version: str


class BatchPredictionResult(BaseModel):
    predictions: list[PredictionResult]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_source: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    decision_threshold: float
    feature_count: int
    primary_metric: str
    loaded_from: str
