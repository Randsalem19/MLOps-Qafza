"""Feature construction for inference.

This MUST stay in lockstep with Tasks/Task-02/notebooks/05_feature_engineering.ipynb.
If that notebook's ``build_feature_frame`` ever changes, mirror the change here
— the whole point of Task 3 is that the same input produces the same output
as the notebook. See tests/test_features.py for the parity check.
"""

from __future__ import annotations

import pandas as pd

from .config import get_config


def orders_to_dataframe(orders: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(orders)


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds the five datetime-derived columns Notebook 05 computes. Identical
    logic, copied 1:1 from the notebook."""
    out = frame.copy()
    purchase = pd.to_datetime(out["order_purchase_timestamp"], errors="coerce")
    estimated = pd.to_datetime(out["order_estimated_delivery_date"], errors="coerce")
    out["purchase_month"] = purchase.dt.month
    out["purchase_day_of_week"] = purchase.dt.dayofweek
    out["purchase_hour"] = purchase.dt.hour
    out["purchase_is_weekend"] = purchase.dt.dayofweek.isin([5, 6]).astype("int8")
    out["estimated_delivery_days"] = (estimated - purchase).dt.total_seconds() / 86400
    return out


def select_model_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Returns exactly the columns the fitted ColumnTransformer expects, in
    the same order used at training time (numeric_features + categorical_features)."""
    config = get_config()
    numeric = list(config.features.numeric) + list(config.features.derived_numeric)
    categorical = list(config.features.categorical)
    columns = numeric + categorical
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")
    return frame[columns]


def prepare_features(orders: list[dict]) -> pd.DataFrame:
    """orders -> raw DataFrame -> engineered DataFrame -> model-ready columns."""
    raw = orders_to_dataframe(orders)
    engineered = build_feature_frame(raw)
    return select_model_columns(engineered)
