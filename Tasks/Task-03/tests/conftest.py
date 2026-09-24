"""Shared fixtures.

We never load the real (large, gitignored) Task-02 artifacts in tests. Instead
we fit a tiny, deterministic preprocessor+model on synthetic data with the
EXACT same columns and dtypes the real pipeline uses, and point the service at
those via PROJECT_ROOT / CONFIG_PATH. This makes tests fast, hermetic, and
independent of whether the student has copied the real model files yet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

NUMERIC_FEATURES = [
    "customer_zip_code_prefix",
    "item_count",
    "product_count",
    "seller_count",
    "price_total",
    "price_mean",
    "freight_total",
    "freight_mean",
    "product_weight_mean",
    "product_length_mean",
    "product_height_mean",
    "product_width_mean",
    "product_photos_mean",
    "seller_lat_mean",
    "seller_lng_mean",
    "seller_state_nunique",
    "payment_count",
    "payment_value_total",
    "payment_installments_max",
    "payment_type_count",
    "customer_lat",
    "customer_lng",
    "customer_seller_distance_km",
]
DERIVED_NUMERIC = [
    "purchase_month",
    "purchase_day_of_week",
    "purchase_hour",
    "purchase_is_weekend",
    "estimated_delivery_days",
]
CATEGORICAL_FEATURES = ["customer_state", "primary_seller_state", "primary_category", "dominant_payment_type"]


def _sample_order(order_id: str = "fixture-order-1", **overrides) -> dict:
    base = {
        "order_id": order_id,
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
    base.update(overrides)
    return base


@pytest.fixture
def sample_order() -> dict:
    return _sample_order()


@pytest.fixture
def make_order():
    return _sample_order


@pytest.fixture(scope="session")
def project_root(tmp_path_factory) -> Path:
    """Builds a throwaway project directory: real config/expectations copied
    from the repo, a tiny fitted model in place of the real one."""
    root = tmp_path_factory.mktemp("task3_project")
    (root / "config").mkdir()
    (root / "models").mkdir()
    (root / "logs").mkdir()

    config_text = (REPO_ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
    config_text = config_text.replace(
        'tracking_uri: "sqlite:///mlflow.db"',
        f'tracking_uri: "sqlite:///{root}/mlflow.db"',
    )
    (root / "config" / "config.yaml").write_text(config_text, encoding="utf-8")
    (root / "config" / "expectations.json").write_text(
        (REPO_ROOT / "config" / "expectations.json").read_text(encoding="utf-8"), encoding="utf-8"
    )

    rng = np.random.default_rng(7)
    n = 150
    frame = pd.DataFrame({c: rng.normal(20, 5, n).clip(min=0) for c in NUMERIC_FEATURES + DERIVED_NUMERIC})
    frame["customer_state"] = rng.choice(["SP", "RJ", "MG"], n)
    frame["primary_seller_state"] = rng.choice(["SP", "RJ", "MG"], n)
    frame["primary_category"] = rng.choice(["bed_bath_table", "electronics", "toys"], n)
    frame["dominant_payment_type"] = rng.choice(["credit_card", "boleto"], n)
    labels = rng.integers(0, 2, n)

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=True),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES + DERIVED_NUMERIC),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=1.0,
    )

    columns = NUMERIC_FEATURES + DERIVED_NUMERIC + CATEGORICAL_FEATURES
    X = preprocessor.fit_transform(frame[columns])
    model = LogisticRegression(class_weight="balanced", max_iter=500).fit(X, labels)

    joblib.dump(preprocessor, root / "models" / "05_preprocessor.joblib")
    joblib.dump(model, root / "models" / "06_logistic_regression.joblib")
    feature_names = preprocessor.get_feature_names_out().tolist()
    (root / "models" / "05_feature_names.json").write_text(json.dumps(feature_names), encoding="utf-8")
    (root / "models" / "06_results_summary.json").write_text(
        json.dumps(
            {"selected_threshold": 0.5, "best_C": 1.0, "primary_metric": "PR-AUC", "test_results": []}
        ),
        encoding="utf-8",
    )
    return root


@pytest.fixture(autouse=True)
def _isolated_project(project_root, monkeypatch):
    """Points the service's config loader at the throwaway project for every
    test, and clears memoized singletons between tests."""
    monkeypatch.setenv("PROJECT_ROOT", str(project_root))
    monkeypatch.setenv("CONFIG_PATH", str(project_root / "config" / "config.yaml"))

    from inference_service import config as config_module
    from inference_service import model_loader as model_loader_module

    config_module.get_config.cache_clear()
    config_module.get_expectations.cache_clear()
    model_loader_module._CACHED_MODEL = None
    yield
    model_loader_module._CACHED_MODEL = None
