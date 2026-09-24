"""Generates a tiny synthetic preprocessor + model with the exact schema the
real Task-02 artifacts have, and writes them to models/.

Used ONLY by CI (see .github/workflows/ci.yml, job compose-smoke-test) so the
docker-compose stack has something to load and can prove the API actually
comes up and answers /predict — without ever committing the real (large,
gitignored) model files to the repo.

This mirrors the fixture logic in tests/conftest.py; if that fixture's schema
ever changes, update both.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
CATEGORICAL_FEATURES = [
    "customer_state",
    "primary_seller_state",
    "primary_category",
    "dominant_payment_type",
]


def main() -> None:
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

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
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=5,
                    sparse_output=True,
                ),
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

    joblib.dump(preprocessor, models_dir / "05_preprocessor.joblib")
    joblib.dump(model, models_dir / "06_logistic_regression.joblib")
    feature_names = preprocessor.get_feature_names_out().tolist()
    (models_dir / "05_feature_names.json").write_text(json.dumps(feature_names), encoding="utf-8")
    (models_dir / "06_results_summary.json").write_text(
        json.dumps(
            {
                "selected_threshold": 0.5,
                "best_C": 1.0,
                "primary_metric": "PR-AUC",
                "test_results": [],
            }
        ),
        encoding="utf-8",
    )
    print(f"Wrote CI fixture model artifacts to {models_dir}")


if __name__ == "__main__":
    main()
