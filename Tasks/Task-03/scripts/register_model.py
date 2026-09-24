"""One-time (or per-retrain) registration script.

Task 2 saved the preprocessor and the model as two separate joblib files
(``05_preprocessor.joblib`` and ``06_logistic_regression.joblib``). This
script combines them into a single ``sklearn.pipeline.Pipeline`` and logs +
registers that pipeline with MLflow, so the running service can load one
versioned artifact from the registry instead of two loose files.

Usage:
    python scripts/register_model.py

Reads paths from config/config.yaml (model.local_fallback.*) and
config/config.yaml (model.mlflow.*) — nothing is hardcoded here either.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inference_service.config import get_config, resolve_path  # noqa: E402


def main() -> None:
    config = get_config()
    fallback = config.model.local_fallback

    preprocessor_path = resolve_path(fallback.preprocessor_path)
    model_path = resolve_path(fallback.model_path)
    feature_names_path = resolve_path(fallback.feature_names_path)
    metadata_path = resolve_path(fallback.metadata_path)

    for path in (preprocessor_path, model_path, feature_names_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Copy the Task-02 artifacts into models/ first "
                "(see README 'Getting the model artifacts')."
            )

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    threshold = metadata.get("selected_threshold", config.model.default_decision_threshold)

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", config.model.mlflow.tracking_uri))
    mlflow.set_experiment(config.service.name)

    with mlflow.start_run(run_name="register-task2-model") as run:
        mlflow.log_param("decision_threshold", threshold)
        mlflow.log_param("best_C", metadata.get("best_C"))
        mlflow.log_param("primary_metric", metadata.get("primary_metric"))
        for record in metadata.get("test_results", []):
            model_label = record.get("model", "model").lower().replace(" ", "_")
            for metric_name, value in record.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{model_label}_{metric_name}", value)
        mlflow.set_tag("feature_names", json.dumps(feature_names))

        result = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name=config.model.name,
        )
        print(f"Logged run {run.info.run_id}")
        print(f"Registered model URI: {result.model_uri}")
        version = result.registered_model_version

    # Aliases are the current MLflow way to mark "this version is production" —
    # the numbered stages (Staging/Production) the task material refers to are
    # deprecated in MLflow 2.9+ and being removed; an alias is the same idea.
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=config.model.name,
        alias=config.model.mlflow.registry_alias,
        version=version,
    )
    print(f"Set alias '{config.model.mlflow.registry_alias}' -> version {version}")


if __name__ == "__main__":
    main()
