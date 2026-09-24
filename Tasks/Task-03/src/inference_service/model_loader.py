"""Loads the fitted preprocessor + model exactly once per process.

Primary source: MLflow Model Registry (a real deployment loads from here, or
from a remote artifact store — never from someone's local notebooks/ folder).
Fallback: the raw joblib files copied from Task 2's artifacts/, used for local
development when no MLflow server/registry is configured yet.

Either path returns the same ``LoadedModel`` shape, so the rest of the service
does not care which one was used — it only checks ``.source``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from .config import get_config, resolve_path

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    preprocessor: Any
    model: Any
    feature_names: list[str]
    decision_threshold: float
    model_version: str
    source: str  # "mlflow" or "local_fallback"


def _load_from_mlflow() -> LoadedModel | None:
    config = get_config()
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        logger.warning("mlflow is not installed; skipping registry load")
        return None

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", config.model.mlflow.tracking_uri))
    model_name = config.model.name
    alias = config.model.mlflow.registry_alias

    try:
        client = mlflow.tracking.MlflowClient()
        version_info = client.get_model_version_by_alias(model_name, alias)
        pyfunc_uri = f"models:/{model_name}@{alias}"
        loaded = mlflow.sklearn.load_model(pyfunc_uri)  # a scikit-learn Pipeline logged as one artifact
        run = client.get_run(version_info.run_id)
        threshold = float(run.data.params.get("decision_threshold", config.model.default_decision_threshold))
        feature_names = json.loads(run.data.tags.get("feature_names", "[]"))

        # The registered artifact is a single sklearn Pipeline(preprocessor, model);
        # expose both halves so the rest of the service can keep calling
        # preprocessor.transform(...) then model.predict_proba(...) uniformly.
        preprocessor = loaded.named_steps["preprocessor"]
        model = loaded.named_steps["model"]

        logger.info(
            "Loaded model '%s' v%s from MLflow registry (alias=%s)", model_name, version_info.version, alias
        )
        return LoadedModel(
            preprocessor=preprocessor,
            model=model,
            feature_names=feature_names,
            decision_threshold=threshold,
            model_version=str(version_info.version),
            source="mlflow",
        )
    except Exception as exc:  # registry unreachable, model/alias not set yet, etc.
        logger.info("Could not load model from MLflow registry (%s); trying local fallback", exc)
        return None


def _load_from_local_fallback() -> LoadedModel:
    config = get_config()
    fallback = config.model.local_fallback

    preprocessor_path = resolve_path(fallback.preprocessor_path)
    model_path = resolve_path(fallback.model_path)
    feature_names_path = resolve_path(fallback.feature_names_path)
    metadata_path = resolve_path(fallback.metadata_path)

    for path in (preprocessor_path, model_path, feature_names_path):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Local fallback artifact missing: {path}. "
                "Copy it from Tasks/Task-02/artifacts/ (see README 'Getting the model artifacts')."
            )

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    feature_names = json.loads(Path(feature_names_path).read_text(encoding="utf-8"))

    threshold = float(config.model.default_decision_threshold)
    if Path(metadata_path).exists():
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        threshold = float(metadata.get("selected_threshold", threshold))

    logger.info("Loaded model from local fallback artifacts at %s", model_path)
    return LoadedModel(
        preprocessor=preprocessor,
        model=model,
        feature_names=feature_names,
        decision_threshold=threshold,
        model_version="local-fallback",
        source="local_fallback",
    )


_CACHED_MODEL: LoadedModel | None = None


def get_model(force_reload: bool = False) -> LoadedModel:
    global _CACHED_MODEL
    if _CACHED_MODEL is not None and not force_reload:
        return _CACHED_MODEL

    loaded = _load_from_mlflow()
    if loaded is None:
        loaded = _load_from_local_fallback()

    _CACHED_MODEL = loaded
    return loaded
