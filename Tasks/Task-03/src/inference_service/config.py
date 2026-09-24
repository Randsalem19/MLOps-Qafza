"""Loads config/config.yaml once and exposes it as a typed, dotted-access object.

Every other module in this service reads paths, feature lists, and thresholds
through this module — nothing is hardcoded elsewhere.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", PROJECT_ROOT / "config" / "config.yaml"))


class _DotDict(dict):
    """dict that also supports attribute access, recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, _DotDict):
            value = _DotDict(value)
            self[item] = value
        return value


@functools.lru_cache(maxsize=1)
def get_config() -> _DotDict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found at {CONFIG_PATH}. "
            "Set CONFIG_PATH or run the service from the project root."
        )
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return _DotDict(raw)


def resolve_path(relative_path: str) -> Path:
    """Every relative path in config.yaml is resolved against PROJECT_ROOT,
    so the service works the same whether it runs on a laptop or in a
    container with a different working directory."""
    path = Path(relative_path)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


@functools.lru_cache(maxsize=1)
def get_expectations() -> dict:
    config = get_config()
    path = resolve_path(config.validation.expectations_path)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
