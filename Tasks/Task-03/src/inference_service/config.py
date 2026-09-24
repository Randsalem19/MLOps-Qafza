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


def get_project_root() -> Path:
    """Read fresh on every call (not frozen at import time) so tests can
    point the whole service at a throwaway directory via the PROJECT_ROOT
    env var, and so a long-lived process picks up an env var set before
    get_config() is first called."""
    return Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))


def get_config_path() -> Path:
    return Path(os.environ.get("CONFIG_PATH", get_project_root() / "config" / "config.yaml"))


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


@functools.lru_cache(maxsize=8)
def _load_config_from_path(config_path_str: str) -> _DotDict:
    config_path = Path(config_path_str)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Set CONFIG_PATH or run the service from the project root."
        )
    with open(config_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return _DotDict(raw)


def get_config() -> _DotDict:
    # Cache key is the resolved path itself, so switching CONFIG_PATH (e.g.
    # between tests) transparently loads the right file with no manual
    # cache-clearing required, while a long-running process still only
    # parses each distinct config file once.
    return _load_config_from_path(str(get_config_path()))


def resolve_path(relative_path: str) -> Path:
    """Every relative path in config.yaml is resolved against the current
    project root, so the service works the same whether it runs on a laptop,
    in a container, or in a test's isolated temp directory."""
    path = Path(relative_path)
    return path if path.is_absolute() else (get_project_root() / path)


@functools.lru_cache(maxsize=8)
def _load_expectations_from_path(path_str: str) -> dict:
    with open(path_str, encoding="utf-8") as handle:
        return json.load(handle)


def get_expectations() -> dict:
    config = get_config()
    path = resolve_path(config.validation.expectations_path)
    return _load_expectations_from_path(str(path))
