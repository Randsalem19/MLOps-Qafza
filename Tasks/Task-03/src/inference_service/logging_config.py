"""Structured logging setup.

Every module gets its logger with ``logging.getLogger(__name__)`` after
``configure_logging()`` has run once at process startup (app/main.py does this
on import). Logs go to console (human-readable) and to a rotating file
(JSON lines, easy to grep/parse for monitoring).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
from pathlib import Path

from .config import get_config, resolve_path

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # allow callers to attach structured fields via logger.info(..., extra={"extra_fields": {...}})
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            payload.update(extra_fields)
        return json.dumps(payload)


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    config = get_config()
    log_dir: Path = resolve_path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / config.logging.log_file
    level = getattr(logging, str(config.logging.level).upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    root.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    if config.logging.json_format:
        file_handler.setFormatter(JsonFormatter())
    else:
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    root.addHandler(file_handler)

    _CONFIGURED = True
    logging.getLogger(__name__).info(
        "Logging configured", extra={"extra_fields": {"log_file": str(log_file)}}
    )
