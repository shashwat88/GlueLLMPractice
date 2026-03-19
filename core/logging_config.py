"""Project logging configuration (GlueLLM-inspired).

This project is a CLI-focused assessment. We keep user-facing output on stdout
via `print()`, but provide optional structured logging for diagnostics.

Logging is environment-configurable and supports:
- Rotating file logs (enabled by default)
- Optional colored console logs (disabled by default)
- Optional JSON log formatting
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import colorlog
from pythonjsonlogger import json

DEFAULT_LOG_FILE_NAME = "gluellm_practice.log"


def setup_logging(
    *,
    log_level: str = "INFO",
    log_file_level: str = "DEBUG",
    log_dir: Path | str | None = None,
    log_file_name: str = DEFAULT_LOG_FILE_NAME,
    log_json_format: bool = False,
    log_max_bytes: int = 10 * 1024 * 1024,
    log_backup_count: int = 5,
    console_output: bool = False,
    force: bool = False,
) -> None:
    """Configure project logging handlers and formatting.

    This function is safe to call multiple times; by default it will not
    reconfigure if handlers are already present, unless `force=True`.

    Environment variables:
        PROJECT_DISABLE_LOGGING: Set to true/1/yes to disable setup.
        PROJECT_LOG_LEVEL: Console log level (DEBUG/INFO/WARNING/ERROR/CRITICAL).
        PROJECT_LOG_FILE_LEVEL: File log level (typically DEBUG).
        PROJECT_LOG_DIR: Directory for log files.
        PROJECT_LOG_FILE_NAME: Log file name.
        PROJECT_LOG_JSON_FORMAT: true/1/yes enables JSON formatting.
        PROJECT_LOG_MAX_BYTES: Max file size before rotation.
        PROJECT_LOG_BACKUP_COUNT: Number of rotated backups.
        PROJECT_LOG_CONSOLE_OUTPUT: true/1/yes enables console logging.

    Args:
        log_level: Console log level.
        log_file_level: File log level.
        log_dir: Directory for log files. Defaults to `<project_root>/logs`.
        log_file_name: Log file name.
        log_json_format: Enable JSON formatting for file logs.
        log_max_bytes: Max size for rotating file handler.
        log_backup_count: Number of backups to keep.
        console_output: Enable console logging.
        force: If True, remove existing handlers and reconfigure.
    """
    if os.getenv("PROJECT_DISABLE_LOGGING", "false").lower() in ("true", "1", "yes"):
        return

    project_logger = logging.getLogger("gluellm_practice")
    if project_logger.handlers and not force:
        return

    log_level = os.getenv("PROJECT_LOG_LEVEL", log_level).upper()
    log_file_level = os.getenv("PROJECT_LOG_FILE_LEVEL", log_file_level).upper()
    log_dir_env = os.getenv("PROJECT_LOG_DIR")
    if log_dir_env:
        log_dir = log_dir_env
    log_file_name = os.getenv("PROJECT_LOG_FILE_NAME", log_file_name)
    log_json_format = os.getenv("PROJECT_LOG_JSON_FORMAT", str(log_json_format)).lower() in (
        "true",
        "1",
        "yes",
    )
    console_output = os.getenv("PROJECT_LOG_CONSOLE_OUTPUT", str(console_output)).lower() in (
        "true",
        "1",
        "yes",
    )

    try:
        log_max_bytes = int(os.getenv("PROJECT_LOG_MAX_BYTES", str(log_max_bytes)))
    except ValueError:
        log_max_bytes = 10 * 1024 * 1024
    try:
        log_backup_count = int(os.getenv("PROJECT_LOG_BACKUP_COUNT", str(log_backup_count)))
    except ValueError:
        log_backup_count = 5

    if log_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        log_dir_path = project_root / "logs"
    else:
        log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    numeric_console_level = getattr(logging, log_level, logging.INFO)
    numeric_file_level = getattr(logging, log_file_level, logging.DEBUG)

    project_logger.setLevel(logging.DEBUG)
    project_logger.propagate = False

    if force:
        project_logger.handlers.clear()

    if console_output:
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_console_level)
        console_handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                reset=True,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
        )
        project_logger.addHandler(console_handler)

    log_file_path = log_dir_path / log_file_name
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=log_max_bytes,
        backupCount=log_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_file_level)
    if log_json_format:
        file_handler.setFormatter(
            json.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    else:
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    project_logger.addHandler(file_handler)

    # Record the configuration to the project logger so it lands in the file.
    project_logger.info(
        "Logging configured: console=%s file=%s path=%s json=%s",
        log_level if console_output else "disabled",
        log_file_level,
        str(log_file_path),
        log_json_format,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger with project logging configured."""
    setup_logging()
    # Return a child logger so records propagate to `gluellm_practice` handlers.
    return logging.getLogger(f"gluellm_practice.{name}")


def _truncate_for_log(text: str, max_chars: int = 2000) -> str:
    """Truncate string for logging to avoid huge payloads."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... (truncated, total {len(text)} chars)"


def log_llm_request_response(request: str, response: str, *, max_chars: int = 2000) -> None:
    """Log LLM request and response to the project log file (truncated if needed)."""
    log = get_logger("llm")
    log.info("LLM request: %s", _truncate_for_log(request, max_chars))
    log.info("LLM response: %s", _truncate_for_log(response, max_chars))
