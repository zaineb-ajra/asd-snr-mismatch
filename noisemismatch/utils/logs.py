"""Logging to stderr and to a file, so that runs inside tmux leave a persistent trace."""

from __future__ import annotations

import logging
from pathlib import Path

LOG_FORMAT: str = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Configures the root logger to write to stderr and optionally to a file.

    Existing root handlers are removed first, so calling this twice does not
    duplicate messages. The log file is opened in append mode, so a resumed run
    keeps the history of the interrupted one.

    Args:
        log_file: File to append log records to; its parent directory is
            created if needed. If None, logs go to stderr only.
        level: Logging level of the root logger.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(level)
