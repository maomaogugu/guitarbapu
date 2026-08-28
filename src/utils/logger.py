"""Rotating application logs and uncaught-exception crash reports."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import platform
import sys
import threading
import traceback
from typing import Callable

from src import __version__

from .paths import crash_report_dir, user_log_dir


LOGGER_NAME = "guitarbapu"
LOG_FILE_NAME = "guitarbapu.log"
_configured_log_path: Path | None = None


def configure_logging(
    *,
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
) -> Path:
    """Configure one rotating UTF-8 log file and return its path."""

    global _configured_log_path
    directory = Path(log_dir) if log_dir is not None else user_log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / LOG_FILE_NAME
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    resolved_path = path.resolve(strict=False)
    for handler in tuple(logger.handlers):
        if (
            isinstance(handler, RotatingFileHandler)
            and Path(handler.baseFilename) == resolved_path
        ):
            _configured_log_path = path
            return path
        if isinstance(handler, RotatingFileHandler):
            logger.removeHandler(handler)
            handler.close()

    handler = RotatingFileHandler(
        path,
        maxBytes=2 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    _configured_log_path = path
    logger.info(
        "GuitarBapu %s logging started (Python %s, %s)",
        __version__,
        platform.python_version(),
        platform.platform(),
    )
    return path


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the application logger."""

    return logging.getLogger(LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}")


def current_log_path() -> Path | None:
    return _configured_log_path


def write_crash_report(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback,
    *,
    report_dir: str | Path | None = None,
    thread_name: str | None = None,
) -> Path:
    """Write a bounded diagnostic report without environment-variable dumps."""

    directory = Path(report_dir) if report_dir is not None else crash_report_dir()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    target = directory / timestamp.strftime("crash-%Y%m%dT%H%M%S.%fZ.txt")
    trace = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    lines = [
        f"GuitarBapu version: {__version__}",
        f"UTC time: {timestamp.isoformat()}",
        f"Python: {platform.python_version()} ({sys.executable})",
        f"Platform: {platform.platform()}",
        f"Frozen build: {bool(getattr(sys, 'frozen', False))}",
        f"Thread: {thread_name or threading.current_thread().name}",
        f"Log file: {_configured_log_path or 'not configured'}",
        "",
        "Traceback:",
        trace,
    ]
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def install_exception_hooks(
    *,
    report_dir: str | Path | None = None,
    report_callback: Callable[[Path], None] | None = None,
) -> None:
    """Install process and thread hooks that log and persist uncaught errors."""

    logger = get_logger("crash")

    def handle(exc_type, exc_value, exc_traceback, *, thread_name=None) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        try:
            report = write_crash_report(
                exc_type,
                exc_value,
                exc_traceback,
                report_dir=report_dir,
                thread_name=thread_name,
            )
        except Exception:
            logger.exception("Could not write crash report")
            return
        if report_callback is not None:
            try:
                report_callback(report)
            except Exception:
                logger.exception("Crash-report callback failed")

    def system_hook(exc_type, exc_value, exc_traceback) -> None:
        handle(exc_type, exc_value, exc_traceback)

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        handle(
            args.exc_type,
            args.exc_value,
            args.exc_traceback,
            thread_name=args.thread.name if args.thread is not None else None,
        )

    sys.excepthook = system_hook
    threading.excepthook = thread_hook


__all__ = [
    "configure_logging",
    "current_log_path",
    "get_logger",
    "install_exception_hooks",
    "write_crash_report",
]
