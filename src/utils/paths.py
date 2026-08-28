"""Cross-platform user data, cache, log, and crash-report paths."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Mapping


APP_NAME = "GuitarBapu"


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _home(home: str | Path | None) -> Path:
    return Path.home() if home is None else Path(home)


def user_data_dir(
    *,
    platform_name: str | None = None,
    home: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the per-user application-data directory without creating it."""

    platform_name = platform_name or sys.platform
    environment = _environment(environ)
    home_path = _home(home)
    if platform_name == "darwin":
        return home_path / "Library" / "Application Support" / APP_NAME
    if platform_name.startswith("win"):
        root = environment.get("APPDATA") or environment.get("LOCALAPPDATA")
        if root:
            return Path(root) / APP_NAME
        return home_path / "AppData" / "Roaming" / APP_NAME
    root = environment.get("XDG_DATA_HOME")
    if root:
        return Path(root) / APP_NAME
    return home_path / ".local" / "share" / APP_NAME


def user_cache_dir(
    *,
    platform_name: str | None = None,
    home: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the application cache directory without creating it."""

    platform_name = platform_name or sys.platform
    environment = _environment(environ)
    home_path = _home(home)
    if platform_name == "darwin":
        return home_path / "Library" / "Caches" / APP_NAME
    if platform_name.startswith("win"):
        root = environment.get("LOCALAPPDATA") or environment.get("APPDATA")
        if root:
            return Path(root) / APP_NAME / "cache"
        return home_path / "AppData" / "Local" / APP_NAME / "cache"
    root = environment.get("XDG_CACHE_HOME")
    if root:
        return Path(root) / APP_NAME
    return home_path / ".cache" / APP_NAME


def user_log_dir(
    *,
    platform_name: str | None = None,
    home: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the application log directory without creating it."""

    platform_name = platform_name or sys.platform
    environment = _environment(environ)
    home_path = _home(home)
    if platform_name == "darwin":
        return home_path / "Library" / "Logs" / APP_NAME
    if platform_name.startswith("win"):
        root = environment.get("LOCALAPPDATA") or environment.get("APPDATA")
        if root:
            return Path(root) / APP_NAME / "logs"
        return home_path / "AppData" / "Local" / APP_NAME / "logs"
    root = environment.get("XDG_STATE_HOME")
    if root:
        return Path(root) / APP_NAME / "logs"
    return home_path / ".local" / "state" / APP_NAME / "logs"


def crash_report_dir(**kwargs) -> Path:
    """Return the directory used for human-readable crash reports."""

    return user_data_dir(**kwargs) / "crashes"


__all__ = [
    "APP_NAME",
    "crash_report_dir",
    "user_cache_dir",
    "user_data_dir",
    "user_log_dir",
]
