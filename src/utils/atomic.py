"""Atomic file replacement helpers for user data and exports."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


def atomic_replace(target: Path, writer: Callable[[Path], None]) -> Path:
    """Write to a temporary sibling and atomically replace ``target``."""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        writer(temporary)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


__all__ = ["atomic_replace"]
