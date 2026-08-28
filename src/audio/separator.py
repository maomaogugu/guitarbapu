"""Backend-independent contracts for optional audio source separation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Callable, Protocol


class SeparationError(RuntimeError):
    """Raised when a separation backend cannot produce the requested stem."""


class SeparationCancelled(SeparationError):
    """Raised when the caller safely cancels a separation job."""


@dataclass(frozen=True)
class SeparationProgress:
    stage: str
    fraction: float | None
    message: str

    def __post_init__(self) -> None:
        if self.fraction is not None and not 0 <= self.fraction <= 1:
            raise ValueError("fraction must be between 0 and 1")


@dataclass(frozen=True)
class Stem:
    name: str
    path: Path
    sample_rate: int
    channels: int
    duration: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("stem name must not be empty")
        if self.sample_rate <= 0 or self.channels <= 0:
            raise ValueError("stem audio metadata must be positive")
        if self.duration < 0:
            raise ValueError("stem duration must be non-negative")


@dataclass(frozen=True)
class SeparationResult:
    source_path: Path
    model_name: str
    device: str
    cache_key: str
    stems: tuple[Stem, ...] = field(default_factory=tuple)
    from_cache: bool = False

    def stem(self, name: str) -> Stem:
        for item in self.stems:
            if item.name == name:
                return item
        raise SeparationError(f"分离结果中没有 {name!r} stem")


ProgressCallback = Callable[[SeparationProgress], None]


class Separator(Protocol):
    def separate(
        self,
        source_path: str | Path,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_event: Event | None = None,
    ) -> SeparationResult:
        """Separate ``source_path`` and return cached file-backed stems."""


__all__ = [
    "ProgressCallback",
    "SeparationCancelled",
    "SeparationError",
    "SeparationProgress",
    "SeparationResult",
    "Separator",
    "Stem",
]
