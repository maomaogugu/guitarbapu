"""Logical transcription-track model stored inside a project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..audio.analyzer import AudioAnalysis
from ..music.tab import Tablature
from ..music.track import TrackRole


@dataclass(frozen=True)
class TranscriptionTrack:
    """An independently editable logical track sharing a project audio source."""

    track_id: str
    name: str
    role: TrackRole
    analysis: AudioAnalysis
    tablature: Tablature
    source_name: str = "original"
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.track_id.strip():
            raise ValueError("track_id must not be empty")
        if not self.name.strip():
            raise ValueError("track name must not be empty")
        if not isinstance(self.role, TrackRole):
            object.__setattr__(self, "role", TrackRole(self.role))
        if not self.source_name.strip():
            raise ValueError("source_name must not be empty")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("track confidence must be between 0 and 1")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("track metadata must be a mapping")


__all__ = ["TranscriptionTrack"]
