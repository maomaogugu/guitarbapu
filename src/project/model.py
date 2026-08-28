"""Top-level editable transcription project model."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..audio.analyzer import AudioAnalysis
from ..music.tab import Tablature
from .track import TranscriptionTrack


@dataclass(frozen=True)
class TranscriptionProject:
    """Analysis and tablature saved independently from decoded audio samples.

    ``audio_path`` is an optional reference. The waveform is intentionally not
    embedded, so a project remains small and its TAB can still be opened when
    the original audio has moved or is unavailable.
    """

    analysis: AudioAnalysis
    tablature: Tablature
    audio_path: Path | None = None
    analysis_parameters: Mapping[str, Any] = field(default_factory=dict)
    tracks: tuple[TranscriptionTrack, ...] = field(default_factory=tuple)
    active_track_id: str | None = None

    def __post_init__(self) -> None:
        if self.audio_path is not None and not isinstance(self.audio_path, Path):
            object.__setattr__(self, "audio_path", Path(self.audio_path))
        if not isinstance(self.analysis_parameters, Mapping):
            raise TypeError("analysis_parameters must be a mapping")
        track_ids = [track.track_id for track in self.tracks]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("project track IDs must be unique")
        if self.active_track_id is not None and self.active_track_id not in track_ids:
            raise ValueError("active_track_id must identify a project track")

    @property
    def active_track(self) -> TranscriptionTrack | None:
        if not self.tracks:
            return None
        track_id = self.active_track_id or self.tracks[0].track_id
        return next(track for track in self.tracks if track.track_id == track_id)


__all__ = ["TranscriptionProject"]
