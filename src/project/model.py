"""Top-level editable transcription project model."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..audio.analyzer import AudioAnalysis
from ..music.tab import Tablature


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

    def __post_init__(self) -> None:
        if self.audio_path is not None and not isinstance(self.audio_path, Path):
            object.__setattr__(self, "audio_path", Path(self.audio_path))
        if not isinstance(self.analysis_parameters, Mapping):
            raise TypeError("analysis_parameters must be a mapping")


__all__ = ["TranscriptionProject"]
