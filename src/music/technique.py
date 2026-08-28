"""Structured guitar-technique detections shared across pipeline layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .note import Note


class GuitarTechnique(str, Enum):
    """Technique labels already supported by the TAB event editor."""

    SLIDE = "slide"
    HAMMER_ON = "hammer-on"
    PULL_OFF = "pull-off"
    BEND = "bend"
    VIBRATO = "vibrato"

    @property
    def display_name(self) -> str:
        return {
            GuitarTechnique.SLIDE: "滑弦",
            GuitarTechnique.HAMMER_ON: "击弦",
            GuitarTechnique.PULL_OFF: "勾弦",
            GuitarTechnique.BEND: "推弦",
            GuitarTechnique.VIBRATO: "颤音",
        }[self]


@dataclass(frozen=True)
class TechniqueDetection:
    """A conservative technique candidate attached to one detected note.

    Transition techniques are attached to the destination ``note`` and retain
    the preceding note in ``related_note``. Bend and vibrato detections only
    use ``note``. The confidence describes the technique classification, not
    the underlying pitch-detection confidence.
    """

    technique: GuitarTechnique
    note: Note
    confidence: float
    related_note: Note | None = None
    pitch_change_semitones: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.technique, GuitarTechnique):
            object.__setattr__(
                self,
                "technique",
                GuitarTechnique(self.technique),
            )
        confidence = float(self.confidence)
        object.__setattr__(self, "confidence", confidence)
        if not 0 <= confidence <= 1:
            raise ValueError("technique confidence must be between 0 and 1")
        if self.pitch_change_semitones is not None:
            pitch_change = float(self.pitch_change_semitones)
            object.__setattr__(self, "pitch_change_semitones", pitch_change)
            if not math.isfinite(pitch_change):
                raise ValueError("pitch_change_semitones must be finite")

    @property
    def note_key(self) -> tuple[int, float, float]:
        """Stable key used to map a detection back to a note/TAB event."""

        return (
            self.note.midi,
            round(self.note.start, 6),
            round(self.note.duration, 6),
        )


__all__ = ["GuitarTechnique", "TechniqueDetection"]
