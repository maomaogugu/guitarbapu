"""Chord events inferred from simultaneous note groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .note import Note


_PITCH_CLASS_NAMES = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

_QUALITY_TEMPLATES = (
    ("major", (0, 4, 7)),
    ("minor", (0, 3, 7)),
    ("sus2", (0, 2, 7)),
    ("sus4", (0, 5, 7)),
    ("diminished", (0, 3, 6)),
    ("augmented", (0, 4, 8)),
    ("power", (0, 7)),
)

_QUALITY_SUFFIXES = {
    "major": "",
    "minor": "m",
    "power": "5",
    "sus2": "sus2",
    "sus4": "sus4",
    "diminished": "dim",
    "augmented": "aug",
}


def infer_chord_identity(midis: Iterable[int]) -> tuple[int | None, str | None]:
    """Return a conservative root pitch class and chord quality."""

    ordered = tuple(sorted(set(int(value) for value in midis)))
    if len(ordered) < 2:
        return (None, None)
    pitch_classes = {value % 12 for value in ordered}
    preferred_roots = [ordered[0] % 12]
    preferred_roots.extend(
        value for value in range(12) if value not in preferred_roots
    )
    for root in preferred_roots:
        for quality, intervals in _QUALITY_TEMPLATES:
            template = {(root + interval) % 12 for interval in intervals}
            if template == pitch_classes:
                return (root, quality)
    return (None, None)


@dataclass(frozen=True)
class Chord:
    """A simultaneous group of octave-specific MIDI pitches."""

    midis: tuple[int, ...]
    start: float = 0.0
    duration: float = 0.0
    root_pitch_class: int | None = None
    quality: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        normalized = tuple(sorted(set(int(value) for value in self.midis)))
        if len(normalized) < 2:
            raise ValueError("a chord must contain at least two distinct pitches")
        if len(normalized) > 6:
            raise ValueError("a guitar chord cannot contain more than six pitches")
        if any(not 0 <= value <= 127 for value in normalized):
            raise ValueError("chord MIDI pitches must be between 0 and 127")
        if self.start < 0 or self.duration < 0:
            raise ValueError("chord timing must be non-negative")
        if self.root_pitch_class is not None and not 0 <= self.root_pitch_class <= 11:
            raise ValueError("root_pitch_class must be between 0 and 11")
        if self.quality is not None and self.quality not in _QUALITY_SUFFIXES:
            raise ValueError(f"unsupported chord quality: {self.quality}")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "midis", normalized)

    @classmethod
    def from_midis(
        cls,
        midis: Iterable[int],
        *,
        start: float = 0.0,
        duration: float = 0.0,
        confidence: float | None = None,
    ) -> "Chord":
        normalized = tuple(sorted(set(int(value) for value in midis)))
        root, quality = infer_chord_identity(normalized)
        return cls(
            normalized,
            start=start,
            duration=duration,
            root_pitch_class=root,
            quality=quality,
            confidence=confidence,
        )

    @property
    def name(self) -> str:
        if self.root_pitch_class is None or self.quality is None:
            return "/".join(Note(midi).name for midi in self.midis)
        return (
            _PITCH_CLASS_NAMES[self.root_pitch_class]
            + _QUALITY_SUFFIXES[self.quality]
        )

    def notes(self) -> tuple[Note, ...]:
        return tuple(
            Note(
                midi,
                start=self.start,
                duration=self.duration,
                confidence=self.confidence,
            )
            for midi in self.midis
        )


__all__ = ["Chord", "infer_chord_identity"]
