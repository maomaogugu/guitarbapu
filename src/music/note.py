"""Core note events used by transcription and tablature layers."""

from dataclasses import dataclass
import math


_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass(frozen=True)
class Note:
    """A normalized musical note event.

    ``midi`` is used as the interchange pitch representation; timing is in
    seconds until the project introduces a beat/tempo model.
    """

    midi: int
    start: float = 0.0
    duration: float = 0.0
    velocity: int = 100

    @classmethod
    def from_frequency(
        cls,
        frequency: float,
        *,
        start: float = 0.0,
        duration: float = 0.0,
        velocity: int = 100,
    ) -> "Note":
        """Create the nearest equal-tempered MIDI note for ``frequency``."""

        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError("frequency must be a finite positive number")
        midi = int(round(69 + 12 * math.log2(frequency / 440.0)))
        return cls(midi=midi, start=start, duration=duration, velocity=velocity)

    @property
    def frequency(self) -> float:
        """The equal-tempered frequency represented by this note, in hertz."""

        return 440.0 * 2 ** ((self.midi - 69) / 12)

    @property
    def name(self) -> str:
        """Scientific pitch notation name, such as ``A4`` or ``C#3``."""

        octave = self.midi // 12 - 1
        return f"{_NOTE_NAMES[self.midi % 12]}{octave}"

    def __post_init__(self) -> None:
        if not 0 <= self.midi <= 127:
            raise ValueError("midi must be between 0 and 127")
        if self.start < 0 or self.duration < 0:
            raise ValueError("start and duration must be non-negative")
        if not 0 <= self.velocity <= 127:
            raise ValueError("velocity must be between 0 and 127")
