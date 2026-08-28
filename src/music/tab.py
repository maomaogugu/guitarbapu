"""Structured guitar tablature data shared by generators and renderers."""

from dataclasses import dataclass, field

from .guitar import Guitar
from .note import Note


@dataclass(frozen=True)
class TabEvent:
    """A note placed on one string and fret."""

    string: int
    fret: int
    start: float = 0.0
    duration: float = 0.0
    note: Note | None = None
    start_beat: float | None = None
    duration_beats: float | None = None
    measure: int = 1
    tie_to_next: bool = False
    technique: str | None = None
    technique_confidence: float | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.string < 1:
            raise ValueError("string must be positive")
        if self.fret < 0:
            raise ValueError("fret must be non-negative")
        if self.start < 0 or self.duration < 0:
            raise ValueError("event timing must be non-negative")
        if self.start_beat is not None and self.start_beat < 0:
            raise ValueError("start_beat must be non-negative")
        if self.duration_beats is not None and self.duration_beats <= 0:
            raise ValueError("duration_beats must be positive")
        if self.measure < 1:
            raise ValueError("measure must be positive")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if (
            self.technique_confidence is not None
            and not 0 <= self.technique_confidence <= 1
        ):
            raise ValueError("technique_confidence must be between 0 and 1")


@dataclass(frozen=True)
class TabRest:
    """A rest included in the TAB timeline."""

    start: float
    duration: float
    start_beat: float | None = None
    duration_beats: float | None = None
    measure: int = 1

    def __post_init__(self) -> None:
        if self.start < 0 or self.duration < 0:
            raise ValueError("rest timing must be non-negative")
        if self.measure < 1:
            raise ValueError("measure must be positive")


@dataclass(frozen=True)
class UnmappedNote:
    """A note retained for diagnosis when no guitar position is available."""

    note: Note
    reason: str
    start_beat: float | None = None
    measure: int = 1

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("reason must not be empty")
        if self.measure < 1:
            raise ValueError("measure must be positive")


@dataclass(frozen=True)
class Tablature:
    """A collection of tab events for a particular guitar definition."""

    guitar: Guitar = field(default_factory=Guitar.standard)
    events: tuple[TabEvent, ...] = field(default_factory=tuple)
    rests: tuple[TabRest, ...] = field(default_factory=tuple)
    unmapped_notes: tuple[UnmappedNote, ...] = field(default_factory=tuple)
    tempo_bpm: float | None = None
    time_signature: tuple[int, int] = (4, 4)
    subdivision: int = 4
    measure_count: int = 1
    diagnostics: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.tempo_bpm is not None and self.tempo_bpm <= 0:
            raise ValueError("tempo_bpm must be positive when provided")
        numerator, denominator = self.time_signature
        if numerator < 1 or denominator < 1:
            raise ValueError("time signature values must be positive")
        if self.subdivision < 1:
            raise ValueError("subdivision must be positive")
        if self.measure_count < 1:
            raise ValueError("measure_count must be positive")

    @property
    def warnings(self) -> tuple[str, ...]:
        unmapped = tuple(
            f"{item.note.name} at {item.note.start:.2f}s: {item.reason}"
            for item in self.unmapped_notes
        )
        return self.diagnostics + unmapped

    @property
    def beats_per_measure(self) -> float:
        numerator, denominator = self.time_signature
        return numerator * 4.0 / denominator


__all__ = ["TabEvent", "TabRest", "Tablature", "UnmappedNote"]
