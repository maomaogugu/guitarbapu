"""Tempo, quantization, and rest models shared by analysis and tablature."""

from dataclasses import dataclass, field
from typing import Iterable

from .note import Note


@dataclass(frozen=True)
class TimingInfo:
    """Global timing information inferred from an audio recording."""

    tempo_bpm: float | None = None
    beat_times: tuple[float, ...] = field(default_factory=tuple)
    time_signature: tuple[int, int] | None = None
    subdivision: int = 4

    def __post_init__(self) -> None:
        if self.tempo_bpm is not None and self.tempo_bpm <= 0:
            raise ValueError("tempo_bpm must be positive when provided")
        if self.subdivision < 1:
            raise ValueError("subdivision must be positive")
        if self.time_signature is not None:
            numerator, denominator = self.time_signature
            if numerator < 1 or denominator < 1:
                raise ValueError("time signature values must be positive")

    @property
    def seconds_per_beat(self) -> float | None:
        if self.tempo_bpm is None:
            return None
        return 60.0 / self.tempo_bpm


@dataclass(frozen=True)
class QuantizedNote:
    """A note with editable beat coordinates while preserving its source."""

    source: Note
    note: Note
    start_beat: float | None = None
    duration_beats: float | None = None
    tie_to_next: bool = False


@dataclass(frozen=True)
class Rest:
    """A silent region between detected notes."""

    start: float
    duration: float
    start_beat: float | None = None
    duration_beats: float | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.duration < 0:
            raise ValueError("rest timing must be non-negative")


def _quantize(value: float, step: float) -> float:
    return round(value / step) * step


def quantize_notes(
    notes: Iterable[Note], timing: TimingInfo
) -> tuple[QuantizedNote, ...]:
    """Snap note boundaries to a beat subdivision, preserving source timing."""

    ordered = sorted(notes, key=lambda note: (note.start, note.midi))
    seconds_per_beat = timing.seconds_per_beat
    if seconds_per_beat is None:
        return tuple(QuantizedNote(source=note, note=note) for note in ordered)

    step = 1.0 / timing.subdivision
    quantized: list[QuantizedNote] = []
    for source in ordered:
        start_beat = max(0.0, _quantize(source.start / seconds_per_beat, step))
        end_beat = _quantize(
            (source.start + source.duration) / seconds_per_beat, step
        )
        duration_beats = max(step, end_beat - start_beat)
        tie_to_next = False
        if timing.time_signature is not None:
            beats_per_measure = float(timing.time_signature[0])
            start_measure = int(start_beat // beats_per_measure)
            end_measure = int(
                (start_beat + duration_beats - 1e-9) // beats_per_measure
            )
            tie_to_next = end_measure > start_measure
        note = Note(
            midi=source.midi,
            start=start_beat * seconds_per_beat,
            duration=duration_beats * seconds_per_beat,
            velocity=source.velocity,
            frequency_hz=source.frequency_hz,
            confidence=source.confidence,
        )
        quantized.append(
            QuantizedNote(
                source=source,
                note=note,
                start_beat=start_beat,
                duration_beats=duration_beats,
                tie_to_next=tie_to_next,
            )
        )
    return tuple(quantized)


def find_rests(
    notes: Iterable[Note],
    *,
    total_duration: float,
    timing: TimingInfo,
    min_duration: float = 0.12,
) -> tuple[Rest, ...]:
    """Return leading, internal, and trailing silence regions."""

    if total_duration < 0 or min_duration < 0:
        raise ValueError("durations must be non-negative")
    ordered = sorted(notes, key=lambda note: note.start)
    rests: list[Rest] = []
    cursor = 0.0
    seconds_per_beat = timing.seconds_per_beat
    step = 1.0 / timing.subdivision

    for note in ordered:
        gap_end = min(note.start, total_duration)
        gap = gap_end - cursor
        if gap + 1e-9 >= min_duration:
            start_beat = duration_beats = None
            if seconds_per_beat is not None:
                start_beat = max(0.0, _quantize(cursor / seconds_per_beat, step))
                duration_beats = max(step, _quantize(gap / seconds_per_beat, step))
            rests.append(Rest(cursor, gap, start_beat, duration_beats))
        cursor = max(cursor, note.start + note.duration)

    trailing = total_duration - cursor
    if trailing + 1e-9 >= min_duration:
        start_beat = duration_beats = None
        if seconds_per_beat is not None:
            start_beat = max(0.0, _quantize(cursor / seconds_per_beat, step))
            duration_beats = max(step, _quantize(trailing / seconds_per_beat, step))
        rests.append(Rest(cursor, trailing, start_beat, duration_beats))
    return tuple(rests)


__all__ = [
    "QuantizedNote",
    "Rest",
    "TimingInfo",
    "find_rests",
    "quantize_notes",
]
