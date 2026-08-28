"""Post-process raw monophonic pitch events into stable musical notes."""

from dataclasses import dataclass
from typing import Iterable

from .note import Note


@dataclass(frozen=True)
class NoteProcessor:
    """Filter pitch glitches and join notes separated by very short gaps."""

    min_note_duration: float = 0.08
    merge_gap: float = 0.08
    blip_max_duration: float = 0.06
    min_onset_segment: float = 0.15

    def __post_init__(self) -> None:
        if self.min_note_duration < 0:
            raise ValueError("min_note_duration must be non-negative")
        if self.merge_gap < 0:
            raise ValueError("merge_gap must be non-negative")
        if self.blip_max_duration < 0:
            raise ValueError("blip_max_duration must be non-negative")
        if self.min_onset_segment < 0:
            raise ValueError("min_onset_segment must be non-negative")

    @staticmethod
    def _end(note: Note) -> float:
        return note.start + note.duration

    @staticmethod
    def _combine(first: Note, second: Note) -> Note:
        """Join two events of the same pitch and preserve useful metadata."""

        duration = (
            max(NoteProcessor._end(first), NoteProcessor._end(second)) - first.start
        )
        weights = (max(first.duration, 1e-9), max(second.duration, 1e-9))

        frequency_hz = None
        if first.frequency_hz is not None and second.frequency_hz is not None:
            frequency_hz = (
                first.frequency_hz * weights[0] + second.frequency_hz * weights[1]
            ) / sum(weights)
        else:
            frequency_hz = first.frequency_hz or second.frequency_hz

        confidence = None
        if first.confidence is not None and second.confidence is not None:
            confidence = (
                first.confidence * weights[0] + second.confidence * weights[1]
            ) / sum(weights)
        else:
            confidence = (
                first.confidence
                if first.confidence is not None
                else second.confidence
            )

        return Note(
            midi=first.midi,
            start=first.start,
            duration=duration,
            velocity=max(first.velocity, second.velocity),
            frequency_hz=frequency_hz,
            confidence=confidence,
        )

    def _remove_blips(self, notes: list[Note]) -> list[Note]:
        """Remove a tiny wrong pitch between two occurrences of one note."""

        if len(notes) < 3:
            return notes

        cleaned: list[Note] = []
        index = 0
        while index < len(notes):
            if 0 < index < len(notes) - 1:
                previous = cleaned[-1] if cleaned else notes[index - 1]
                current = notes[index]
                following = notes[index + 1]
                left_gap = max(0.0, current.start - self._end(previous))
                right_gap = max(0.0, following.start - self._end(current))
                if (
                    current.duration <= self.blip_max_duration
                    and previous.midi == following.midi
                    and left_gap <= self.merge_gap
                    and right_gap <= self.merge_gap
                ):
                    if cleaned:
                        cleaned[-1] = self._combine(previous, following)
                    else:
                        cleaned.append(self._combine(previous, following))
                    index += 2
                    continue
            cleaned.append(notes[index])
            index += 1
        return cleaned

    def _merge_same_pitch(self, notes: Iterable[Note]) -> list[Note]:
        merged: list[Note] = []
        for note in notes:
            if merged:
                previous = merged[-1]
                gap = note.start - self._end(previous)
                if note.midi == previous.midi and gap <= self.merge_gap:
                    merged[-1] = self._combine(previous, note)
                    continue
            merged.append(note)
        return merged

    def _resolve_overlaps(self, notes: Iterable[Note]) -> list[Note]:
        """Trim earlier events when different pitches overlap in mono audio."""

        resolved: list[Note] = []
        for note in notes:
            if resolved and note.start < self._end(resolved[-1]):
                previous = resolved[-1]
                if previous.midi == note.midi:
                    resolved[-1] = self._combine(previous, note)
                    continue
                trimmed_duration = max(0.0, note.start - previous.start)
                if trimmed_duration >= self.min_note_duration:
                    resolved[-1] = Note(
                        midi=previous.midi,
                        start=previous.start,
                        duration=trimmed_duration,
                        velocity=previous.velocity,
                        frequency_hz=previous.frequency_hz,
                        confidence=previous.confidence,
                    )
                else:
                    resolved.pop()
            resolved.append(note)
        return resolved

    def split_at_onsets(
        self, notes: Iterable[Note], onset_times: Iterable[float]
    ) -> tuple[Note, ...]:
        """Split a sustained pitch at reliable re-attack times."""

        onsets = sorted(time for time in onset_times if time >= 0)
        split_notes: list[Note] = []
        for note in notes:
            boundaries = [note.start]
            note_end = self._end(note)
            minimum_segment = max(self.min_note_duration, self.min_onset_segment)
            boundaries.extend(
                onset
                for onset in onsets
                if onset - note.start >= minimum_segment
                and note_end - onset >= minimum_segment
            )
            boundaries.append(note_end)
            for start, end in zip(boundaries, boundaries[1:]):
                split_notes.append(
                    Note(
                        midi=note.midi,
                        start=start,
                        duration=end - start,
                        velocity=note.velocity,
                        frequency_hz=note.frequency_hz,
                        confidence=note.confidence,
                    )
                )
        return tuple(split_notes)

    def process(
        self,
        notes: Iterable[Note],
        *,
        onset_times: Iterable[float] = (),
    ) -> tuple[Note, ...]:
        """Return sorted, de-glitched, merged, and optionally onset-split notes."""

        ordered = sorted(notes, key=lambda note: (note.start, note.midi))
        if not ordered:
            return ()

        cleaned = self._remove_blips(ordered)
        cleaned = self._merge_same_pitch(cleaned)
        cleaned = self._resolve_overlaps(cleaned)
        cleaned = [
            note for note in cleaned if note.duration + 1e-9 >= self.min_note_duration
        ]
        onset_tuple = tuple(onset_times)
        if onset_tuple:
            return self.split_at_onsets(cleaned, onset_tuple)
        return tuple(cleaned)


__all__ = ["NoteProcessor"]
