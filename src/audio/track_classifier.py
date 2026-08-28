"""Conservative logical lead/rhythm classification for detected events."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from ..music.chord import Chord
from ..music.note import Note
from ..music.timing import find_rests, quantize_notes
from ..music.track import TrackRole
from .analyzer import AudioAnalysis
from .rhythm import RhythmAnalysis


@dataclass(frozen=True)
class TrackCandidate:
    """A classified logical event track before TAB generation."""

    role: TrackRole
    name: str
    analysis: AudioAnalysis
    confidence: float


class TrackClassifier:
    """Split events by texture without claiming independent audio stems."""

    def __init__(self, *, confidence_threshold: float = 0.35) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.confidence_threshold = float(confidence_threshold)

    @staticmethod
    def _event_key(note: Note) -> tuple[float, float]:
        return (round(note.start, 6), round(note.duration, 6))

    @staticmethod
    def _note_key(note: Note) -> tuple[int, float, float]:
        return (note.midi, round(note.start, 6), round(note.duration, 6))

    @staticmethod
    def _group_confidence(notes: Iterable[Note]) -> float:
        values = [note.confidence for note in notes if note.confidence is not None]
        return sum(values) / len(values) if values else 0.6

    def _role_for_group(
        self,
        notes: tuple[Note, ...],
        *,
        chord_keys: set[tuple[float, float]],
    ) -> tuple[TrackRole, float]:
        confidence = self._group_confidence(notes)
        if confidence < self.confidence_threshold:
            return (TrackRole.UNKNOWN, confidence)
        if len(notes) >= 2 or self._event_key(notes[0]) in chord_keys:
            return (TrackRole.RHYTHM, confidence)
        return (TrackRole.LEAD, confidence)

    def _filtered_analysis(
        self,
        source: AudioAnalysis,
        notes: tuple[Note, ...],
        *,
        role: TrackRole,
    ) -> AudioAnalysis:
        note_keys = {self._note_key(note) for note in notes}
        chords = tuple(
            chord
            for chord in source.chords
            if all(
                (midi, round(chord.start, 6), round(chord.duration, 6))
                in note_keys
                for midi in chord.midis
            )
        )
        rhythm = None
        if source.rhythm is not None:
            timing = source.rhythm.timing
            quantized = tuple(
                item
                for item in source.rhythm.quantized_notes
                if self._note_key(item.source) in note_keys
            )
            if len(quantized) != len(notes):
                quantized = quantize_notes(notes, timing)
            rhythm = RhythmAnalysis(
                timing=timing,
                onset_times=tuple(sorted({note.start for note in notes})),
                quantized_notes=quantized,
                rests=find_rests(
                    notes,
                    total_duration=source.duration_seconds,
                    timing=timing,
                ),
            )
        features = dict(source.features)
        features.update(
            {
                "logical_track": True,
                "track_role": role.value,
                "track_classifier": "structural-v1",
            }
        )
        return AudioAnalysis(
            duration_seconds=source.duration_seconds,
            sample_rate=source.sample_rate,
            features=features,
            notes=notes,
            raw_notes=notes,
            rhythm=rhythm,
            chords=chords,
        )

    def classify(self, analysis: AudioAnalysis) -> tuple[TrackCandidate, ...]:
        """Assign every cleaned note group to exactly one logical track."""

        if not analysis.notes:
            return ()
        grouped: dict[tuple[float, float], list[Note]] = defaultdict(list)
        for note in analysis.notes:
            grouped[self._event_key(note)].append(note)
        chord_keys = {
            (round(chord.start, 6), round(chord.duration, 6))
            for chord in analysis.chords
        }
        by_role: dict[TrackRole, list[Note]] = defaultdict(list)
        confidence_by_role: dict[TrackRole, list[float]] = defaultdict(list)
        for key in sorted(grouped):
            notes = tuple(sorted(grouped[key], key=lambda note: note.midi))
            role, confidence = self._role_for_group(
                notes,
                chord_keys=chord_keys,
            )
            by_role[role].extend(notes)
            confidence_by_role[role].append(confidence)

        candidates: list[TrackCandidate] = []
        for role in (TrackRole.LEAD, TrackRole.RHYTHM, TrackRole.UNKNOWN):
            notes = tuple(by_role.get(role, ()))
            if not notes:
                continue
            confidences = confidence_by_role[role]
            confidence = sum(confidences) / len(confidences)
            candidates.append(
                TrackCandidate(
                    role=role,
                    name=role.display_name,
                    analysis=self._filtered_analysis(analysis, notes, role=role),
                    confidence=confidence,
                )
            )
        return tuple(candidates)


__all__ = ["TrackCandidate", "TrackClassifier"]
