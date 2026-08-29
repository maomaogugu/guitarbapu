"""Tests for conservative logical lead/rhythm event classification."""

import pytest

from src.audio.analyzer import AudioAnalysis
from src.audio.rhythm import RhythmAnalysis
from src.audio.track_classifier import TrackClassifier
from src.music.chord import Chord
from src.music.note import Note
from src.music.tab_generator import TabGenerator
from src.music.timing import TimingInfo, quantize_notes
from src.music.technique import GuitarTechnique, TechniqueDetection
from src.music.track import TrackRole
from src.project.track import TranscriptionTrack


def _analysis(notes, *, chords=(), techniques=()):
    notes = tuple(notes)
    timing = TimingInfo(tempo_bpm=120.0, time_signature=(4, 4))
    return AudioAnalysis(
        duration_seconds=3.0,
        sample_rate=44_100,
        notes=notes,
        raw_notes=notes,
        rhythm=RhythmAnalysis(
            timing=timing,
            quantized_notes=quantize_notes(notes, timing),
        ),
        chords=tuple(chords),
        techniques=tuple(techniques),
    )


def test_single_note_material_creates_lead_candidate():
    analysis = _analysis(
        (
            Note(64, start=0.0, duration=0.5, confidence=0.9),
            Note(67, start=0.5, duration=0.5, confidence=0.8),
        )
    )

    candidates = TrackClassifier().classify(analysis)

    assert len(candidates) == 1
    assert candidates[0].role is TrackRole.LEAD
    assert candidates[0].analysis.notes == analysis.notes
    assert candidates[0].confidence == pytest.approx(0.85)


def test_simultaneous_notes_create_rhythm_candidate_and_playable_tab():
    notes = tuple(
        Note(midi, start=0.0, duration=1.0, confidence=0.9)
        for midi in (48, 52, 55)
    )
    analysis = _analysis(
        notes,
        chords=(Chord.from_midis((48, 52, 55), duration=1.0, confidence=0.9),),
    )

    candidate = TrackClassifier().classify(analysis)[0]
    tablature = TabGenerator().generate(candidate.analysis)

    assert candidate.role is TrackRole.RHYTHM
    assert candidate.analysis.chords[0].name == "C"
    assert len(tablature.events) == 3
    assert len({event.string for event in tablature.events}) == 3


def test_mixed_material_splits_without_duplicate_notes():
    lead = Note(69, start=0.0, duration=0.5, confidence=0.9)
    chord_notes = tuple(
        Note(midi, start=1.0, duration=1.0, confidence=0.8)
        for midi in (48, 52, 55)
    )
    analysis = _analysis(
        (lead,) + chord_notes,
        chords=(Chord.from_midis((48, 52, 55), start=1.0, duration=1.0),),
    )

    candidates = TrackClassifier().classify(analysis)

    assert [candidate.role for candidate in candidates] == [
        TrackRole.LEAD,
        TrackRole.RHYTHM,
    ]
    classified = [
        (note.midi, note.start, note.duration)
        for candidate in candidates
        for note in candidate.analysis.notes
    ]
    expected = [(note.midi, note.start, note.duration) for note in analysis.notes]
    assert sorted(classified) == sorted(expected)
    assert len(classified) == len(set(classified))


def test_fingerstyle_option_moves_highest_chord_tone_to_lead():
    notes = tuple(
        Note(midi, start=0.0, duration=1.0, confidence=0.9)
        for midi in (48, 55, 64)
    )
    analysis = _analysis(notes)

    default_roles = TrackClassifier().classify(analysis)
    melody_roles = TrackClassifier(
        extract_melody_from_polyphony=True,
    ).classify(analysis)

    assert [candidate.role for candidate in default_roles] == [TrackRole.RHYTHM]
    assert [candidate.role for candidate in melody_roles] == [
        TrackRole.LEAD,
        TrackRole.RHYTHM,
    ]
    assert [note.midi for note in melody_roles[0].analysis.notes] == [64]
    assert [note.midi for note in melody_roles[1].analysis.notes] == [48, 55]

    high_threshold_roles = TrackClassifier(
        extract_melody_from_polyphony=True,
        melody_min_midi=65,
    ).classify(analysis)
    assert [candidate.role for candidate in high_threshold_roles] == [TrackRole.RHYTHM]


def test_low_confidence_group_is_kept_as_unknown():
    analysis = _analysis(
        (Note(64, start=0.0, duration=0.5, confidence=0.1),)
    )

    candidate = TrackClassifier(confidence_threshold=0.35).classify(analysis)[0]

    assert candidate.role is TrackRole.UNKNOWN
    assert candidate.analysis.features["logical_track"] is True


def test_track_model_validates_role_confidence_and_metadata():
    analysis = _analysis((Note(64, duration=0.5),))
    tablature = TabGenerator().generate(analysis)
    track = TranscriptionTrack(
        track_id="lead",
        name="Lead",
        role="lead",
        analysis=analysis,
        tablature=tablature,
        confidence=0.7,
    )

    assert track.role is TrackRole.LEAD
    with pytest.raises(ValueError, match="confidence"):
        TranscriptionTrack(
            track_id="bad",
            name="Bad",
            role=TrackRole.UNKNOWN,
            analysis=analysis,
            tablature=tablature,
            confidence=2.0,
        )


def test_classifier_keeps_technique_only_with_its_logical_note():
    lead = Note(69, start=0.0, duration=0.5, confidence=0.9)
    chord_notes = tuple(
        Note(midi, start=1.0, duration=1.0, confidence=0.8)
        for midi in (48, 52, 55)
    )
    detection = TechniqueDetection(GuitarTechnique.VIBRATO, lead, 0.8)
    analysis = _analysis(
        (lead,) + chord_notes,
        chords=(Chord.from_midis((48, 52, 55), start=1.0, duration=1.0),),
        techniques=(detection,),
    )

    lead_track, rhythm_track = TrackClassifier().classify(analysis)

    assert lead_track.analysis.techniques == (detection,)
    assert rhythm_track.analysis.techniques == ()
