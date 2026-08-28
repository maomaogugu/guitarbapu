"""Deterministic tests for the Phase 9C contour classifier."""

import numpy as np

from src.audio.technique_analyzer import TechniqueAnalyzer, TechniqueFeatures
from src.music.note import Note
from src.music.technique import GuitarTechnique, TechniqueDetection


def _features(times, pitch, *, attacks=()):
    times = np.asarray(times, dtype=np.float32)
    onset = np.zeros(times.size, dtype=np.float32)
    for when, strength in attacks:
        onset[int(np.argmin(np.abs(times - when)))] = strength
    return TechniqueFeatures(
        frame_times=times,
        pitch_midi=np.asarray(pitch, dtype=np.float32),
        rms=np.ones(times.size, dtype=np.float32),
        onset_strength=onset,
    )


def test_technique_detection_validates_and_exposes_stable_note_key():
    note = Note(64, start=0.25, duration=0.5)
    detection = TechniqueDetection("slide", note, np.float32(0.8))

    assert detection.technique is GuitarTechnique.SLIDE
    assert detection.note_key == (64, 0.25, 0.5)
    assert isinstance(detection.confidence, float)


def test_continuous_pitch_transition_is_classified_as_slide():
    times = np.arange(0.0, 1.0, 0.01)
    pitch = np.where(
        times < 0.38,
        60.0,
        np.where(times <= 0.62, 60.0 + (times - 0.38) / 0.24 * 4.0, 64.0),
    )
    notes = (
        Note(60, start=0.0, duration=0.5),
        Note(64, start=0.5, duration=0.5),
    )

    detections = TechniqueAnalyzer().classify(
        notes,
        _features(times, pitch, attacks=((0.0, 1.0), (0.5, 0.2))),
    )

    assert len(detections) == 1
    assert detections[0].technique is GuitarTechnique.SLIDE
    assert detections[0].note == notes[1]
    assert detections[0].related_note == notes[0]


def test_weak_rearticulation_distinguishes_hammer_on_and_pull_off():
    times = np.arange(0.0, 1.0, 0.01)
    attacks = ((0.0, 1.0), (0.5, 0.15))
    analyzer = TechniqueAnalyzer()

    hammer_notes = (
        Note(60, start=0.0, duration=0.5),
        Note(64, start=0.5, duration=0.5),
    )
    hammer_pitch = np.where(times < 0.5, 60.0, 64.0)
    hammer = analyzer.classify(
        hammer_notes,
        _features(times, hammer_pitch, attacks=attacks),
    )

    pull_notes = (
        Note(64, start=0.0, duration=0.5),
        Note(60, start=0.5, duration=0.5),
    )
    pull_pitch = np.where(times < 0.5, 64.0, 60.0)
    pull = analyzer.classify(
        pull_notes,
        _features(times, pull_pitch, attacks=attacks),
    )

    assert [item.technique for item in hammer] == [GuitarTechnique.HAMMER_ON]
    assert [item.technique for item in pull] == [GuitarTechnique.PULL_OFF]


def test_rising_sustained_pitch_is_classified_as_bend():
    times = np.arange(0.0, 0.8, 0.01)
    pitch = np.where(times < 0.5, 60.0 + times / 0.5 * 2.0, 62.0)
    note = Note(60, start=0.0, duration=0.8)

    detections = TechniqueAnalyzer().classify(
        (note,),
        _features(times, pitch, attacks=((0.0, 1.0),)),
    )

    assert len(detections) == 1
    assert detections[0].technique is GuitarTechnique.BEND
    assert detections[0].pitch_change_semitones is not None
    assert detections[0].pitch_change_semitones > 1.2


def test_periodic_pitch_modulation_is_classified_as_vibrato():
    times = np.arange(0.0, 1.0, 0.01)
    pitch = 69.0 + 0.22 * np.sin(2 * np.pi * 6.0 * times)
    note = Note(69, start=0.0, duration=1.0)

    detections = TechniqueAnalyzer().classify(
        (note,),
        _features(times, pitch, attacks=((0.0, 1.0),)),
    )

    assert len(detections) == 1
    assert detections[0].technique is GuitarTechnique.VIBRATO


def test_strong_second_attack_and_polyphonic_notes_remain_unclassified():
    times = np.arange(0.0, 1.0, 0.01)
    notes = (
        Note(60, start=0.0, duration=0.5),
        Note(64, start=0.5, duration=0.5),
    )
    pitch = np.where(times < 0.5, 60.0, 64.0)
    analyzer = TechniqueAnalyzer()

    picked = analyzer.classify(
        notes,
        _features(times, pitch, attacks=((0.0, 1.0), (0.5, 1.0))),
    )
    chord = analyzer.classify(
        (
            Note(60, start=0.0, duration=1.0),
            Note(64, start=0.0, duration=1.0),
        ),
        _features(times, np.full(times.size, 60.0), attacks=((0.0, 1.0),)),
    )

    assert picked == ()
    assert chord == ()
