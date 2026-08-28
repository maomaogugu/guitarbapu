"""Tests for the Phase 2 pitch-analysis boundary."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest

from src.audio.analyzer import AudioAnalyzer
from src.audio.loader import AudioData
from src.music.note import Note


def test_note_can_convert_frequency_to_midi_and_name():
    note = Note.from_frequency(440.0, start=1.25, duration=0.5)

    assert note.midi == 69
    assert note.name == "A4"
    assert note.frequency == pytest.approx(440.0)
    assert note.start == 1.25


def test_note_rejects_invalid_frequency():
    with pytest.raises(ValueError):
        Note.from_frequency(0)


def test_pitch_track_filters_quiet_frames_and_merges_notes(monkeypatch):
    fake_librosa = SimpleNamespace(
        yin=lambda *args, **kwargs: np.array([440.0, 440.0, 659.25, np.nan]),
        feature=SimpleNamespace(
            rms=lambda **kwargs: np.array([[1.0, 1.0, 1.0, 0.0]])
        ),
    )
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)

    audio = AudioData(
        waveform=np.ones(16, dtype=np.float32),
        sample_rate=8000,
        duration=16 / 8000,
        channels=1,
    )
    analyzer = AudioAnalyzer(
        frame_length=4,
        hop_length=2,
        min_note_duration=0,
    )

    pitches = analyzer.detect_pitch(audio)
    notes = analyzer.detect_notes(audio)

    assert np.allclose(pitches[:3], [440.0, 440.0, 659.25])
    assert np.isnan(pitches[3])
    assert [note.midi for note in notes] == [69, 76]
    assert notes[0].duration == pytest.approx(4 / 8000)
    assert notes[1].start == pytest.approx(4 / 8000)


def test_analyze_returns_notes_and_pitch_feature(monkeypatch):
    fake_librosa = SimpleNamespace(
        yin=lambda *args, **kwargs: np.array([440.0]),
        feature=SimpleNamespace(rms=lambda **kwargs: np.array([[1.0]])),
    )
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    audio = AudioData(np.ones(8), 8000, 8 / 8000, 1)

    result = AudioAnalyzer(
        frame_length=4,
        hop_length=2,
        min_note_duration=0,
    ).analyze(audio)

    assert result.sample_rate == 8000
    assert [note.midi for note in result.notes] == [69]
    assert result.notes[0].duration == pytest.approx(2 / 8000)
    assert result.notes[0].frequency_hz == pytest.approx(440.0)
    assert result.raw_notes == result.notes
    assert result.rhythm is not None
    assert result.features["pitch_hz"].shape == (1,)


def test_silence_does_not_create_notes(monkeypatch):
    fake_librosa = SimpleNamespace(
        yin=lambda *args, **kwargs: np.array([440.0, 440.0]),
        feature=SimpleNamespace(rms=lambda **kwargs: np.zeros((1, 2))),
    )
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    audio = AudioData(np.zeros(8), 8000, 8 / 8000, 1)

    assert AudioAnalyzer(frame_length=4, hop_length=2).detect_notes(audio) == ()
