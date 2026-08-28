"""Tests for Phase 4 rhythm-analysis integration."""

from types import SimpleNamespace
import sys

import numpy as np

from src.audio.loader import AudioData
from src.audio.rhythm import RhythmAnalyzer
from src.music.note import Note


def test_detect_returns_onsets_tempo_and_beats(monkeypatch):
    fake_librosa = SimpleNamespace(
        onset=SimpleNamespace(
            onset_strength=lambda **kwargs: np.ones(8),
            onset_detect=lambda **kwargs: np.array([0.25, 0.75]),
        ),
        beat=SimpleNamespace(
            beat_track=lambda **kwargs: (np.array([120.0]), np.array([1, 3]))
        ),
        frames_to_time=lambda frames, **kwargs: np.array([0.5, 1.0]),
    )
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    audio = AudioData(np.ones(16), 8000, 2.0, 1)

    onsets, timing = RhythmAnalyzer(hop_length=2).detect(audio)

    assert onsets == (0.25, 0.75)
    assert timing.tempo_bpm == 120.0
    assert timing.beat_times == (0.5, 1.0)


def test_rhythm_failure_falls_back_to_seconds(monkeypatch):
    monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace())
    audio = AudioData(np.ones(16), 8000, 2.0, 1)
    note = Note(64, start=0.2, duration=0.5)

    result = RhythmAnalyzer().analyze(audio, (note,))

    assert result.timing.tempo_bpm is None
    assert result.quantized_notes[0].note is note
