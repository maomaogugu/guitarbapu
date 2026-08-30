"""Tests for the optional Basic Pitch transcription backend."""

import numpy as np
import pytest

from src.audio.basic_pitch_backend import BasicPitchAnalyzer
from src.audio.loader import AudioData


def _sine_audio(midis=(57, 64), duration=1.2, sample_rate=22_050):
    frame_count = round(duration * sample_rate)
    time = np.arange(frame_count, dtype=np.float32) / sample_rate
    attack = np.minimum(1.0, np.arange(frame_count) / (sample_rate * 0.02))
    waveform = np.zeros(frame_count, dtype=np.float64)
    for midi in midis:
        frequency = 440.0 * 2 ** ((midi - 69) / 12)
        waveform += np.sin(2 * np.pi * frequency * time)
    peak = float(np.max(np.abs(waveform)))
    waveform = waveform * attack * (0.5 / peak)
    return AudioData(
        waveform=waveform.astype(np.float32),
        sample_rate=sample_rate,
        duration=duration,
        channels=1,
    )


def test_parameter_validation():
    with pytest.raises(ValueError, match="onset_threshold"):
        BasicPitchAnalyzer(onset_threshold=0.0)
    with pytest.raises(ValueError, match="frame_threshold"):
        BasicPitchAnalyzer(frame_threshold=1.5)
    with pytest.raises(ValueError, match="minimum_note_length_ms"):
        BasicPitchAnalyzer(minimum_note_length_ms=0.0)
    with pytest.raises(ValueError, match="midi range"):
        BasicPitchAnalyzer(min_midi=90, max_midi=40)
    with pytest.raises(ValueError, match="min_confidence"):
        BasicPitchAnalyzer(min_confidence=1.5)


def test_min_confidence_filters_weak_notes():
    try:
        analyzer = BasicPitchAnalyzer(min_confidence=0.0)
        analysis = analyzer.analyze(_sine_audio())
        filtered = BasicPitchAnalyzer(min_confidence=0.99).analyze(_sine_audio())
    except RuntimeError as exc:
        pytest.skip(f"basic-pitch 未安装: {exc}")
    assert len(filtered.notes) <= len(analysis.notes)


def test_analyze_produces_neural_notes():
    try:
        analyzer = BasicPitchAnalyzer()
        analysis = analyzer.analyze(_sine_audio())
    except RuntimeError as exc:
        pytest.skip(f"basic-pitch 未安装: {exc}")
    assert analysis.features["analysis_mode"] == "basic_pitch"
    midis = {note.midi for note in analysis.notes}
    assert 57 in midis or 64 in midis
