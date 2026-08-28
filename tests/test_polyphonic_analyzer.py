"""Synthetic-audio tests for the experimental polyphonic analyzer."""

import numpy as np
import pytest

from src.audio.loader import AudioData
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer
from src.music.tab_generator import TabGenerator


def _frequency(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _audio(
    midis,
    *,
    duration=0.8,
    sample_rate=22_050,
    harmonics=False,
):
    frame_count = round(duration * sample_rate)
    time = np.arange(frame_count, dtype=np.float32) / sample_rate
    attack = np.minimum(1.0, np.arange(frame_count) / (sample_rate * 0.02))
    waveform = np.zeros(frame_count, dtype=np.float64)
    harmonic_levels = ((1, 1.0), (2, 0.35), (3, 0.2), (4, 0.12))
    for midi in midis:
        frequency = _frequency(midi)
        levels = harmonic_levels if harmonics else ((1, 1.0),)
        for harmonic, level in levels:
            waveform += level * np.sin(
                2 * np.pi * frequency * harmonic * time
            )
    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        waveform = waveform * attack * (0.4 / peak)
    return AudioData(
        waveform=waveform.astype(np.float32),
        sample_rate=sample_rate,
        duration=duration,
        channels=1,
    )


def test_detects_c_major_as_simultaneous_notes_and_chord():
    analysis = PolyphonicAudioAnalyzer().analyze(_audio((48, 52, 55)))

    assert tuple(note.midi for note in analysis.notes) == (48, 52, 55)
    assert len({note.start for note in analysis.notes}) == 1
    assert len(analysis.chords) == 1
    assert analysis.chords[0].name == "C"
    assert analysis.features["analysis_mode"] == "polyphonic"


def test_harmonic_suppression_keeps_power_chord_fundamentals():
    analysis = PolyphonicAudioAnalyzer().analyze(
        _audio((40, 47), harmonics=True)
    )

    assert tuple(note.midi for note in analysis.notes) == (40, 47)
    assert analysis.chords[0].name == "E5"


def test_silence_returns_no_notes_or_chords():
    audio = AudioData(
        waveform=np.zeros(22_050, dtype=np.float32),
        sample_rate=22_050,
        duration=1.0,
        channels=1,
    )

    analysis = PolyphonicAudioAnalyzer().analyze(audio)

    assert analysis.notes == ()
    assert analysis.chords == ()


def test_single_note_remains_supported_in_polyphonic_mode():
    analysis = PolyphonicAudioAnalyzer().analyze(_audio((57,)))

    assert tuple(note.midi for note in analysis.notes) == (57,)
    assert analysis.chords == ()


def test_consecutive_chords_are_split_by_onsets():
    first = _audio((48, 52, 55), duration=0.8)
    second = _audio((52, 55, 59), duration=0.8)
    sample_rate = first.sample_rate
    silence = np.zeros(round(sample_rate * 0.1), dtype=np.float32)
    waveform = np.concatenate((first.waveform, silence, second.waveform))
    audio = AudioData(
        waveform=waveform,
        sample_rate=sample_rate,
        duration=len(waveform) / sample_rate,
        channels=1,
    )

    analysis = PolyphonicAudioAnalyzer().analyze(audio)

    assert [chord.name for chord in analysis.chords] == ["C", "Em"]
    assert analysis.chords[1].start > analysis.chords[0].start


def test_polyphony_is_capped_at_six_pitches():
    analysis = PolyphonicAudioAnalyzer(
        relative_pitch_threshold=0.15
    ).analyze(_audio((41, 44, 47, 50, 53, 56, 59)))

    simultaneous = [note for note in analysis.notes if note.start == 0.0]
    assert 1 <= len(simultaneous) <= 6


def test_detected_chord_generates_distinct_guitar_strings():
    analysis = PolyphonicAudioAnalyzer().analyze(_audio((48, 52, 55)))

    tablature = TabGenerator().generate(analysis)

    assert len(tablature.events) == 3
    assert len({event.string for event in tablature.events}) == 3
    assert tablature.unmapped_notes == ()


def test_short_strum_is_grouped_as_one_chord():
    sample_rate = 22_050
    duration = 1.0
    frame_count = round(sample_rate * duration)
    waveform = np.zeros(frame_count, dtype=np.float64)
    for index, midi in enumerate((48, 52, 55)):
        start = round(index * 0.03 * sample_rate)
        time = np.arange(frame_count - start) / sample_rate
        envelope = np.minimum(
            1.0,
            np.arange(frame_count - start) / (sample_rate * 0.01),
        ) * np.exp(-1.2 * time)
        waveform[start:] += envelope * np.sin(
            2 * np.pi * _frequency(midi) * time
        )
    waveform = (waveform * (0.4 / np.max(np.abs(waveform)))).astype(np.float32)

    analysis = PolyphonicAudioAnalyzer().analyze(
        AudioData(waveform, sample_rate, duration, 1)
    )

    assert [chord.name for chord in analysis.chords] == ["C"]
    assert analysis.chords[0].midis == (48, 52, 55)


def test_rejects_sample_rate_below_configured_pitch_range():
    audio = AudioData(
        waveform=np.zeros(1000, dtype=np.float32),
        sample_rate=1000,
        duration=1.0,
        channels=1,
    )

    with pytest.raises(ValueError, match="sample rate is too low"):
        PolyphonicAudioAnalyzer().analyze(audio)
