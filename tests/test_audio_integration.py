"""Small synthesized-audio integration tests using the real librosa backend."""

import numpy as np
import soundfile as sf

from src.audio.analyzer import AudioAnalyzer
from src.audio.loader import AudioData, load_audio
from src.audio.technique_analyzer import TechniqueAnalyzer
from src.music.note import Note
from src.music.tab_generator import TabGenerator
from src.music.tab_renderer import TextTabRenderer
from src.music.technique import GuitarTechnique


def _tone(frequency: float, seconds: float, sample_rate: int) -> np.ndarray:
    time = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (0.6 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def test_real_wav_pipeline_detects_two_known_notes(tmp_path):
    sample_rate = 22_050
    silence = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
    waveform = np.concatenate(
        (
            silence,
            _tone(440.0, 0.5, sample_rate),
            silence,
            _tone(329.63, 0.5, sample_rate),
            silence,
        )
    )
    path = tmp_path / "two_notes.wav"
    sf.write(path, waveform, sample_rate, subtype="PCM_16")

    result = AudioAnalyzer(energy_threshold=0.15).analyze(load_audio(path))

    detected = [note.midi for note in result.notes]
    assert 69 in detected  # A4
    assert 64 in detected  # E4
    assert all(note.duration >= 0.08 for note in result.notes)
    assert result.rhythm is not None

    tablature = TabGenerator().generate(result)
    rendered = TextTabRenderer().render(tablature)
    assert len(tablature.events) >= 2
    assert "A4" not in rendered  # Text TAB represents notes as fret positions.
    assert "Mapped:" in rendered


def _frequency_modulated_tone(midi_track, sample_rate):
    frequencies = 440.0 * 2 ** ((midi_track - 69.0) / 12.0)
    phase = np.cumsum(2 * np.pi * frequencies / sample_rate)
    return (0.55 * np.sin(phase)).astype(np.float32)


def test_real_librosa_backend_detects_five_techniques_and_rejects_repicking():
    sample_rate = 22_050
    times = np.arange(sample_rate, dtype=np.float32) / sample_rate
    analyzer = TechniqueAnalyzer()

    transition_notes = (
        Note(60, start=0.0, duration=0.5),
        Note(64, start=0.5, duration=0.5),
    )
    slide_pitch = np.where(
        times < 0.38,
        60.0,
        np.where(
            times < 0.62,
            60.0 + (times - 0.38) / 0.24 * 4.0,
            64.0,
        ),
    )
    slide = analyzer.detect(
        AudioData(
            _frequency_modulated_tone(slide_pitch, sample_rate),
            sample_rate,
            1.0,
            1,
        ),
        transition_notes,
    )

    jump_pitch = np.where(times < 0.5, 60.0, 64.0)
    hammer_on = analyzer.detect(
        AudioData(
            _frequency_modulated_tone(jump_pitch, sample_rate),
            sample_rate,
            1.0,
            1,
        ),
        transition_notes,
    )
    pull_notes = (
        Note(64, start=0.0, duration=0.5),
        Note(60, start=0.5, duration=0.5),
    )
    pull_pitch = np.where(times < 0.5, 64.0, 60.0)
    pull_off = analyzer.detect(
        AudioData(
            _frequency_modulated_tone(pull_pitch, sample_rate),
            sample_rate,
            1.0,
            1,
        ),
        pull_notes,
    )

    bend_pitch = 60.0 + 2.0 * np.minimum(times / 0.55, 1.0)
    bend = analyzer.detect(
        AudioData(
            _frequency_modulated_tone(bend_pitch, sample_rate),
            sample_rate,
            1.0,
            1,
        ),
        (Note(60, start=0.0, duration=1.0),),
    )

    vibrato_pitch = 69.0 + 0.22 * np.sin(2 * np.pi * 6.0 * times)
    vibrato = analyzer.detect(
        AudioData(
            _frequency_modulated_tone(vibrato_pitch, sample_rate),
            sample_rate,
            1.0,
            1,
        ),
        (Note(69, start=0.0, duration=1.0),),
    )

    local_time = np.where(times < 0.5, times, times - 0.5)
    repick_envelope = np.minimum(local_time / 0.015, 1.0) * np.exp(
        -1.8 * local_time
    )
    repicked_waveform = (
        repick_envelope
        * _frequency_modulated_tone(jump_pitch, sample_rate)
    ).astype(np.float32)
    repicked = analyzer.detect(
        AudioData(repicked_waveform, sample_rate, 1.0, 1),
        transition_notes,
    )

    assert [item.technique for item in slide] == [GuitarTechnique.SLIDE]
    assert [item.technique for item in hammer_on] == [GuitarTechnique.HAMMER_ON]
    assert [item.technique for item in pull_off] == [GuitarTechnique.PULL_OFF]
    assert [item.technique for item in bend] == [GuitarTechnique.BEND]
    assert [item.technique for item in vibrato] == [GuitarTechnique.VIBRATO]
    assert repicked == ()
