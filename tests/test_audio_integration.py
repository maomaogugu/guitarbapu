"""Small synthesized-audio integration tests using the real librosa backend."""

import numpy as np
import soundfile as sf

from src.audio.analyzer import AudioAnalyzer
from src.audio.loader import load_audio
from src.music.tab_generator import TabGenerator
from src.music.tab_renderer import TextTabRenderer


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
