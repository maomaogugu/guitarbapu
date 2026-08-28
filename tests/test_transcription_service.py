"""Tests for optional separation orchestration without running a real model."""

from pathlib import Path
from threading import Event

import numpy as np
import pytest
import soundfile as sf

from src.audio.analyzer import AudioAnalysis
from src.audio.loader import load_audio
from src.audio.separator import (
    SeparationCancelled,
    SeparationError,
    SeparationResult,
    Stem,
)
from src.audio.transcription_service import TranscriptionService
from src.music.note import Note
from src.music.tab_generator import TabGenerator
from src.music.technique import GuitarTechnique, TechniqueDetection
from src.music.track import TrackRole


class _Analyzer:
    def __init__(self):
        self.durations = []

    def analyze(self, audio):
        self.durations.append(audio.duration)
        note = Note(64, start=0.0, duration=max(0.1, audio.duration))
        return AudioAnalysis(
            duration_seconds=audio.duration,
            sample_rate=audio.sample_rate,
            notes=(note,),
            raw_notes=(note,),
        )


class _Separator:
    def __init__(self, guitar_path: Path):
        self.guitar_path = guitar_path
        self.calls = 0

    def separate(self, source_path, *, progress_callback=None, cancel_event=None):
        self.calls += 1
        info = sf.info(self.guitar_path)
        return SeparationResult(
            source_path=Path(source_path),
            model_name="fake-guitar-model",
            device="cpu",
            cache_key="a" * 64,
            stems=(
                Stem(
                    "guitar",
                    self.guitar_path,
                    info.samplerate,
                    info.channels,
                    info.duration,
                ),
            ),
        )


class _TechniqueAnalyzer:
    def detect(self, audio, notes, *, pitch_hz=None, pitch_hop_length=None):
        note = tuple(notes)[0]
        return (
            TechniqueDetection(
                GuitarTechnique.VIBRATO,
                note,
                0.86,
            ),
        )


def _wav(path: Path, *, seconds: float, value: float) -> Path:
    sample_rate = 1000
    sf.write(
        path,
        np.full(round(sample_rate * seconds), value, dtype=np.float32),
        sample_rate,
    )
    return path


def test_direct_transcription_reuses_preloaded_audio(tmp_path):
    source = _wav(tmp_path / "source.wav", seconds=1.0, value=0.1)
    analyzer = _Analyzer()
    progress = []
    service = TranscriptionService(
        analyzer=analyzer,
        tab_generator=TabGenerator(),
    )

    result = service.transcribe(
        source,
        audio=load_audio(source),
        progress_callback=progress.append,
    )

    assert result.separation is None
    assert result.analyzed_audio_path == source.resolve()
    assert analyzer.durations == [1.0]
    assert len(result.tablature.events) == 1
    assert len(result.tracks) == 1
    assert result.tracks[0].role is TrackRole.LEAD
    assert result.tracks[0].source_name == "original"
    assert progress[-1].stage == "complete"


def test_transcription_maps_technique_candidates_into_root_and_track_tabs(tmp_path):
    source = _wav(tmp_path / "source.wav", seconds=1.0, value=0.1)
    service = TranscriptionService(
        analyzer=_Analyzer(),
        tab_generator=TabGenerator(),
        technique_analyzer=_TechniqueAnalyzer(),
    )

    result = service.transcribe(source)

    assert result.analysis.techniques[0].technique is GuitarTechnique.VIBRATO
    assert result.tablature.events[0].technique == "vibrato"
    assert result.tablature.events[0].technique_confidence == 0.86
    assert result.tracks[0].analysis.techniques == result.analysis.techniques
    assert result.tracks[0].tablature.events[0].technique == "vibrato"


def test_separated_transcription_analyzes_guitar_stem(tmp_path):
    source = _wav(tmp_path / "mix.wav", seconds=2.0, value=0.2)
    guitar = _wav(tmp_path / "guitar.wav", seconds=0.5, value=0.4)
    analyzer = _Analyzer()
    separator = _Separator(guitar)
    service = TranscriptionService(
        analyzer=analyzer,
        tab_generator=TabGenerator(),
        separator=separator,
    )

    result = service.transcribe(source, use_separation=True)

    assert separator.calls == 1
    assert result.analyzed_audio_path == guitar
    assert result.separation is not None
    assert analyzer.durations == [0.5]
    assert result.tracks[0].source_name == "guitar"


def test_requesting_separation_without_backend_is_explicit(tmp_path):
    source = _wav(tmp_path / "source.wav", seconds=0.1, value=0.0)

    with pytest.raises(SeparationError, match="没有可用"):
        TranscriptionService(analyzer=_Analyzer()).transcribe(
            source, use_separation=True
        )


def test_transcription_honors_cancel_before_analysis(tmp_path):
    source = _wav(tmp_path / "source.wav", seconds=0.1, value=0.0)
    cancelled = Event()
    cancelled.set()

    with pytest.raises(SeparationCancelled):
        TranscriptionService(analyzer=_Analyzer()).transcribe(
            source, cancel_event=cancelled
        )
