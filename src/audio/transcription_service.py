"""Coordinate optional separation with the existing transcription pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from typing import Protocol

from src.music.tab import Tablature
from src.music.tab_generator import TabGenerator
from src.project.track import TranscriptionTrack

from .analyzer import AudioAnalysis, AudioAnalyzer
from .loader import AudioData, load_audio
from .separator import (
    ProgressCallback,
    SeparationCancelled,
    SeparationError,
    SeparationProgress,
    SeparationResult,
    Separator,
)
from .track_classifier import TrackClassifier
from .technique_analyzer import TechniqueAnalyzer


class Analyzer(Protocol):
    def analyze(self, audio: AudioData) -> AudioAnalysis:
        """Return an analysis compatible with TAB generation."""


@dataclass(frozen=True)
class TranscriptionResult:
    source_audio_path: Path
    analyzed_audio_path: Path
    analysis: AudioAnalysis
    tablature: Tablature
    separation: SeparationResult | None = None
    tracks: tuple[TranscriptionTrack, ...] = ()


class TranscriptionService:
    """Keep optional AI separation outside Audio Loader and GUI algorithms."""

    def __init__(
        self,
        *,
        analyzer: Analyzer | None = None,
        tab_generator: TabGenerator | None = None,
        separator: Separator | None = None,
        track_classifier: TrackClassifier | None = None,
        technique_analyzer: TechniqueAnalyzer | None = None,
    ) -> None:
        self.analyzer = analyzer or AudioAnalyzer()
        self.tab_generator = tab_generator or TabGenerator()
        self.separator = separator
        self.track_classifier = track_classifier or TrackClassifier()
        self.technique_analyzer = technique_analyzer or TechniqueAnalyzer()

    @staticmethod
    def _emit(
        callback: ProgressCallback | None,
        stage: str,
        fraction: float | None,
        message: str,
    ) -> None:
        if callback is not None:
            callback(SeparationProgress(stage, fraction, message))

    @staticmethod
    def _check_cancel(cancel_event: Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise SeparationCancelled("转录任务已取消")

    def transcribe(
        self,
        source_path: str | Path,
        *,
        audio: AudioData | None = None,
        use_separation: bool = False,
        progress_callback: ProgressCallback | None = None,
        cancel_event: Event | None = None,
    ) -> TranscriptionResult:
        source = Path(source_path).expanduser().resolve(strict=True)
        self._check_cancel(cancel_event)
        separation = None
        analyzed_path = source
        analyzed_audio = audio

        if use_separation:
            if self.separator is None:
                raise SeparationError("已请求吉他分离，但没有可用的 Separator")
            separation = self.separator.separate(
                source,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
            analyzed_path = separation.stem("guitar").path
            self._check_cancel(cancel_event)
            self._emit(
                progress_callback,
                "loading_stem",
                None,
                "正在加载分离后的吉他音轨…",
            )
            analyzed_audio = load_audio(analyzed_path)
        elif analyzed_audio is None:
            self._emit(
                progress_callback,
                "loading_audio",
                None,
                "正在加载原音频…",
            )
            analyzed_audio = load_audio(source)

        self._check_cancel(cancel_event)
        self._emit(
            progress_callback,
            "analyzing",
            None,
            "正在检测吉他音高和节奏…",
        )
        analysis = self.analyzer.analyze(analyzed_audio)
        self._check_cancel(cancel_event)
        self._emit(
            progress_callback,
            "detecting_techniques",
            None,
            "正在识别滑弦、击勾弦、推弦和颤音候选…",
        )
        pitch_hz = analysis.features.get("pitch_hz")
        hop_length = getattr(self.analyzer, "hop_length", None)
        techniques = self.technique_analyzer.detect(
            analyzed_audio,
            analysis.notes,
            pitch_hz=pitch_hz,
            pitch_hop_length=hop_length,
        )
        analysis = replace(analysis, techniques=techniques)
        self._check_cancel(cancel_event)
        self._emit(
            progress_callback,
            "generating_tab",
            0.98,
            "正在生成六线谱…",
        )
        tablature = self.tab_generator.generate(analysis)
        source_name = "guitar" if separation is not None else "original"
        tracks = tuple(
            TranscriptionTrack(
                track_id=f"logical-{candidate.role.value}",
                name=candidate.name,
                role=candidate.role,
                analysis=candidate.analysis,
                tablature=self.tab_generator.generate(candidate.analysis),
                source_name=source_name,
                confidence=candidate.confidence,
                metadata={
                    "logical": True,
                    "classifier": "structural-v1",
                    "independent_audio": False,
                },
            )
            for candidate in self.track_classifier.classify(analysis)
        )
        self._emit(progress_callback, "complete", 1.0, "转录完成")
        return TranscriptionResult(
            source_audio_path=source,
            analyzed_audio_path=analyzed_path,
            analysis=analysis,
            tablature=tablature,
            separation=separation,
            tracks=tracks,
        )


__all__ = ["TranscriptionResult", "TranscriptionService"]
