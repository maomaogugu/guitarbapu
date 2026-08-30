"""Neural transcription backend powered by Spotify's Basic Pitch.

Basic Pitch (ICASSP 2022 multipitch model) dramatically outperforms the
handcrafted CQT pipeline on dense fingerstyle passages.  The dependency is
optional: importing this module is cheap, but instantiating
:class:`BasicPitchAnalyzer` requires the ``basic-pitch`` extra + TensorFlow.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..music.note import Note
from .analyzer import AudioAnalysis
from .loader import AudioData

_MODEL = None


def _load_model():
    """Import basic-pitch lazily and cache the ICASSP 2022 model."""

    global _MODEL
    if _MODEL is not None:
        return _MODEL
    # The bundled SavedModel predates Keras 3; keep the legacy Keras stack.
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    # basic-pitch 0.3 calls scipy.signal.gaussian, removed in SciPy >= 1.13.
    import scipy.signal
    import scipy.signal.windows

    if not hasattr(scipy.signal, "gaussian"):
        scipy.signal.gaussian = scipy.signal.windows.gaussian
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Basic Pitch 后端需要安装：pip install 'basic-pitch[onnx]' tensorflow tf_keras"
        ) from exc
    _MODEL = Model(ICASSP_2022_MODEL_PATH)
    return _MODEL


class BasicPitchAnalyzer:
    """Drop-in replacement for the CQT analyzer with a neural multipitch model."""

    def __init__(
        self,
        *,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length_ms: float = 127.7,
        minimum_frequency: float | None = 75.0,
        maximum_frequency: float | None = 1400.0,
        min_midi: int = 40,
        max_midi: int = 88,
        min_confidence: float = 0.0,
        simplify: bool = True,
    ) -> None:
        if not 0 < onset_threshold <= 1:
            raise ValueError("onset_threshold must be in (0, 1]")
        if not 0 < frame_threshold <= 1:
            raise ValueError("frame_threshold must be in (0, 1]")
        if minimum_note_length_ms <= 0:
            raise ValueError("minimum_note_length_ms must be positive")
        if not 0 <= min_midi <= max_midi <= 127:
            raise ValueError("midi range must satisfy 0 <= min <= max <= 127")
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        self.onset_threshold = float(onset_threshold)
        self.frame_threshold = float(frame_threshold)
        self.minimum_note_length_ms = float(minimum_note_length_ms)
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self.min_midi = int(min_midi)
        self.max_midi = int(max_midi)
        self.min_confidence = float(min_confidence)
        self.simplify = bool(simplify)

    def detect_notes(self, audio: AudioData) -> tuple[Note, ...]:
        import json
        import subprocess
        import sys

        import soundfile as sf

        waveform = np.asarray(audio.waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = Path(handle.name)
        out_path = temp_path.with_suffix(".json")
        try:
            sf.write(temp_path, waveform, audio.sample_rate)
            # TensorFlow 在部分平台上退出时会崩溃（recursive_mutex /
            # std::terminate at shutdown），推理放进子进程隔离，避免拖垮 GUI。
            timeout = 300.0 + float(audio.duration) * 4.0
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.audio.basic_pitch_worker",
                    str(temp_path),
                    str(out_path),
                    str(self.onset_threshold),
                    str(self.frame_threshold),
                    str(self.minimum_note_length_ms),
                    "" if self.minimum_frequency is None else str(self.minimum_frequency),
                    "" if self.maximum_frequency is None else str(self.maximum_frequency),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Basic Pitch 推理子进程失败："
                    + (completed.stderr.strip().splitlines() or ["未知错误"])[-1]
                )
            note_events = json.loads(out_path.read_text(encoding="utf-8"))
        finally:
            temp_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
        notes = []
        for start, end, midi, amplitude in note_events:
            if amplitude < self.min_confidence:
                continue
            if self.min_midi <= midi <= self.max_midi:
                notes.append(
                    Note(
                        midi=int(midi),
                        start=float(start),
                        duration=max(0.0, float(end - start)),
                        velocity=int(round(100 * float(amplitude))),
                        confidence=float(amplitude),
                    )
                )
        notes.sort(key=lambda note: (note.start, note.midi))
        if self.simplify:
            from ..music.note_filter import simplify_notes

            # Neural output is far denser than a human can play; humanize it.
            return tuple(simplify_notes(tuple(notes)))
        return tuple(notes)

    def analyze(self, audio: AudioData) -> AudioAnalysis:
        notes = self.detect_notes(audio)
        rhythm = None
        if notes:
            try:
                from .rhythm import RhythmAnalyzer

                onset_times, timing = RhythmAnalyzer().detect(audio)
                from .rhythm import RhythmAnalysis

                from ..music.timing import quantize_notes

                rhythm = RhythmAnalysis(
                    timing=timing,
                    onset_times=tuple(onset_times),
                    quantized_notes=quantize_notes(notes, timing),
                )
            except Exception:
                rhythm = None
        features: dict[str, Any] = {"analysis_mode": "basic_pitch"}
        return AudioAnalysis(
            duration_seconds=float(audio.duration),
            sample_rate=int(audio.sample_rate),
            features=features,
            notes=notes,
            raw_notes=notes,
            rhythm=rhythm,
        )
