"""Basic monophonic pitch analysis for decoded audio."""

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import numpy as np

from ..music.chord import Chord
from ..music.note import Note
from ..music.note_processor import NoteProcessor

from .librosa_compat import import_librosa
from .loader import AudioData
from .rhythm import RhythmAnalysis, RhythmAnalyzer


@dataclass(frozen=True)
class AudioAnalysis:
    """Pitch-analysis output passed from audio processing to music models."""

    duration_seconds: float
    sample_rate: int
    features: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[Note, ...] = field(default_factory=tuple)
    raw_notes: tuple[Note, ...] = field(default_factory=tuple)
    rhythm: RhythmAnalysis | None = None
    chords: tuple[Chord, ...] = field(default_factory=tuple)


class AudioAnalyzer:
    """Extract a first-pass monophonic pitch track with librosa's YIN."""

    def __init__(
        self,
        *,
        fmin_hz: float = 65.41,
        fmax_hz: float = 1046.50,
        frame_length: int = 2048,
        hop_length: int = 512,
        energy_threshold: float = 0.1,
        min_note_duration: float = 0.08,
        merge_gap: float = 0.08,
        blip_max_duration: float = 0.06,
        min_onset_segment: float = 0.15,
        beat_subdivision: int = 4,
    ) -> None:
        if not 0 < fmin_hz < fmax_hz:
            raise ValueError("fmin_hz must be positive and lower than fmax_hz")
        if frame_length < 2 or frame_length % 2:
            raise ValueError("frame_length must be an even integer >= 2")
        if hop_length < 1:
            raise ValueError("hop_length must be a positive integer")
        if not 0 <= energy_threshold <= 1:
            raise ValueError("energy_threshold must be between 0 and 1")

        self.fmin_hz = float(fmin_hz)
        self.fmax_hz = float(fmax_hz)
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.energy_threshold = float(energy_threshold)
        self.note_processor = NoteProcessor(
            min_note_duration=min_note_duration,
            merge_gap=merge_gap,
            blip_max_duration=blip_max_duration,
            min_onset_segment=min_onset_segment,
        )
        self.rhythm_analyzer = RhythmAnalyzer(
            hop_length=self.hop_length,
            subdivision=beat_subdivision,
        )

    @staticmethod
    def _waveform(audio: AudioData) -> np.ndarray:
        waveform = np.asarray(audio.waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if waveform.ndim != 1:
            raise ValueError("audio waveform must be a one-dimensional array")
        return waveform

    def detect_pitch(self, audio: AudioData) -> np.ndarray:
        """Return one YIN frequency estimate per frame, in hertz.

        Unvoiced/low-energy frames are represented as ``numpy.nan``. The
        decoded waveform and sample rate come directly from ``AudioData``.
        """

        waveform = self._waveform(audio)
        if audio.sample_rate <= 0:
            raise ValueError("audio.sample_rate must be positive")
        if waveform.size == 0:
            return np.empty(0, dtype=np.float32)

        try:
            librosa = import_librosa()
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("librosa is required for pitch detection") from exc

        try:
            frequencies = np.asarray(
                librosa.yin(
                    waveform,
                    fmin=self.fmin_hz,
                    fmax=self.fmax_hz,
                    sr=audio.sample_rate,
                    frame_length=self.frame_length,
                    hop_length=self.hop_length,
                ),
                dtype=np.float32,
            )
            rms = np.asarray(
                librosa.feature.rms(
                    y=waveform,
                    frame_length=self.frame_length,
                    hop_length=self.hop_length,
                    center=True,
                )[0],
                dtype=np.float32,
            )
        except Exception as exc:
            raise RuntimeError("librosa could not compute a pitch track") from exc

        frame_count = min(frequencies.size, rms.size)
        frequencies = frequencies[:frame_count]
        rms = rms[:frame_count]
        if frame_count == 0:
            return np.empty(0, dtype=np.float32)

        peak_energy = float(np.max(rms))
        # Relative gating avoids treating YIN's guesses on silence as notes.
        if not math.isfinite(peak_energy) or peak_energy <= 0:
            return np.full(frame_count, np.nan, dtype=np.float32)
        threshold = peak_energy * self.energy_threshold
        voiced = np.isfinite(frequencies) & (frequencies > 0)
        if threshold > 0:
            voiced &= rms >= threshold
        return np.where(voiced, frequencies, np.nan).astype(np.float32)

    def _notes_from_frequencies(
        self, frequencies: np.ndarray, sample_rate: int
    ) -> tuple[Note, ...]:
        """Convert a pitch track into contiguous, quantized note events."""

        if frequencies.size == 0:
            return ()

        frame_duration = self.hop_length / sample_rate
        notes: list[Note] = []
        active_midi: int | None = None
        active_start = 0.0
        active_end = 0.0
        active_frequencies: list[float] = []

        def append_active_note() -> None:
            if active_midi is None:
                return
            frequency_hz = (
                float(np.median(active_frequencies))
                if active_frequencies
                else None
            )
            confidence = None
            if frequency_hz is not None and active_frequencies:
                target = 440.0 * 2 ** ((active_midi - 69) / 12)
                cents = np.asarray(
                    [1200 * math.log2(value / target) for value in active_frequencies]
                )
                error = float(np.median(np.abs(cents)))
                spread = float(np.std(cents))
                confidence = max(0.0, min(1.0, 1.0 - (error + spread) / 100.0))
            notes.append(
                Note(
                    midi=active_midi,
                    start=active_start,
                    duration=max(0.0, active_end - active_start),
                    frequency_hz=frequency_hz,
                    confidence=confidence,
                )
            )

        for index, frequency in enumerate(frequencies):
            midi: int | None = None
            if math.isfinite(float(frequency)):
                try:
                    midi = Note.from_frequency(float(frequency)).midi
                except ValueError:
                    midi = None

            frame_start = index * frame_duration
            frame_end = frame_start + frame_duration
            if midi != active_midi:
                append_active_note()
                active_midi = midi
                active_start = frame_start
                active_frequencies = []
            if midi is not None:
                active_end = frame_end
                active_frequencies.append(float(frequency))

        append_active_note()
        return tuple(notes)

    def detect_notes(self, audio: AudioData) -> tuple[Note, ...]:
        """Run pitch detection and convert it into ``Note`` events."""

        raw_notes = self._notes_from_frequencies(
            self.detect_pitch(audio), audio.sample_rate
        )
        return self.note_processor.process(raw_notes)

    def analyze(self, audio: AudioData) -> AudioAnalysis:
        """Analyze decoded audio and return pitch features plus note events."""

        frequencies = self.detect_pitch(audio)
        raw_notes = self._notes_from_frequencies(frequencies, audio.sample_rate)
        onset_times, timing = self.rhythm_analyzer.detect(audio)
        notes = self.note_processor.process(raw_notes, onset_times=onset_times)
        rhythm = self.rhythm_analyzer.build_analysis(
            audio,
            notes,
            onset_times=onset_times,
            timing=timing,
        )
        return AudioAnalysis(
            duration_seconds=float(audio.duration),
            sample_rate=int(audio.sample_rate),
            features={
                "pitch_hz": frequencies,
                "onset_times": np.asarray(onset_times, dtype=np.float32),
                "tempo_bpm": timing.tempo_bpm,
            },
            notes=notes,
            raw_notes=raw_notes,
            rhythm=rhythm,
        )
