"""Onset, tempo, quantization, and rest analysis for monophonic audio."""

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np

from ..music.note import Note
from ..music.timing import (
    QuantizedNote,
    Rest,
    TimingInfo,
    find_rests,
    quantize_notes,
)
from .librosa_compat import import_librosa
from .loader import AudioData


@dataclass(frozen=True)
class RhythmAnalysis:
    """Timing information derived from an audio waveform and its notes."""

    timing: TimingInfo = field(default_factory=TimingInfo)
    onset_times: tuple[float, ...] = field(default_factory=tuple)
    quantized_notes: tuple[QuantizedNote, ...] = field(default_factory=tuple)
    rests: tuple[Rest, ...] = field(default_factory=tuple)


class RhythmAnalyzer:
    """Estimate attacks and tempo, then build an editable beat grid."""

    def __init__(
        self,
        *,
        hop_length: int = 512,
        subdivision: int = 4,
        min_rest_duration: float = 0.12,
    ) -> None:
        if hop_length < 1:
            raise ValueError("hop_length must be positive")
        if subdivision < 1:
            raise ValueError("subdivision must be positive")
        if min_rest_duration < 0:
            raise ValueError("min_rest_duration must be non-negative")
        self.hop_length = int(hop_length)
        self.subdivision = int(subdivision)
        self.min_rest_duration = float(min_rest_duration)

    @staticmethod
    def _waveform(audio: AudioData) -> np.ndarray:
        waveform = np.asarray(audio.waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if waveform.ndim != 1:
            raise ValueError("audio waveform must be one-dimensional")
        return waveform

    @staticmethod
    def _tempo_value(value: object) -> float | None:
        values = np.asarray(value, dtype=float).reshape(-1)
        if not values.size:
            return None
        tempo = float(values[0])
        if not math.isfinite(tempo) or tempo <= 0:
            return None
        return tempo

    @staticmethod
    def _resolve_double_time(
        onset_envelope: np.ndarray,
        tempo_bpm: float | None,
        beat_times: np.ndarray,
        sample_rate: int,
        hop_length: int,
    ) -> tuple[float | None, np.ndarray]:
        """Fold double-/half-time tempo estimates toward a plausible pop tempo.

        Onset-based trackers often lock onto the eighth-note pulse of slow
        fingerstyle pieces and report double the notated BPM.  Both octaves fit
        the onset grid equally well, so pick the octave-ambiguous candidate
        closest to typical popular-song tempo, keeping the detector's answer on
        ties.
        """

        if tempo_bpm is None:
            return tempo_bpm, beat_times
        candidates = {tempo_bpm: beat_times}
        half = tempo_bpm / 2.0
        if 40.0 <= half <= 130.0:
            candidates[half] = beat_times[::2]
        double = tempo_bpm * 2.0
        if double <= 200.0 and beat_times.size:
            upsampled = np.concatenate(
                (beat_times, beat_times + float(30.0 / tempo_bpm))
            )
            candidates[double] = np.sort(upsampled)
        chosen = min(
            candidates,
            key=lambda value: (abs(value - 100.0), 0 if value == tempo_bpm else 1),
        )
        return chosen, np.asarray(candidates[chosen])

    def detect(self, audio: AudioData) -> tuple[tuple[float, ...], TimingInfo]:
        """Return onset times and global beat information.

        Rhythm inference is best-effort: quiet or non-rhythmic recordings keep
        second-based notes and return an unknown tempo instead of failing the
        pitch-analysis pipeline.
        """

        waveform = self._waveform(audio)
        fallback = TimingInfo(subdivision=self.subdivision)
        if waveform.size == 0 or audio.sample_rate <= 0:
            return (), fallback

        try:
            librosa = import_librosa()
            onset_envelope = librosa.onset.onset_strength(
                y=waveform,
                sr=audio.sample_rate,
                hop_length=self.hop_length,
            )
            onset_times = librosa.onset.onset_detect(
                onset_envelope=onset_envelope,
                sr=audio.sample_rate,
                hop_length=self.hop_length,
                units="time",
                backtrack=True,
            )
            tempo, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_envelope,
                sr=audio.sample_rate,
                hop_length=self.hop_length,
            )
            beat_times = librosa.frames_to_time(
                beat_frames,
                sr=audio.sample_rate,
                hop_length=self.hop_length,
            )
        except Exception:
            return (), fallback

        tempo_bpm = self._tempo_value(tempo)
        tempo_bpm, beat_times = self._resolve_double_time(
            np.asarray(onset_envelope, dtype=float),
            tempo_bpm,
            np.asarray(beat_times),
            audio.sample_rate,
            self.hop_length,
        )
        timing = TimingInfo(
            tempo_bpm=tempo_bpm,
            beat_times=tuple(float(value) for value in np.asarray(beat_times)),
            # Automatic time-signature detection is intentionally deferred.
            time_signature=None,
            subdivision=self.subdivision,
        )
        return (
            tuple(float(value) for value in np.asarray(onset_times)),
            timing,
        )

    def build_analysis(
        self,
        audio: AudioData,
        notes: Iterable[Note],
        *,
        onset_times: Iterable[float] = (),
        timing: TimingInfo | None = None,
    ) -> RhythmAnalysis:
        """Quantize notes and derive rests from already detected timing."""

        note_tuple = tuple(notes)
        if timing is None:
            detected_onsets, timing = self.detect(audio)
            onset_times = tuple(onset_times) or detected_onsets
        onset_tuple = tuple(float(value) for value in onset_times)
        return RhythmAnalysis(
            timing=timing,
            onset_times=onset_tuple,
            quantized_notes=quantize_notes(note_tuple, timing),
            rests=find_rests(
                note_tuple,
                total_duration=float(audio.duration),
                timing=timing,
                min_duration=self.min_rest_duration,
            ),
        )

    def analyze(self, audio: AudioData, notes: Iterable[Note]) -> RhythmAnalysis:
        """Run complete timing analysis for a cleaned note sequence."""

        onset_times, timing = self.detect(audio)
        return self.build_analysis(
            audio,
            notes,
            onset_times=onset_times,
            timing=timing,
        )


__all__ = ["RhythmAnalysis", "RhythmAnalyzer"]
