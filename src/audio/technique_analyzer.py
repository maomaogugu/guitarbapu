"""Conservative guitar-technique recognition from pitch and attack contours."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from ..music.note import Note
from ..music.technique import GuitarTechnique, TechniqueDetection
from .librosa_compat import import_librosa
from .loader import AudioData


@dataclass(frozen=True)
class TechniqueFeatures:
    """Frame-level features consumed by the deterministic classifier."""

    frame_times: np.ndarray
    pitch_midi: np.ndarray
    rms: np.ndarray
    onset_strength: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(
            np.asarray(value, dtype=np.float32)
            for value in (
                self.frame_times,
                self.pitch_midi,
                self.rms,
                self.onset_strength,
            )
        )
        if any(value.ndim != 1 for value in arrays):
            raise ValueError("technique features must be one-dimensional")
        if len({value.size for value in arrays}) > 1:
            raise ValueError("technique feature arrays must have equal lengths")
        for name, value in zip(
            ("frame_times", "pitch_midi", "rms", "onset_strength"),
            arrays,
        ):
            object.__setattr__(self, name, value)


class TechniqueAnalyzer:
    """Detect explainable technique candidates without a large extra model.

    The baseline deliberately favors precision over recall. It only classifies
    monophonic note passages and returns no label when pitch continuity or
    attack evidence is weak. All labels remain editable in the GUI.
    """

    def __init__(
        self,
        *,
        fmin_hz: float = 65.41,
        fmax_hz: float = 1046.50,
        frame_length: int = 2048,
        hop_length: int = 256,
        energy_threshold: float = 0.06,
        max_transition_gap: float = 0.12,
        legato_onset_threshold: float = 0.50,
        min_bend_semitones: float = 0.55,
        vibrato_rate_hz: tuple[float, float] = (3.5, 9.0),
    ) -> None:
        if not 0 < fmin_hz < fmax_hz:
            raise ValueError("fmin_hz must be positive and below fmax_hz")
        if frame_length < 2 or frame_length % 2:
            raise ValueError("frame_length must be an even integer >= 2")
        if hop_length < 1:
            raise ValueError("hop_length must be positive")
        if not 0 <= energy_threshold <= 1:
            raise ValueError("energy_threshold must be between 0 and 1")
        if max_transition_gap < 0:
            raise ValueError("max_transition_gap must be non-negative")
        if not 0 <= legato_onset_threshold <= 1:
            raise ValueError("legato_onset_threshold must be between 0 and 1")
        if min_bend_semitones <= 0:
            raise ValueError("min_bend_semitones must be positive")
        if not 0 < vibrato_rate_hz[0] < vibrato_rate_hz[1]:
            raise ValueError("vibrato_rate_hz must be a positive range")

        self.fmin_hz = float(fmin_hz)
        self.fmax_hz = float(fmax_hz)
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.energy_threshold = float(energy_threshold)
        self.max_transition_gap = float(max_transition_gap)
        self.legato_onset_threshold = float(legato_onset_threshold)
        self.min_bend_semitones = float(min_bend_semitones)
        self.vibrato_rate_hz = (
            float(vibrato_rate_hz[0]),
            float(vibrato_rate_hz[1]),
        )

    @staticmethod
    def _waveform(audio: AudioData) -> np.ndarray:
        waveform = np.asarray(audio.waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if waveform.ndim != 1:
            raise ValueError("audio waveform must be one-dimensional")
        return waveform

    def extract_features(
        self,
        audio: AudioData,
        *,
        pitch_hz: np.ndarray | None = None,
        pitch_hop_length: int | None = None,
    ) -> TechniqueFeatures:
        """Extract aligned YIN pitch, RMS, and amplitude-attack frame tracks."""

        waveform = self._waveform(audio)
        if waveform.size == 0 or audio.sample_rate <= 0:
            empty = np.empty(0, dtype=np.float32)
            return TechniqueFeatures(empty, empty, empty, empty)
        if waveform.size < self.frame_length:
            empty = np.empty(0, dtype=np.float32)
            return TechniqueFeatures(empty, empty, empty, empty)
        if audio.sample_rate / 2 <= self.fmin_hz:
            raise ValueError("audio sample rate is too low for technique analysis")

        librosa = import_librosa()
        hop_length = int(pitch_hop_length or self.hop_length)
        if pitch_hz is None:
            safe_fmax = min(self.fmax_hz, audio.sample_rate / 2 * 0.95)
            frequencies = np.asarray(
                librosa.yin(
                    waveform,
                    fmin=self.fmin_hz,
                    fmax=safe_fmax,
                    sr=audio.sample_rate,
                    frame_length=self.frame_length,
                    hop_length=hop_length,
                ),
                dtype=np.float32,
            )
        else:
            frequencies = np.asarray(pitch_hz, dtype=np.float32).reshape(-1)
        rms = np.asarray(
            librosa.feature.rms(
                y=waveform,
                frame_length=self.frame_length,
                hop_length=hop_length,
                center=True,
            )[0],
            dtype=np.float32,
        )
        times = np.asarray(
            librosa.frames_to_time(
                np.arange(frequencies.size),
                sr=audio.sample_rate,
                hop_length=hop_length,
            ),
            dtype=np.float32,
        )
        frame_count = min(frequencies.size, rms.size, times.size)
        frequencies = frequencies[:frame_count]
        rms = rms[:frame_count]
        times = times[:frame_count]
        onset = np.maximum(
            rms - np.concatenate((rms[:1], rms[:-1])),
            0.0,
        ).astype(np.float32)

        peak_rms = float(np.max(rms)) if rms.size else 0.0
        voiced = np.isfinite(frequencies) & (frequencies > 0)
        if math.isfinite(peak_rms) and peak_rms > 0:
            voiced &= rms >= peak_rms * self.energy_threshold
        else:
            voiced &= False
        pitch_midi = np.full(frame_count, np.nan, dtype=np.float32)
        pitch_midi[voiced] = (
            69.0 + 12.0 * np.log2(frequencies[voiced] / 440.0)
        )
        peak_onset = float(np.max(onset)) if onset.size else 0.0
        if math.isfinite(peak_onset) and peak_onset > 0:
            onset = onset / peak_onset
        else:
            onset = np.zeros_like(onset)
        return TechniqueFeatures(times, pitch_midi, rms, onset)

    @staticmethod
    def _note_key(note: Note) -> tuple[int, float, float]:
        return (note.midi, round(note.start, 6), round(note.duration, 6))

    @staticmethod
    def _smooth(values: np.ndarray, width: int = 5) -> np.ndarray:
        if values.size == 0:
            return values.astype(np.float32)
        radius = max(1, width // 2)
        smoothed = np.full(values.size, np.nan, dtype=np.float32)
        for index in range(values.size):
            window = values[
                max(0, index - radius) : min(values.size, index + radius + 1)
            ]
            finite = window[np.isfinite(window)]
            if finite.size:
                smoothed[index] = float(np.median(finite))
        return smoothed

    @staticmethod
    def _window(
        features: TechniqueFeatures,
        start: float,
        end: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = (features.frame_times >= start) & (features.frame_times <= end)
        return features.frame_times[mask], features.pitch_midi[mask]

    @staticmethod
    def _attack(features: TechniqueFeatures, time: float) -> float:
        mask = np.abs(features.frame_times - time) <= 0.05
        if not np.any(mask):
            return 0.0
        return float(np.max(features.onset_strength[mask]))

    @staticmethod
    def _is_monophonic(note: Note, notes: tuple[Note, ...]) -> bool:
        start = note.start
        end = note.start + note.duration
        for other in notes:
            if other is note:
                continue
            other_start = other.start
            other_end = other.start + other.duration
            if max(start, other_start) < min(end, other_end) - 0.02:
                return False
        return True

    def _transition_detection(
        self,
        previous: Note,
        current: Note,
        features: TechniqueFeatures,
    ) -> TechniqueDetection | None:
        gap = current.start - (previous.start + previous.duration)
        interval = current.midi - previous.midi
        if gap < -0.06 or gap > self.max_transition_gap:
            return None
        if not 1 <= abs(interval) <= 12:
            return None

        start = max(previous.start, current.start - 0.14)
        end = min(
            current.start + max(0.10, current.duration * 0.25),
            current.start + 0.18,
        )
        _, contour = self._window(features, start, end)
        contour = self._smooth(contour)
        contour = contour[np.isfinite(contour)]
        current_attack = self._attack(features, current.start)

        if contour.size >= 7:
            third = max(2, contour.size // 3)
            early = float(np.median(contour[:third]))
            late = float(np.median(contour[-third:]))
            observed = late - early
            direction = 1.0 if interval > 0 else -1.0
            differences = np.diff(contour) * direction
            monotonic = float(np.mean(differences >= -0.04))
            absolute_steps = np.abs(np.diff(contour))
            motion_fraction = float(np.mean(absolute_steps >= 0.04))
            large_step = float(np.percentile(absolute_steps, 90))
            low, high = sorted((previous.midi, current.midi))
            intermediate = float(
                np.mean((contour > low + 0.15) & (contour < high - 0.15))
            )
            if (
                observed * direction >= min(0.65, abs(interval) * 0.45)
                and monotonic >= 0.58
                and motion_fraction >= 0.55
                and large_step <= 0.45
                and intermediate >= 0.18
            ):
                confidence = min(
                    0.96,
                    0.45
                    + 0.20 * min(1.0, abs(observed) / abs(interval))
                    + 0.20 * monotonic
                    + 0.15 * motion_fraction,
                )
                return TechniqueDetection(
                    GuitarTechnique.SLIDE,
                    current,
                    confidence,
                    related_note=previous,
                    pitch_change_semitones=observed,
                )

        if (
            gap <= 0.08
            and abs(interval) <= 7
            and current_attack <= self.legato_onset_threshold
        ):
            technique = (
                GuitarTechnique.HAMMER_ON
                if interval > 0
                else GuitarTechnique.PULL_OFF
            )
            confidence = min(
                0.92,
                0.52
                + 0.28 * (1.0 - min(1.0, current_attack))
                + 0.12 * (1.0 - abs(interval) / 12.0),
            )
            return TechniqueDetection(
                technique,
                current,
                confidence,
                related_note=previous,
                pitch_change_semitones=float(interval),
            )
        return None

    def _bend_detection(
        self,
        note: Note,
        features: TechniqueFeatures,
    ) -> TechniqueDetection | None:
        if note.duration < 0.18:
            return None
        margin = min(note.duration * 0.15, 0.08)
        _, contour = self._window(
            features,
            note.start + margin,
            note.start + note.duration - margin,
        )
        contour = self._smooth(contour)
        contour = contour[np.isfinite(contour)]
        contour = contour[np.abs(contour - note.midi) <= 4.0]
        if contour.size < 10:
            return None
        fifth = max(2, contour.size // 5)
        early = float(np.median(contour[:fifth]))
        peak = float(np.percentile(contour[fifth:], 90))
        late = float(np.median(contour[-fifth:]))
        rise = peak - early
        sustained_rise = late - early
        differences = np.diff(contour)
        upward_consistency = float(np.mean(differences >= -0.05))
        if (
            self.min_bend_semitones <= rise <= 3.5
            and sustained_rise >= self.min_bend_semitones * 0.65
            and upward_consistency >= 0.60
        ):
            confidence = min(
                0.95,
                0.42
                + 0.24 * min(1.0, rise / 2.0)
                + 0.20 * upward_consistency
                + 0.14 * min(1.0, sustained_rise / rise),
            )
            return TechniqueDetection(
                GuitarTechnique.BEND,
                note,
                confidence,
                pitch_change_semitones=rise,
            )
        return None

    def _vibrato_detection(
        self,
        note: Note,
        features: TechniqueFeatures,
    ) -> TechniqueDetection | None:
        if note.duration < 0.35:
            return None
        margin = min(note.duration * 0.15, 0.08)
        times, contour = self._window(
            features,
            note.start + margin,
            note.start + note.duration - margin,
        )
        finite = np.isfinite(contour) & (np.abs(contour - note.midi) <= 2.0)
        times = times[finite]
        contour = contour[finite]
        if contour.size < 18:
            return None
        coefficients = np.polyfit(times, contour, 1)
        residual = contour - np.polyval(coefficients, times)
        residual -= float(np.mean(residual))
        amplitude_cents = float(np.std(residual) * 100.0)
        if not 8.0 <= amplitude_cents <= 65.0:
            return None
        frame_step = float(np.median(np.diff(times)))
        if not math.isfinite(frame_step) or frame_step <= 0:
            return None
        frequencies = np.fft.rfftfreq(residual.size, d=frame_step)
        power = np.abs(np.fft.rfft(residual)) ** 2
        band = (
            (frequencies >= self.vibrato_rate_hz[0])
            & (frequencies <= self.vibrato_rate_hz[1])
        )
        if not np.any(band):
            return None
        band_indices = np.flatnonzero(band)
        peak_index = int(band_indices[np.argmax(power[band])])
        rate = float(frequencies[peak_index])
        band_power = float(np.sum(power[band]))
        peak_ratio = float(power[peak_index] / band_power) if band_power > 0 else 0.0
        cycles = float(rate * float(times[-1] - times[0]))
        if peak_ratio < 0.35 or cycles < 1.5:
            return None
        confidence = float(
            min(
                0.94,
                0.42
                + 0.24 * min(1.0, amplitude_cents / 35.0)
                + 0.22 * peak_ratio
                + 0.12 * min(1.0, cycles / 4.0),
            )
        )
        return TechniqueDetection(
            GuitarTechnique.VIBRATO,
            note,
            confidence,
            pitch_change_semitones=amplitude_cents / 100.0,
        )

    def classify(
        self,
        notes: Iterable[Note],
        features: TechniqueFeatures,
    ) -> tuple[TechniqueDetection, ...]:
        """Classify frame tracks for monophonic notes with one primary label."""

        note_tuple = tuple(sorted(notes, key=lambda note: (note.start, note.midi)))
        if not note_tuple or features.frame_times.size == 0:
            return ()
        monophonic = {
            self._note_key(note): self._is_monophonic(note, note_tuple)
            for note in note_tuple
        }
        detections: dict[tuple[int, float, float], TechniqueDetection] = {}
        transition_notes: set[tuple[int, float, float]] = set()
        previous: Note | None = None
        for note in note_tuple:
            if not monophonic[self._note_key(note)]:
                previous = None
                continue
            if previous is not None:
                detection = self._transition_detection(previous, note, features)
                if detection is not None:
                    detections[detection.note_key] = detection
                    transition_notes.add(self._note_key(previous))
                    transition_notes.add(detection.note_key)
            previous = note

        for note in note_tuple:
            key = self._note_key(note)
            if not monophonic[key] or key in transition_notes:
                continue
            detection = self._bend_detection(note, features)
            if detection is None:
                detection = self._vibrato_detection(note, features)
            if detection is not None:
                detections[key] = detection
        return tuple(
            detections[key]
            for key in sorted(detections, key=lambda item: (item[1], item[0]))
        )

    def detect(
        self,
        audio: AudioData,
        notes: Iterable[Note],
        *,
        pitch_hz: np.ndarray | None = None,
        pitch_hop_length: int | None = None,
    ) -> tuple[TechniqueDetection, ...]:
        """Best-effort feature extraction and technique classification."""

        note_tuple = tuple(notes)
        if not note_tuple:
            return ()
        try:
            features = self.extract_features(
                audio,
                pitch_hz=pitch_hz,
                pitch_hop_length=pitch_hop_length,
            )
        except Exception:
            # Technique recognition is optional and must never make otherwise
            # usable pitch/TAB analysis fail on a backend or sample-rate issue.
            return ()
        return self.classify(note_tuple, features)


__all__ = ["TechniqueAnalyzer", "TechniqueFeatures"]
