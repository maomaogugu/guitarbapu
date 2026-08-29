"""Experimental CQT-based polyphonic guitar pitch analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..music.chord import Chord
from ..music.note import Note
from ..music.timing import TimingInfo
from .analyzer import AudioAnalysis
from .librosa_compat import import_librosa
from .loader import AudioData
from .rhythm import RhythmAnalyzer


@dataclass(frozen=True)
class _PitchSegment:
    start: float
    end: float
    midis: tuple[int, ...]
    confidences: tuple[float, ...]


class PolyphonicAudioAnalyzer:
    """Detect conservative simultaneous pitch groups with a CQT spectrum.

    This is an experimental deterministic baseline. It deliberately returns
    fewer high-confidence pitches instead of treating every overtone as a
    fretted note.
    """

    _HARMONIC_INTERVALS = frozenset((12, 19, 24, 28, 31, 34, 36))

    def __init__(
        self,
        *,
        min_midi: int = 40,
        max_midi: int = 88,
        hop_length: int = 512,
        bins_per_semitone: int = 3,
        energy_threshold: float = 0.08,
        relative_pitch_threshold: float = 0.24,
        harmonic_ratio: float = 0.58,
        max_polyphony: int = 6,
        min_segment_duration: float = 0.10,
        beat_subdivision: int = 4,
        attack_weight: float = 0.0,
        harmonic_salience: float = 0.0,
        pre_emphasis: float = 0.0,
        log_compress: bool = False,
        track_note_onsets: bool = False,
        note_onset_sensitivity: float = 0.12,
    ) -> None:
        if not 0 <= min_midi < max_midi <= 127:
            raise ValueError("MIDI range must be within 0..127")
        if hop_length < 1 or bins_per_semitone < 1:
            raise ValueError("hop_length and bins_per_semitone must be positive")
        if not 0 <= energy_threshold <= 1:
            raise ValueError("energy_threshold must be between 0 and 1")
        if not 0 < relative_pitch_threshold <= 1:
            raise ValueError("relative_pitch_threshold must be between 0 and 1")
        if not 0 < harmonic_ratio <= 1:
            raise ValueError("harmonic_ratio must be between 0 and 1")
        if not 1 <= max_polyphony <= 6:
            raise ValueError("max_polyphony must be between 1 and 6")
        if min_segment_duration <= 0:
            raise ValueError("min_segment_duration must be positive")
        if not 0 <= attack_weight <= 1:
            raise ValueError("attack_weight must be between 0 and 1")
        if not 0 <= harmonic_salience <= 1:
            raise ValueError("harmonic_salience must be between 0 and 1")
        if not 0 <= pre_emphasis < 1:
            raise ValueError("pre_emphasis must be in [0, 1]")
        if not 0 < note_onset_sensitivity <= 1:
            raise ValueError("note_onset_sensitivity must be between 0 and 1")

        self.min_midi = int(min_midi)
        self.max_midi = int(max_midi)
        self.hop_length = int(hop_length)
        self.bins_per_semitone = int(bins_per_semitone)
        self.energy_threshold = float(energy_threshold)
        self.relative_pitch_threshold = float(relative_pitch_threshold)
        self.harmonic_ratio = float(harmonic_ratio)
        self.max_polyphony = int(max_polyphony)
        self.min_segment_duration = float(min_segment_duration)
        self.attack_weight = float(attack_weight)
        self.harmonic_salience = float(harmonic_salience)
        self.pre_emphasis = float(pre_emphasis)
        self.log_compress = bool(log_compress)
        self.track_note_onsets = bool(track_note_onsets)
        self.note_onset_sensitivity = float(note_onset_sensitivity)
        self.rhythm_analyzer = RhythmAnalyzer(
            hop_length=self.hop_length,
            subdivision=beat_subdivision,
        )

    def _waveform(self, audio: AudioData) -> np.ndarray:
        waveform = np.asarray(audio.waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if waveform.ndim != 1:
            raise ValueError("audio waveform must be one-dimensional")
        if self.pre_emphasis > 0 and waveform.size > 1:
            emphasized = waveform[1:] - self.pre_emphasis * waveform[:-1]
            waveform = np.concatenate((waveform[:1], emphasized))
        return waveform

    def _midi_strengths(
        self, waveform: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        midi_count = self.max_midi - self.min_midi + 1
        if waveform.size == 0:
            return (
                np.empty((midi_count, 0), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )
        try:
            librosa = import_librosa()
            cqt = librosa.cqt(
                waveform,
                sr=sample_rate,
                hop_length=self.hop_length,
                fmin=float(librosa.midi_to_hz(self.min_midi)),
                n_bins=midi_count * self.bins_per_semitone,
                bins_per_octave=12 * self.bins_per_semitone,
            )
            magnitude = np.abs(cqt).astype(np.float32)
            strengths = magnitude.reshape(
                midi_count,
                self.bins_per_semitone,
                magnitude.shape[1],
            ).max(axis=1)
            rms = np.asarray(
                librosa.feature.rms(
                    y=waveform,
                    frame_length=2048,
                    hop_length=self.hop_length,
                    center=True,
                )[0],
                dtype=np.float32,
            )
            frame_times = np.asarray(
                librosa.frames_to_time(
                    np.arange(strengths.shape[1]),
                    sr=sample_rate,
                    hop_length=self.hop_length,
                ),
                dtype=np.float32,
            )
        except Exception as exc:
            raise RuntimeError("librosa could not compute a polyphonic CQT") from exc

        frame_count = min(strengths.shape[1], rms.size, frame_times.size)
        return (
            strengths[:, :frame_count],
            rms[:frame_count],
            frame_times[:frame_count],
        )

    def _boundaries(
        self, duration: float, onset_times: tuple[float, ...]
    ) -> tuple[float, ...]:
        boundaries = [0.0]
        for onset in sorted(set(float(value) for value in onset_times)):
            if onset <= 0 or onset >= duration:
                continue
            if onset - boundaries[-1] >= self.min_segment_duration:
                boundaries.append(onset)
        if duration - boundaries[-1] < self.min_segment_duration and len(boundaries) > 1:
            boundaries.pop()
        boundaries.append(duration)
        return tuple(boundaries)

    def _segment_pitches(
        self,
        strengths: np.ndarray,
        rms: np.ndarray,
        frame_times: np.ndarray,
        *,
        start: float,
        end: float,
        global_peak_rms: float,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        frame_mask = (frame_times >= start) & (frame_times < end)
        voiced = frame_mask & (rms >= global_peak_rms * self.energy_threshold)
        if not np.any(voiced):
            return ((), ())
        scores = np.median(strengths[:, voiced], axis=1)
        if self.attack_weight > 0:
            attack_window = min(0.12, max(0.03, (end - start) * 0.35))
            attack_mask = voiced & (frame_times < start + attack_window)
            if np.any(attack_mask):
                attack_scores = np.max(strengths[:, attack_mask], axis=1)
                scores = (
                    (1.0 - self.attack_weight) * scores
                    + self.attack_weight * attack_scores
                )
        scores = np.maximum(scores - float(np.median(scores)), 0.0)
        if self.log_compress:
            scores = np.log1p(scores * 40.0)
        if self.harmonic_salience > 0:
            salience = scores.copy()
            for shift, weight in ((12, 0.6), (19, 0.4), (24, 0.3), (28, 0.2)):
                shifted = np.zeros_like(scores)
                shifted[:-shift] = scores[shift:]
                salience = salience + weight * shifted
            scores = (
                (1.0 - self.harmonic_salience) * scores
                + self.harmonic_salience * salience
            )
        maximum = float(np.max(scores)) if scores.size else 0.0
        if not math.isfinite(maximum) or maximum <= 0:
            return ((), ())

        minimum = maximum * self.relative_pitch_threshold
        candidates: list[int] = []
        for index, score in enumerate(scores):
            if score < minimum:
                continue
            left = scores[index - 1] if index > 0 else -1.0
            right = scores[index + 1] if index + 1 < scores.size else -1.0
            if score >= left and score >= right:
                candidates.append(index)

        accepted: list[int] = []
        for index in candidates:
            midi = self.min_midi + index
            harmonic = False
            for lower_index in accepted:
                lower_midi = self.min_midi + lower_index
                if (
                    midi - lower_midi in self._HARMONIC_INTERVALS
                    and scores[index] <= scores[lower_index] * self.harmonic_ratio
                ):
                    harmonic = True
                    break
            if not harmonic:
                accepted.append(index)

        accepted = sorted(
            accepted,
            key=lambda index: (-float(scores[index]), self.min_midi + index),
        )[: self.max_polyphony]
        accepted.sort()
        midis = tuple(self.min_midi + index for index in accepted)
        confidences = tuple(
            max(0.0, min(1.0, float(scores[index] / maximum)))
            for index in accepted
        )
        return (midis, confidences)

    def _segments(
        self,
        audio: AudioData,
        onset_times: tuple[float, ...],
        strengths: np.ndarray,
        rms: np.ndarray,
        frame_times: np.ndarray,
        global_peak_rms: float,
    ) -> tuple[_PitchSegment, ...]:
        if rms.size == 0:
            return ()

        segments: list[_PitchSegment] = []
        boundaries = self._boundaries(float(audio.duration), onset_times)
        for start, end in zip(boundaries, boundaries[1:]):
            midis, confidences = self._segment_pitches(
                strengths,
                rms,
                frame_times,
                start=start,
                end=end,
                global_peak_rms=global_peak_rms,
            )
            if not midis:
                continue
            segment = _PitchSegment(start, end, midis, confidences)
            if segments and segments[-1].midis == segment.midis:
                previous = segments[-1]
                combined = tuple(
                    (left + right) / 2.0
                    for left, right in zip(previous.confidences, confidences)
                )
                segments[-1] = _PitchSegment(
                    previous.start,
                    end,
                    midis,
                    combined,
                )
            else:
                segments.append(segment)
        return tuple(segments)

    def _note_onset_events(
        self,
        strengths: np.ndarray,
        rms: np.ndarray,
        frame_times: np.ndarray,
        global_peak_rms: float,
    ) -> tuple[Note, ...]:
        """Track pitch-band energy rises to catch fingerstyle note attacks.

        Global onset detection tends to miss soft melody plucks that occur
        while chord tones still ring.  Tracking an exponential-decay envelope
        per MIDI band lets every string re-attack create its own event, even
        when other notes sustain underneath.
        """

        if strengths.size == 0 or frame_times.size < 2:
            return ()
        frame_dt = float(frame_times[-1] - frame_times[0]) / (
            frame_times.size - 1
        )
        if frame_dt <= 0:
            return ()
        # Guitar notes decay audibly; refresh the envelope a little slower
        # than a natural string decay so only genuine re-attacks trigger.
        decay = math.exp(-frame_dt / 0.55)
        refractory = 0.09  # seconds between two onsets on the same string
        min_duration = 0.05
        notes: list[Note] = []
        voiced = rms >= global_peak_rms * self.energy_threshold
        for index in range(strengths.shape[0]):
            series = strengths[index]
            peak = float(np.max(series))
            floor = float(np.median(series))
            if not math.isfinite(peak) or peak <= floor:
                continue
            threshold = floor + self.note_onset_sensitivity * (peak - floor)
            envelope = 0.0
            last_onset = -math.inf
            rising_start = 0
            onset_frames: list[int] = []
            for frame, value in enumerate(series):
                trigger = max(threshold, envelope * 1.18)
                if (
                    voiced[frame]
                    and value >= trigger
                    and frame_times[frame] - last_onset >= refractory
                ):
                    last_onset = float(frame_times[rising_start])
                    onset_frames.append(rising_start)
                if value < envelope:
                    rising_start = frame
                envelope = max(value, envelope * decay)
            midi = self.min_midi + index
            for onset_index, frame in enumerate(onset_frames):
                start = float(frame_times[frame])
                level = float(series[frame])
                end = float(frame_times[-1])
                for follow in range(frame + 1, series.size):
                    if series[follow] <= level * 0.35:
                        end = float(frame_times[follow])
                        break
                confidence = max(0.0, min(1.0, level / max(peak, 1e-9)))
                notes.append(
                    Note(
                        midi,
                        start=start,
                        duration=max(min_duration, end - start),
                        confidence=confidence,
                    )
                )
        notes.sort(key=lambda note: (note.start, note.midi))
        return tuple(notes)

    def detect_events(
        self, audio: AudioData
    ) -> tuple[
        tuple[Note, ...],
        tuple[Chord, ...],
        tuple[float, ...],
        TimingInfo,
    ]:
        """Return simultaneous notes, chord labels, onset times, and timing."""

        if audio.sample_rate <= 0:
            raise ValueError("audio.sample_rate must be positive")
        if audio.sample_rate / 2 <= Note(self.max_midi).frequency:
            raise ValueError(
                "audio sample rate is too low for the configured polyphonic range"
            )
        onset_times, timing = self.rhythm_analyzer.detect(audio)
        waveform = self._waveform(audio)
        strengths, rms, frame_times = self._midi_strengths(
            waveform, audio.sample_rate
        )
        global_peak_rms = float(np.max(rms)) if rms.size else 0.0
        if not math.isfinite(global_peak_rms) or global_peak_rms <= 0:
            global_peak_rms = 0.0
        notes: list[Note] = []
        chords: list[Chord] = []
        for segment in self._segments(
            audio, onset_times, strengths, rms, frame_times, global_peak_rms
        ):
            duration = max(0.0, segment.end - segment.start)
            segment_notes = tuple(
                Note(
                    midi,
                    start=segment.start,
                    duration=duration,
                    confidence=confidence,
                )
                for midi, confidence in zip(segment.midis, segment.confidences)
            )
            notes.extend(segment_notes)
            if len(segment_notes) >= 2:
                chords.append(
                    Chord.from_midis(
                        segment.midis,
                        start=segment.start,
                        duration=duration,
                        confidence=float(np.mean(segment.confidences)),
                    )
                )
        if self.track_note_onsets and global_peak_rms > 0:
            extra = self._note_onset_events(
                strengths, rms, frame_times, global_peak_rms
            )
            if extra:
                notes.extend(extra)
                notes.sort(key=lambda note: (note.start, note.midi))
        return (tuple(notes), tuple(chords), onset_times, timing)

    def detect_notes(self, audio: AudioData) -> tuple[Note, ...]:
        notes, _, _, _ = self.detect_events(audio)
        return notes

    def analyze(self, audio: AudioData) -> AudioAnalysis:
        """Return overlapping notes and chord events for the existing pipeline."""

        notes, chords, onset_times, timing = self.detect_events(audio)
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
                "analysis_mode": "polyphonic",
                "onset_times": np.asarray(onset_times, dtype=np.float32),
                "tempo_bpm": timing.tempo_bpm,
            },
            notes=notes,
            raw_notes=notes,
            rhythm=rhythm,
            chords=chords,
        )


__all__ = ["PolyphonicAudioAnalyzer"]
