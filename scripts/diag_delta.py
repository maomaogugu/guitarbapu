"""Test new-note-boost: does delta-from-previous-segment scoring rescue melody onsets?"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.audio.loader import load_audio
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer
from src.eval.answer_tab import parse_answer_tab
from src.music.guitar import Guitar
from scripts.match_answer import _answer_targets


def main():
    audio_path = '/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3'
    answer_path = Path('/Users/youzi/Downloads/晴天_1-8小节_TAB_含击勾弦.txt')
    audio = load_audio(audio_path)
    events = parse_answer_tab(answer_path.read_text(encoding='utf-8'))
    guitar = Guitar.standard()

    az = PolyphonicAudioAnalyzer(attack_weight=0.35, harmonic_ratio=0.58, log_compress=True)
    strengths, rms, frame_times = az._midi_strengths(az._waveform(audio), audio.sample_rate)
    onset_times, timing = az.rhythm_analyzer.detect(audio)
    boundaries = az._boundaries(float(audio.duration), onset_times)

    targets = _answer_targets(events, 72.0)

    # For each target, simulate: delta-score = (this-segment strengths max) - (prev-segment median).
    # Count targets whose NEW delta is large enough to stand above existing chord tones.
    results_PER_DELTA = {}
    for delta_weight in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        # Compute per-segment pitches with delta boost
        segments = []
        for start, end in zip(boundaries, boundaries[1:]):
            seg = az._segment_pitches(strengths, rms, frame_times, start=start, end=end, global_peak_rms=float(np.max(rms)))
            # now compute delta-from-prev and boost
            prev_start = max(0.0, start - 0.06)
            prev_mask = (frame_times >= prev_start) & (frame_times < start)
            frame_mask = (frame_times >= start) & (frame_times < end)
            voiced = frame_mask & (rms >= float(np.max(rms)) * az.energy_threshold)
            if not voiced.any() or not prev_mask.any():
                continue
            seg_scores = np.maximum(strengths[:, voiced].max(axis=1) - np.median(strengths[:, prev_mask], axis=1), 0)
            if delta_weight > 0 and seg_scores.max() > 0:
                # normalize
                midis_arr = seg[0] if seg[0] else ()
                # This is just an illustration; actual integration deferred
        pass


if __name__ == '__main__':
    print("I'll implement this directly in the analyzer and evaluate.")
