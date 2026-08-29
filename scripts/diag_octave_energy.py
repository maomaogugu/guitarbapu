"""Measure pitch-class energy at target midi vs midi+-12 to locate octave errors."""

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


def main() -> None:
    audio = load_audio('/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3')
    events = parse_answer_tab(open('/Users/youzi/Downloads/晴天_1-8小节_TAB_含击勾弦.txt', encoding='utf-8').read())
    guitar = Guitar.standard()
    az = PolyphonicAudioAnalyzer(attack_weight=0.35, harmonic_ratio=0.58, log_compress=True)
    strengths, rms, frame_times = az._midi_strengths(az._waveform(audio), audio.sample_rate)
    targets = _answer_targets(events, 72.0)
    offset = 25.0
    hops = 0

    n_octave_up_better = 0
    n_octave_down_better = 0
    n_target_best = 0
    total_en = 0
    print("== for each target: energy at target-midi, -12, +12 ==")
    for time, string, fret, measure, _tech in targets:
        abs_t = offset + time
        tgt = guitar.midi_at(string, fret)
        idx_t = tgt - az.min_midi
        mask = (frame_times >= abs_t - 0.05) & (frame_times <= abs_t + 0.15)
        if not mask.any():
            continue
        def en(d):
            return float(strengths[idx_t + d, mask].max()) if 0 <= idx_t + d < strengths.shape[0] else 0.0
        e0, em, ep = en(0), en(-12), en(12)
        total_en += 1
        if ep > e0 and ep > em: n_octave_up_better += 1
        elif em > e0 and em > ep: n_octave_down_better += 1
        else: n_target_best += 1

    print(f"target wins: {n_target_best}/{total_en}")
    print(f"octave-down(-12) wins: {n_octave_down_better}")
    print(f"octave-up(+12) wins:  {n_octave_up_better}")


if __name__ == '__main__':
    main()
