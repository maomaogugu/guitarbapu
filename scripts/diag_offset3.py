"""Robust offset search: compare expected-note energies vs audio chroma on a dense grid."""

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
    az = PolyphonicAudioAnalyzer(attack_weight=0.0, harmonic_ratio=0.58, log_compress=True)
    strengths, rms, frame_times = az._midi_strengths(az._waveform(audio), audio.sample_rate)
    # normalize each frame to max=1
    mx = strengths.max(axis=0, keepdims=True)
    mx[mx <= 0] = 1.0
    norm = strengths / mx

    targets = _answer_targets(events, 72.0)
    bar_length = 4.0 * 60.0 / 72.0
    span = targets[-1][0] + bar_length

    scores = []
    for offset in np.arange(0.0, audio.duration - span, 0.125):
        total = 0.0
        for time, string, fret, _m, _t in targets:
            tgt = guitar.midi_at(string, fret)
            idx = tgt - az.min_midi
            if idx < 0 or idx >= norm.shape[0]:
                continue
            abs_t = offset + time
            mask = (frame_times >= abs_t - 0.06) & (frame_times <= abs_t + 0.18)
            if not mask.any():
                continue
            total += float(norm[idx, mask].max())
        scores.append((total, offset))

    scores.sort(reverse=True)
    print("Top-10 chroma-coincidence offsets:")
    for s, o in scores[:10]:
        print(f"  offset={o:7.3f} score={s:.2f} avg={s/len(targets):.3f}")


if __name__ == '__main__':
    main()
