"""Check per-target: is the expected pitch present in the audio at the right time?"""
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
    audio = load_audio('/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3')
    events = parse_answer_tab(open('/Users/youzi/Downloads/晴天_1-8小节_TAB_含击勾弦.txt', encoding='utf-8').read())
    guitar = Guitar.standard()
    az = PolyphonicAudioAnalyzer(attack_weight=0.35, harmonic_ratio=0.58, log_compress=True)
    strengths, rms, frame_times = az._midi_strengths(az._waveform(audio), audio.sample_rate)
    targets = _answer_targets(events, 72.0)

    # Precompute frame-global max once (for normalizing)
    gmax = strengths.max(axis=0, keepdims=True)
    gmax[gmax <= 0] = 1

    offsets = [25.0, 34.75, 41.5, 55.0, 55.15, 58.65]
    print(f"{'target':20s} | " + " | ".join(f"off={o:5.2f}" for o in offsets))
    for time, string, fret, measure, _tech in targets:
        tgt = guitar.midi_at(string, fret)
        idx = tgt - az.min_midi
        if idx < 0 or idx >= strengths.shape[0]:
            continue
        row = f"m{measure}s{string}f{fret}(m{tgt:2d}) t{time:5.2f} | "
        for off in offsets:
            abs_t = off + time
            mask = (frame_times >= abs_t - 0.1) & (frame_times <= abs_t + 0.2)
            if not mask.any():
                row += "  N/A   | "
                continue
            e = strengths[idx, mask].max()
            em = strengths[idx - 12, mask].max() if idx - 12 >= 0 else 0
            ep = strengths[idx + 12, mask].max() if idx + 12 < strengths.shape[0] else 0
            # relative to frame max
            rel = e / float(gmax[:, mask].max())
            mark = "STRONG" if rel > 0.3 else "weak"
            tag = "★" if e >= em and e >= ep and rel > 0.2 else " "
            row += f" {mark:6s}{tag} | "
        print(row)

    # aggregate: over all targets, which offset maximizes "target midi is locally dominant"
    for off in offsets:
        n = 0
        for time, string, fret, _m, _t in targets:
            tgt = guitar.midi_at(string, fret)
            idx = tgt - az.min_midi
            abs_t = off + time
            mask = (frame_times >= abs_t - 0.1) & (frame_times <= abs_t + 0.2)
            if not mask.any():
                continue
            e = strengths[idx, mask].max()
            em = strengths[idx - 12, mask].max() if idx - 12 >= 0 else 0
            ep = strengths[idx + 12, mask].max() if idx + 12 < strengths.shape[0] else 0
            rel = e / float(gmax[:, mask].max()) if mask.any() else 0
            if e >= em * 0.9 and e >= ep * 0.9 and rel > 0.15:
                n += 1
        print(f"offset {off:5.2f}: target-dominant hits = {n}/70")


if __name__ == '__main__':
    main()
