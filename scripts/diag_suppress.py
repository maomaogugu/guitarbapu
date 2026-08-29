"""Diagnose: at target times, what candidate pitches are detected, and are higher-octave melody notes suppressed as harmonics?"""
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
    offset = 25.0

    # For melody targets (string 1 with fret>0), check suppression
    print("=== String-1 melody targets (expect midis 64,67,69 etc) ===")
    for time, string, fret, measure, _tech in targets:
        if string != 1: continue
        tgt = guitar.midi_at(string, fret)
        idx = tgt - az.min_midi
        abs_t = offset + time
        mask = (frame_times >= abs_t) & (frame_times <= abs_t + 0.25)
        if not mask.any(): continue
        # what the segment detector sees: max strength per midi in window
        segment = strengths[:, mask]
        scores = np.median(segment, axis=1)
        scores = np.maximum(scores - np.median(scores), 0.0)
        if az.log_compress:
            scores = np.log1p(scores * 40.0)
        mx = scores.max()
        top = np.argsort(scores)[::-1][:8]
        # where does target rank?
        tgt_rank = None
        for rank, c in enumerate(np.argsort(scores)[::-1]):
            if c == idx:
                tgt_rank = rank
                break
        e_target = scores[idx] if 0 <= idx < len(scores) else 0
        e_low = scores[idx - 12] if idx - 12 >= 0 else 0
        e_up = scores[idx + 12] if idx + 12 < len(scores) else 0
        suppressed = e_low > 0 and e_target > 0 and e_target / max(e_low, 1e-9) < az.harmonic_ratio * 1.0
        print(f"m{measure} t{time:5.2f} tgt m{tgt} idx{idx} | score={e_target:.2f} (rank {tgt_rank}) | "
              f"one-12-down m{tgt-12}={e_low:.2f} one-12-up m{tgt+12}={e_up:.2f} | "
              f"ratio_to_low12={e_target/max(e_low,1e-9):.2f} {'SUPPR?' if suppressed else ''}")
        print(f"   top8 accepted candidates: {[(int(c + az.min_midi), round(float(scores[c]),2)) for c in top[:6]]})")


if __name__ == '__main__':
    main()
