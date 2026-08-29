"""Track top-voice per beat over time, find where answer melody pattern best fits."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.audio.loader import load_audio
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer
from src.eval.answer_tab import parse_answer_tab
from src.music.guitar import Guitar


def main():
    audio = load_audio('/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3')
    events = parse_answer_tab(open('/Users/youzi/Downloads/晴天_1-8小节_TAB_含击勾弦.txt', encoding='utf-8').read())
    guitar = Guitar.standard()
    az = PolyphonicAudioAnalyzer(attack_weight=0.35, harmonic_ratio=0.58, log_compress=True)
    strengths, rms, frame_times = az._midi_strengths(az._waveform(audio), audio.sample_rate)

    # Expected melody: take e-string notes (highest voice) from answer
    mel = [(e.measure, e.order, e.string, e.fret, guitar.midi_at(e.string, e.fret)) for e in events]
    bar_len = 4.0 * 60.0 / 72.0

    # Build expected note sequence with relative times (bar1 starts at 0)
    seq = []
    for m, order, string, fret, midi in mel:
        span = next(x.span for x in events if x.measure == m and x.string == string)
        t = (m - 1) * bar_len + (order / span if span else 0.0) * bar_len
        seq.append((t, midi))
    seq.sort()

    # For every offset (step 0.05), compute how many expected midis have dominant local energy in audio
    best = []
    for offset in np.arange(0.0, audio.duration - seq[-1][0] - 1, 0.1):
        hits = 0
        for (rel_t, midi) in seq:
            abs_t = offset + rel_t
            mask = (frame_times >= abs_t - 0.1) & (frame_times <= abs_t + 0.2)
            if not mask.any():
                continue
            idx = midi - az.min_midi
            if idx < 0 or idx >= strengths.shape[0]:
                continue
            e0 = strengths[idx, mask].max()
            # relative to frame max within window
            over = strengths[:, mask].max(axis=0)
            ratio = e0 / max(over.max(), 1e-9)
            if ratio > 0.25:
                hits += 1
        best.append((hits, offset))
    best.sort(reverse=True)
    print("Top 10 offsets by dominant-energy hits:")
    for h, o in best[:10]:
        print(f"  offset={o:6.2f} hits={h}/{len(seq)}")


if __name__ == '__main__':
    main()
