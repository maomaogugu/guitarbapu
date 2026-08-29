"""Check the very first few expected events against audio energy across offsets."""
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
    # to be bar-1 alignment, its first note should be E string fret 3 = midi 43 (G2), a bass pickup.
    # Then G3(55), B3(59) open strings follow.
    first_few = list(targets)[:8]
    print("First 8 expected events (measure, string, fret, midi):")
    for t, s, f, m, tech in first_few:
        print(f"  m{m} s{s} f{f} → midi {guitar.midi_at(s,f)} at t=+{t:.2f}s")
    print()
    # exact-midi recall vs offset, zoomed 0-60s fine
    spans = 4.0 * 60.0 / 72.0 * 8
    candidates = []
    for offset in np.arange(0.0, 60.0, 0.05):
        hits = 0
        for time, string, fret, _m, _t in targets:
            tgt = guitar.midi_at(string, fret)
            idx = tgt - az.min_midi
            if idx < 0 or idx >= strengths.shape[0]:
                continue
            abs_t = offset + time
            mask = (frame_times >= abs_t - 0.08) & (frame_times <= abs_t + 0.2)
            if not mask.any():
                continue
            # target midi must be a local peak in the salience sense
            e = strengths[idx, mask].max()
            em1 = strengths[idx - 12, mask].max() if idx - 12 >= 0 else 0
            ep1 = strengths[idx + 12, mask].max() if idx + 12 < strengths.shape[0] else 0
            if e >= em1 and e >= ep1:
                hits += 1
        candidates.append((hits, offset))
    candidates.sort(reverse=True)
    print("Top offsets where target-midi beats both octaves (exact pitch wins):")
    for h, o in candidates[:10]:
        print(f"  offset={o:6.2f} hits={h}")


if __name__ == '__main__':
    main()
