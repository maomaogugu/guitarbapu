"""Full chroma correlation to verify the answer really lives at offset≈25."""
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
    bar_len = 4.0 * 60.0 / 72.0
    span = targets[-1][0] + bar_len

    # For each target, precompute expected PC
    pcs = [(t, guitar.midi_at(s, f) % 12) for t, s, f, _, _ in targets]

    results = []
    for offset in np.arange(0.0, audio.duration - span, 0.25):
        hits = 0
        for t, pc in pcs:
            abs_t = offset + t
            mask = (frame_times >= abs_t - 0.15) & (frame_times <= abs_t + 0.3)
            if not mask.any():
                continue
            section = strengths[:, mask]
            # get top-4 pitch class strengths
            mean_str = section.mean(axis=1)
            # check if any midi with this pitch class is in top-4
            top_idx = np.argsort(mean_str)[::-1][:6]
            top_pcs = set(int(m) % 12 for m in (top_idx + az.min_midi))
            if pc in top_pcs:
                hits += 1
        results.append((hits, offset))
    results.sort(reverse=True)
    print("Top 10 offsets by pitch-class energy containment:")
    for h, o in results[:10]:
        print(f"  offset={o:6.2f} hits={h}/70")


if __name__ == '__main__':
    main()
