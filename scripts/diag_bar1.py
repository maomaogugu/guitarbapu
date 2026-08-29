"""Examine detected top notes at specified absolute times for manual verification."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.audio.loader import load_audio
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer
from src.music.guitar import Guitar
from scripts.match_answer import _answer_targets
from src.eval.answer_tab import parse_answer_tab


def main():
    audio = load_audio('/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3')
    events = parse_answer_tab(open('/Users/youzi/Downloads/晴天_1-8小节_TAB_含击勾弦.txt', encoding='utf-8').read())
    guitar = Guitar.standard()
    az = PolyphonicAudioAnalyzer(attack_weight=0.35, harmonic_ratio=0.58, log_compress=True)
    notes = az.detect_notes(audio)
    notes = sorted(notes, key=lambda n: n.start)
    # Convert each bar1 target to absolute time with offset=25.0
    bar_len = 4.0 * 60.0 / 72.0
    # Print in bar-major order
    print(f"{'abs_t':>7s} {'expected':>14s} | nearest detected notes (within ±0.5s)")
    for t, s, f, measure, _ in _answer_targets(events, 72.0):
        if measure > 1: continue
        tgt = guitar.midi_at(s, f)
        abs_t = 25.0 + t
        near = [n for n in notes if abs(n.start - abs_t) <= 0.5]
        near_s = ', '.join(f"({n.start-abs_t:+.2f}s m{n.midi})" for n in near[:8])
        flag = '✓' if any(n.midi == tgt for n in near if abs(n.start-abs_t)<=0.3) else ('~' if any(abs(n.midi-tgt)==12 for n in near if abs(n.start-abs_t)<=0.3) else '✗')
        print(f"{abs_t:7.2f} s{s}f{f} m{tgt:2d} {flag} | {near_s}")


if __name__ == '__main__':
    main()
