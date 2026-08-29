"""Examine octave-only and time-off cases in detail."""

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
    analyzer = PolyphonicAudioAnalyzer(
        attack_weight=0.35,
        relative_pitch_threshold=0.24,
        harmonic_ratio=0.58,
        log_compress=True,
    )
    notes = analyzer.detect_notes(audio)
    targets = _answer_targets(events, 72.0)
    offset = 25.0
    tol = 0.3
    notes_sorted = sorted(notes, key=lambda n: n.start)

    shown = 0
    for time, string, fret, measure, technique in targets:
        absolute = offset + time
        target_midi = guitar.midi_at(string, fret)
        near = [
            n for n in notes_sorted
            if abs(n.start - absolute) <= 0.6
            and n.midi in (target_midi - 12, target_midi + 12)
        ]
        if near and not any(
            abs(n.start - absolute) <= tol and n.midi == target_midi
            for n in notes_sorted
        ):
            print(f"m{measure} s{string} f{fret} expect midi={target_midi} abs={absolute:.2f}")
            for n in near[:5]:
                print(f"   detected midi={n.midi} t={n.start:.3f} d={n.duration:.2f} conf={n.confidence:.3f}")
            shown += 1
        if shown >= 20:
            break


if __name__ == '__main__':
    main()
