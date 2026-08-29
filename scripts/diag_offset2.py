"""Fine offset scan using the exact strict/midi scoring formulas."""

from __future__ import annotations

import json
from pathlib import Path
import sys

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
    bar_length = 4.0 * 60.0 / 72.0
    starts = np.asarray([n.start for n in notes], dtype=float)
    midis = np.asarray([n.midi for n in notes], dtype=float)
    span = targets[-1][0] + bar_length

    def score(offset, tol, octave):
        hits = 0
        for time, string, fret, _m, _t in targets:
            tgt = guitar.midi_at(string, fret)
            mask = np.abs(starts - (offset + time)) <= tol
            if not mask.any():
                continue
            if octave:
                if np.isin(midis[mask], (tgt, tgt - 12, tgt + 12)).any():
                    hits += 1
            else:
                if np.isin(midis[mask], (tgt,)).any():
                    hits += 1
        return hits

    # Fine scan 0.1s steps, exact midi
    best = []
    for offset in np.arange(0.0, audio.duration - span, 0.1):
        hits = score(offset, 0.225, False)
        best.append((hits, round(offset, 1)))
    best.sort(reverse=True)
    print("top exact-midi offsets:")
    for h, o in best[:15]:
        print(f"  offset={o:6.1f} hits={h:3d} ({h/len(targets):.3f})")


if __name__ == '__main__':
    main()
