"""Score strict recall across all plausible offsets to find the true bar-1 start."""

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
    notes = az.detect_notes(audio)
    targets = _answer_targets(events, 72.0)
    bar_length = 4.0 * 60.0 / 72.0
    starts = np.asarray([n.start for n in notes], dtype=float)
    midis = np.asarray([n.midi for n in notes], dtype=float)
    span = targets[-1][0] + bar_length

    print(f"notes={len(notes)} targets={len(targets)} span={span:.2f}")
    # tight exact-midi hits at each offset
    results = []
    for offset in np.arange(0.0, audio.duration - span, 0.25):
        hits = 0
        midi_hits = 0
        for time, string, fret, _m, _t in targets:
            tgt = guitar.midi_at(string, fret)
            mask = np.abs(starts - (offset + time)) <= 0.30
            if mask.any():
                pool = midis[mask]
                if np.isin(pool, (tgt,)).any():
                    hits += 1
                    midi_hits += 1
                elif np.isin(pool, (tgt - 12, tgt + 12)).any():
                    midi_hits += 1
        results.append((hits, midi_hits, round(offset, 2)))
    results.sort(reverse=True)
    print("Top 15 offsets by exact-midi hits:")
    for h, mh, o in results[:15]:
        print(f"  offset={o:7.2f} exact={h} strict-fallback-midi={mh}")


if __name__ == '__main__':
    main()
