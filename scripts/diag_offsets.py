"""Print per-target presence of expected midi at several offsets, first 3 bars."""
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
    notes = az.detect_notes(audio)
    note_times = np.asarray([n.start for n in notes])
    note_midis = np.asarray([n.midi for n in notes])
    targets = _answer_targets(events, 72.0)

    offsets = [25.0, 34.75, 41.5, 55.0, 55.15, 58.65]
    # restrict to targets in bar 1 (measure == 1)
    bar1 = [(t, s, f, m) for t, s, f, m, _ in targets if m == 1]
    print(f"bar1 has {len(bar1)} targets")
    header = "target(midi)   " + "  ".join(f"off={o:5.2f}" for o in offsets)
    print(header)
    for time, string, fret, measure in bar1:
        tgt = guitar.midi_at(string, fret)
        row = f"s{string}f{fret}=m{tgt:2d} t{time:5.2f} "
        for off in offsets:
            t_abs = off + time
            mask = np.abs(note_times - t_abs) <= 0.30
            if mask.any():
                pool = note_midis[mask]
                if tgt in pool:
                    row += "  EXACT   "
                elif tgt - 12 in pool or tgt + 12 in pool:
                    row += "  OCTAVE  "
                else:
                    row += "  none    "
            else:
                row += "  none    "
        print(row)


if __name__ == '__main__':
    main()
