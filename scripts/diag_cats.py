"""Detailed per-target categorization of match failures."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.audio.loader import load_audio
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer
from src.eval.answer_tab import parse_answer_tab
from src.music.guitar import Guitar
from src.music.tab_generator import TabGenerator

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
    tablature = TabGenerator().generate(analyzer.analyze(audio))
    targets = _answer_targets(events, 72.0)
    offset = 25.0
    tol = 0.3

    tab = tuple((e.start, e.string, e.fret) for e in tablature.events)
    note_list = tuple((n.start, n.midi) for n in notes)

    cats = Counter()
    examples = {}
    for time, string, fret, measure, technique in targets:
        absolute = offset + time
        target_midi = guitar.midi_at(string, fret)
        ok = any(
            abs(s - absolute) <= tol and ds == string and df == fret
            for s, ds, df in tab
        )
        if ok:
            cat = "STRICT_HIT"
        else:
            exact = any(
                abs(s - absolute) <= tol and m == target_midi
                for s, m in note_list
            )
            octave = any(
                abs(s - absolute) <= tol and m in (target_midi - 12, target_midi + 12)
                for s, m in note_list
            )
            if exact:
                cat = "MIDI_OK_STR_FRET_WRONG"
            elif octave:
                cat = "OCTAVE_ONLY"
            else:
                # check near time but not within tol
                near = any(
                    m in (target_midi, target_midi - 12, target_midi + 12)
                    and abs(s - absolute) <= 0.6
                    for s, m in note_list
                )
                cat = "TIME_OFF" if near else "PITCH_MISSING"
        cats[cat] += 1
        examples.setdefault(cat, []).append((measure, string, fret, round(absolute, 2), target_midi))

    for cat, count in sorted(cats.items()):
        print(f"{cat}: {count}")
        if cat != "STRICT_HIT":
            for ex in examples[cat][:8]:
                print(f"   m{ex[0]} s{ex[1]} f{ex[2]} t={ex[3]} midi={ex[4]}")


if __name__ == '__main__':
    main()
