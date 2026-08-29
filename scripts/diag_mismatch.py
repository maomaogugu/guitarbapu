"""For each answer target miss, show which midis were actually detected nearby."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.match_answer import _answer_targets  # noqa: E402
from src.audio.loader import load_audio  # noqa: E402
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer  # noqa: E402
from src.eval.answer_tab import parse_answer_tab  # noqa: E402
from src.music.guitar import Guitar  # noqa: E402

AUDIO = Path("/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3")
ANSWER = Path("/Users/youzi/Downloads/晴天_1-8小节_TAB_含击勾弦.txt")
OFFSET = 25.0
BARS = 8
TOL = 0.3


def main() -> int:
    audio = load_audio(AUDIO)
    events = parse_answer_tab(ANSWER.read_text(encoding="utf-8"))
    events = [e for e in events if e.measure <= BARS]
    targets = _answer_targets(events, 72.0)
    guitar = Guitar.standard()

    analyzer = PolyphonicAudioAnalyzer(
        attack_weight=0.35, harmonic_ratio=0.58, log_compress=True
    )
    analysis = analyzer.analyze(audio)
    notes = list(analysis.notes)

    counts = Counter()
    print("misses:")
    for time, string, fret, measure, technique in targets:
        midi = guitar.midi_at(string, fret)
        moment = time + OFFSET
        near = [n.midi for n in notes if abs(n.start - moment) <= TOL]
        if midi in near:
            counts["midi_PRESENT"] += 1
            continue
        lows = [m for m in near if midi - m == 12]
        highs = [m for m in near if m - midi == 12]
        if lows:
            cause = "LOW_OCTAVE"
        elif highs:
            cause = "HIGH_OCTAVE"
        elif near:
            cause = "OTHER_MIDI"
        else:
            cause = "SILENCE"
        counts[cause] += 1
        print(
            f"  bar{measure} t={moment:.2f}s s{string}f{fret} midi{midi} "
            f"{cause} near={sorted(near)}"
        )
    print("\ncounts:", dict(counts), "total:", len(targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
