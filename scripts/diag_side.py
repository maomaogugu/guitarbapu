"""Side-by-side expected vs detected events for bars 1-8 at a given offset."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    notes_sorted = sorted(notes, key=lambda n: n.start)

    print("== EXPECTED (aligned at offset) vs DETECTED notes within ±0.5s ==")
    for time, string, fret, measure, _tech in targets:
        abs_t = offset + time
        tgt = guitar.midi_at(string, fret)
        near = [n for n in notes_sorted if abs(n.start - abs_t) <= 0.5]
        near_s = [(round(n.start - abs_t, 3), n.midi, round(n.confidence, 2)) for n in near]
        print(f"m{measure:1d} t{time:6.2f} expect(s{string} f{fret}={tgt}) abs={abs_t:6.2f} | detected: {near_s[:6]}")


if __name__ == '__main__':
    main()
