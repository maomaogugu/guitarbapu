"""Diagnostic: compare top-voice detected melody vs expected melody per measure."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.loader import load_audio
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer
from src.eval.answer_tab import parse_answer_tab
from src.music.guitar import Guitar


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

    # expected: per measure, sorted by time, list of (time, midi)
    bar_length = 4.0 * 60.0 / 72.0
    expected = {}
    for e in events:
        fraction = e.order / e.span if e.span else 0.0
        t = (e.measure - 1) * bar_length + fraction * bar_length
        expected.setdefault(e.measure, []).append((round(t, 3), guitar.midi_at(e.string, e.fret)))
    for m in sorted(expected):
        expected[m].sort()

    offset = 25.0
    # detected: group notes by measure using offset + grid
    detected = {}
    for n in notes:
        m = int((n.start - offset) // bar_length) + 1
        if 1 <= m <= 8:
            detected.setdefault(m, []).append((round(n.start, 3), n.midi))
    for m in detected:
        detected[m].sort()

    for m in range(1, 9):
        exp = expected.get(m, [])
        det = detected.get(m, [])
        exp_midi = [e[1] for e in exp]
        det_midi = [d[1] for d in det]
        print(f"--- m{m} expected({len(exp)}): {exp_midi}")
        print(f"    detected({len(det)}): {det_midi[:40]}")


if __name__ == '__main__':
    main()
