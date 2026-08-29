"""POC: lowest-fret greedy fingering (no DP). Compare strict recall vs baseline."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.loader import load_audio
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer
from src.eval.answer_tab import parse_answer_tab
from src.music.guitar import Guitar
from src.music.tab_generator import TabGenerator
from scripts.match_answer import _answer_targets


class GreedyLowFretOptimizer:
    """Choose the lowest-fret position per note; break ties by string order."""

    def __init__(self, guitar):
        self.guitar = guitar

    def optimize_groups(self, groups):
        results = []
        for group in groups:
            assignment = []
            for note in group:
                pos = self.guitar.find_positions(note) if hasattr(self.guitar, 'find_positions') else None
                # fall back to fretboard
                if pos is None or len(pos) == 0:
                    from src.music.fretboard import Fretboard
                    pos = Fretboard(self.guitar).find_positions(note)
                if not pos:
                    assignment.append(None)
                    continue
                best = min(pos, key=lambda p: (p.fret, -p.string))
                assignment.append(best)
            results.append(tuple(assignment))
        return tuple(results)


def score_with(tablature, targets, offset=25.0):
    hits = 0
    tab_e = list(tablature.events)
    for t, s, f, _, _ in targets:
        abs_t = offset + t
        if any(abs(e.start - abs_t) <= 0.3 and e.string == s and e.fret == f for e in tab_e):
            hits += 1
    return hits, len(targets), len(tab_e)


def main():
    audio_path = '/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3'
    answer_path = Path('/Users/youzi/Downloads/晴天_1-8小节_TAB_含击勾弦.txt')
    audio = load_audio(audio_path)
    answer_events = parse_answer_tab(answer_path.read_text(encoding='utf-8'))
    targets = _answer_targets(answer_events, 72.0)

    az = PolyphonicAudioAnalyzer(attack_weight=0.35, harmonic_ratio=0.58, log_compress=True)
    an = az.analyze(audio)

    print(f"targets={len(targets)}, notes={len(an.notes)}")

    guitar = Guitar.standard()
    greedy = GreedyLowFretOptimizer(guitar)
    tg = TabGenerator(guitar=guitar)
    tg.optimizer = greedy

    from src.music.tab import Tablature
    # Bypass optimizer entirely: build tab but override assignments
    import io
    from src.music.tab import TabEvent
    quantized = tg._quantized_notes(an)
    groups = tg._group_quantized(quantized)
    position_groups = greedy.optimize_groups(tuple(tuple(item.note for item in g) for g in groups))

    events_out = []
    for group, positions in zip(groups, position_groups):
        for item, pos in zip(group, positions):
            if pos is None:
                continue
            events_out.append(TabEvent(
                string=pos.string, fret=pos.fret,
                start=item.note.start, duration=item.note.duration,
                note=item.note,
                start_beat=item.start_beat or 0,
                duration_beats=item.duration_beats or 0.25,
                measure=1,
            ))

    hits = 0
    for t, s, f, m, _ in targets:
        abs_t = 25.0 + t
        if any(abs(e.start - abs_t) <= 0.3 and e.string == s and e.fret == f for e in events_out):
            hits += 1
    print(f"Greedy low-fret: {hits}/{len(targets)} = {hits/len(targets):.3f}")

    # Also try: transpose detected down 12 (octave correction POC)
    from src.music.note import Note
    downshifted = []
    for n in an.notes:
        downshifted.append((n.start, n.midi - 12))
    events12 = []
    for n in an.notes:
        for string in range(1, 7):
            open_midi = guitar.midi_at(string, 0)
            fret = (n.midi - 12) - open_midi
            if 0 <= fret <= 24:
                events12.append((n.start, string, fret))
    hits12 = 0
    for t, s, f, m, _ in targets:
        abs_t = 25.0 + t
        target_midi = guitar.midi_at(s, f)
        if any(abs(x - abs_t) <= 0.3 and xs == s and xf == f for x, xs, xf in events12):
            hits12 += 1
    print(f"Octave-down unfold: {hits12}/{len(targets)}")


if __name__ == '__main__':
    main()
