"""Sweep fingering weights using ONLY optimize_groups, bypassing full TabGenerator."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
from src.audio.loader import load_audio
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer
from src.eval.answer_tab import parse_answer_tab
from src.music.fingering import FingeringOptimizer, FingeringWeights
from src.music.guitar import Guitar
from scripts.match_answer import _answer_targets


def main():
    audio_path = Path('/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3')
    answer_path = Path('/Users/youzi/Downloads/晴天_1-8小节_TAB_含击勾弦.txt')
    audio = load_audio(audio_path)
    answer_events = parse_answer_tab(answer_path.read_text(encoding='utf-8'))
    targets = _answer_targets(answer_events, 72.0)

    az = PolyphonicAudioAnalyzer(attack_weight=0.35, harmonic_ratio=0.58, log_compress=True)
    an = az.analyze(audio)
    # get quantized groups ontologically
    qn = an.rhythm.quantized_notes
    grouped: dict = defaultdict(list)
    for item in qn:
        key = round(item.start_beat if item.start_beat is not None else item.note.start, 6)
        grouped[key].append(item)
    keys = sorted(grouped)
    note_tuples = tuple(tuple(item.note for item in grouped[k]) for k in keys)
    # for each group store the start times so we can compare with answer
    group_time = tuple(min(item.note.start for item in grouped[k]) for k in keys)
    print(f"groups={len(note_tuples)}, total_notes={len(qn)}, targets={len(targets)}")

    best = []
    for fret_height in (0.35, 0.5, 0.8, 1.2, 1.6, 2.0):
        for open_bonus in (0.0, 0.5, 1.0, 2.0):
            w = FingeringWeights(
                fret_height=fret_height, fret_movement=2.0,
                string_movement=0.6, large_shift=2.5,
                open_string_bonus=open_bonus,
            )
            opt = FingeringOptimizer(weights=w)
            assignments = opt.optimize_groups(note_tuples)
            tab_e = []
            for gi, grp_assign in enumerate(assignments):
                for pos in grp_assign:
                    if pos is None:
                        continue
                    tab_e.append((group_time[gi], pos.string, pos.fret))
            hits = 0
            midi_hits = 0
            for t, s, f, _m, _ in targets:
                abs_t = 25.0 + t
                if any(abs(e0 - abs_t) <= 0.3 and es == s and ef == f for e0, es, ef in tab_e):
                    hits += 1
                    midi_hits += 1
                else:
                    tgt_midi = Guitar.standard().midi_at(s, f)
                    if any(abs(e0 - abs_t) <= 0.3 for e0, es, ef in tab_e if Guitar.standard().midi_at(es, ef) in (tgt_midi, tgt_midi-12, tgt_midi+12)):
                        midi_hits += 1
            best.append({
                'fret_height': fret_height, 'open_bonus': open_bonus,
                'hits': hits, 'midi_hits': midi_hits,
                'strict': round(hits / len(targets), 4),
            })

    best.sort(key=lambda r: r['hits'], reverse=True)
    for r in best[:15]:
        print(r)


if __name__ == '__main__':
    main()
