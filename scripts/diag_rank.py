"""Simple melody-string trace chase: answer string1 events vs detector top notes per bar."""
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
    strengths, rms, frame_times = az._midi_strengths(az._waveform(audio), audio.sample_rate)
    targets = _answer_targets(events, 72.0)
    off = 25.0
    # For each target print rank of expected midi across that window and which midis dominate
    print('Expected tab notes (t, midi) and subset where target is among top-3 strongest in 0~0.3s window:')
    dom_count = 0
    for time, string, fret, _m, _ in targets:
        midi = guitar.midi_at(string, fret)
        idx = midi - az.min_midi
        abs_t = off + time
        mask = (frame_times >= abs_t) & (frame_times <= abs_t + 0.3)
        if not mask.any():
            print(f'  t={time:.2f} m{midi} s{string}f{fret} | NO FRAMES')
            continue
        scores = strengths[:, mask].max(axis=1)
        rank = int((scores > scores[idx]).sum() + 1)
        dominant = int(np.argmax(scores)) + az.min_midi
        top3 = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
        top3_m = [int(t) + az.min_midi for t in top3]
        flag = '🎯' if rank <= 3 else ('.' if rank <= 6 else '✗')
        if rank <= 3:
            dom_count += 1
        print(f'  t={time:5.2f} m{midi} s{string}f{fret} rank={rank:2d} {flag} top3={top3_m}')
    print(f'\ntop-3 rate: {dom_count}/{len(targets)}')


if __name__ == '__main__':
    main()
