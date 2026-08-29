"""Detail bar 1 expectations vs detected pitches at each target time."""
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
    notes_sorted = sorted(notes, key=lambda n: n.start)
    onsets, _timing = az.rhythm_analyzer.detect(audio)
    strengths, rms, frame_times = az._midi_strengths(az._waveform(audio), audio.sample_rate)
    targets = _answer_targets(events, 72.0)
    offset = 25.0
    print("Bar1 targets (expected midis):", [(s, f, guitar.midi_at(s, f)) for t, s, f, m, _ in targets if m == 1])
    print()
    for t, s, f, m, _ in targets:
        if m != 1: continue
        abs_t = offset + t
        tgt = guitar.midi_at(s, f)
        idx = tgt - az.min_midi
        # What's near this time in the audio
        mask = (frame_times >= abs_t - 0.1) & (frame_times <= abs_t + 0.3)
        win_peaks = np.argsort(strengths[:, mask].max(axis=1))[::-1][:6]
        # What raw detected notes are near
        near = [(n.start, n.midi) for n in notes_sorted if abs(n.start - abs_t) < 0.35]
        hit = any(midi == tgt for _, midi in near)
        has_oct_low = any(midi == tgt - 12 for _, midi in near)
        has_oct_up = any(midi == tgt + 12 for _, midi in near)
        mark = '✓' if hit else ('oct-low' if has_oct_low else ('oct-up' if has_oct_up else 'MISSING'))
        print(f"m{m} s{s}f{f} t={abs_t:5.2f} m{tgt:2d} {mark:8s} | near_notes={near[:5]} | win_top6={[int(x+az.min_midi) for x in win_peaks]}")


if __name__ == '__main__':
    main()
