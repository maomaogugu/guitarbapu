"""Find the true bar-1 offset by correlating note onset pattern density."""
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
    bar_len = 4.0 * 60.0 / 72.0
    span = targets[-1][0] + bar_len

    # Build target note sequence with pitch classes
    tg = [(t, guitar.midi_at(s, f)) for t, s, f, _, _ in targets]
    # Compute per-frame salience: normalized energy per pitch-class
    pc_mask = np.zeros((az.max_midi - az.min_midi + 1), dtype=bool)

    results = []
    for offset in np.arange(0.0, audio.duration - span, 0.25):
        hit = 0
        octave_hit = 0
        for rel_t, midi in tg:
            idx = midi - az.min_midi
            abs_t = offset + rel_t
            mask = (frame_times >= abs_t - 0.12) & (frame_times <= abs_t + 0.25)
            if not mask.any():
                continue
            e = float(strengths[idx, mask].max())
            em = float(strengths[idx - 12, mask].max()) if idx - 12 >= 0 else 0
            ep = float(strengths[idx + 12, mask].max()) if idx + 12 < strengths.shape[0] else 0
            if e >= max(em, ep) * 0.8:
                hit += 1
            if e >= em * 0.6 or em >= e * 1.2:
                octave_hit += 1
        results.append((hit, octave_hit, offset))
    results.sort(reverse=True)
    print("Top 20 offsets (hit=target wins octaves):")
    for h, oh, o in results[:20]:
        print(f"  offset={o:6.2f} hit={h} oct-hit={oh}")


if __name__ == '__main__':
    main()
