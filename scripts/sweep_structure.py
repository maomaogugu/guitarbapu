"""Try analyzer structural variants that may break the ~0.39 plateau."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.match_answer import run


def score(**kwargs):
    audio_path = Path('/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3')
    answer_path = Path('/Users/youzi/Downloads/晴天_1-8小节_TAB_含击勾弦.txt')
    rep = run(audio_path, answer_path, bars=8, attack_weight=0.35, harmonic_ratio=0.58, log_compress=True, **kwargs)
    return rep['recall'], rep['midi_recall'], rep['offset_seconds'], rep['notes']


if __name__ == '__main__':
    # baseline
    r = score(); print(f"baseline: strict={r[0]} midi={r[1]} offset={r[2]} notes={r[3]}")
    # variations
    for kw, label in [
        (dict(hop_length=256), 'hop 256'),
        (dict(hop_length=384), 'hop 384'),
        (dict(hop_length=1024), 'hop 1024'),
        (dict(bins_per_semitone=1), 'bins=1'),
        (dict(bins_per_semitone=4), 'bins=4'),
        (dict(min_segment_duration=0.05), 'min_seg 0.05'),
        (dict(min_segment_duration=0.15), 'min_seg 0.15'),
        (dict(min_segment_duration=0.20), 'min_seg 0.20'),
    ]:
        try:
            r = score(**kw)
            print(f"{label:20s}: strict={r[0]} midi={r[1]} offset={r[2]} notes={r[3]}")
        except Exception as e:
            print(f"{label:20s}: ERROR {e}")
