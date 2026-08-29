"""Try different track / string preference strategies."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.match_answer import run


def main():
    audio_path = Path('/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3')
    answer_path = Path('/Users/youzi/Downloads/晴天_1-8小节_TAB_含击勾弦.txt')
    for binsize in (2, 3):
        rep = run(
            audio_path, answer_path, bars=8,
            attack_weight=0.35, relative_pitch_threshold=0.24,
            harmonic_ratio=0.58, log_compress=True,
            bins_per_semitone=binsize,
        )
        print(f"bins={binsize}: strict={rep['recall']} midi={rep['midi_recall']} seq={rep['sequence_recall']} notes={rep['notes']}")


if __name__ == '__main__':
    main()
