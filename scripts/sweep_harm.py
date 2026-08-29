"""Sweep harmonic_ratio around the baseline to understand octave suppression cost."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.loader import load_audio
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer

from scripts.match_answer import run


def main() -> None:
    audio_path = Path('/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3')
    answer_path = Path('/Users/youzi/Downloads/晴天_1-8小节_TAB_含击勾弦.txt')
    rows = []
    for harm, aw, rt in itertools.product(
        (0.20, 0.30, 0.40, 0.50, 0.58, 0.70),
        (0.0, 0.35),
        (0.24,),
    ):
        rep = run(
            audio_path,
            answer_path,
            bars=8,
            attack_weight=aw,
            relative_pitch_threshold=rt,
            harmonic_ratio=harm,
            log_compress=True,
        )
        rows.append({
            "harm": harm, "aw": aw, "rel": rt,
            "strict": rep["recall"], "midi": rep["midi_recall"], "seq": rep["sequence_recall"],
            "offset": rep["offset_seconds"], "notes": rep["notes"],
        })
        print(rows[-1])
    Path('build/harm-sweep.json').write_text(json.dumps(rows, indent=2))
    best = max(rows, key=lambda r: r["strict"])
    print("best:", best)


if __name__ == '__main__':
    main()
