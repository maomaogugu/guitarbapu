"""Examine raw CQT around specific target times to verify octave ground truth."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.audio.loader import load_audio
from src.audio.librosa_compat import import_librosa
from src.eval.answer_tab import parse_answer_tab
from src.music.guitar import Guitar

from scripts.match_answer import _answer_targets


def main() -> None:
    audio = load_audio('/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3')
    librosa = import_librosa()
    w = np.asarray(audio.waveform, dtype=np.float32)
    if w.ndim == 2:
        w = w.mean(axis=1)
    for t in (25.0, 25.51, 26.54, 29.17, 33.46, 35.0):
        start = max(0, int((t - 0.2) * audio.sample_rate))
        end = min(len(w), int((t + 0.2) * audio.sample_rate))
        seg = w[start:end]
        if len(seg) == 0:
            continue
        cqt = librosa.cqt(seg, sr=audio.sample_rate, hop_length=512, fmin=80.0, n_bins=72, bins_per_octave=12)
        mag = np.abs(cqt)
        frame_avg = mag.mean(axis=1)
        freqs = librosa.cqt_frequencies(72, fmin=80.0, bins_per_octave=12)
        # top 6 peaks
        order = np.argsort(frame_avg)[::-1][:8]
        print(f"t={t:.2f}: ", end="")
        for idx in order:
            f = freqs[idx]
            midi = librosa.hz_to_midi(f)
            print(f"{f:6.1f}Hz(midi{midi:5.1f}):{frame_avg[idx]:.2f}  ", end="")
        print()


if __name__ == '__main__':
    main()
