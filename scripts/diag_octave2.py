"""Check actual spectral energy at expected vs detected octaves for key targets."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.audio.loader import load_audio
from src.audio.librosa_compat import import_librosa
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer
from src.eval.answer_tab import parse_answer_tab
from src.music.guitar import Guitar

from scripts.match_answer import _answer_targets

CASES = [25.51, 26.54, 29.17, 33.46, 35.0, 35.83, 36.94, 38.61, 39.72, 40.28]


def main() -> None:
    audio = load_audio('/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3')
    librosa = import_librosa()
    w = np.asarray(audio.waveform, dtype=np.float32)
    if w.ndim == 2:
        w = w.mean(axis=1)
    sr = audio.sample_rate
    fmin = 40.0
    n_bins = 84  # 7 octaves
    cqt_all = librosa.cqt(w, sr=sr, hop_length=512, fmin=fmin, n_bins=n_bins, bins_per_octave=12)
    mag = np.abs(cqt_all)
    freqs = librosa.cqt_frequencies(n_bins, fmin=fmin, bins_per_octave=12)
    frame_times = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sr, hop_length=512)

    for t in CASES:
        mask = (frame_times >= t - 0.15) & (frame_times < t + 0.15)
        if not mask.any():
            continue
        avg = mag[:, mask].mean(axis=1)
        # for midi range 40..76, compute the midi-strength profile
        strengths = {}
        for midi in range(40, 78):
            peak = np.max([avg[np.argmin(np.abs(freqs - librosa.midi_to_hz(midi)))]] )
            strengths[midi] = peak
        order = sorted(strengths, key=strengths.get, reverse=True)[:10]
        print(f"t={t:.2f}: " + " ".join(f"m{int(m)}={strengths[int(m)]:.1f}" for m in order))


if __name__ == '__main__':
    main()
