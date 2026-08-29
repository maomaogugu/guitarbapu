"""Detailed spectral probe of a known melody note time."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.audio.loader import load_audio
from src.audio.librosa_compat import import_librosa


def probe(t: float):
    audio = load_audio('/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3')
    librosa = import_librosa()
    w = np.asarray(audio.waveform, dtype=np.float32)
    if w.ndim == 2:
        w = w.mean(axis=1)
    sr = audio.sample_rate
    # fine-grained CQT around t
    seg = w[int((t - 0.3) * sr):int((t + 0.3) * sr)]
    cqt = librosa.cqt(seg, sr=sr, hop_length=256, fmin=40.0, n_bins=96, bins_per_octave=12)
    mag = np.abs(cqt)
    frame_count = max(1, mag.shape[1])
    mid = frame_count // 2
    profile = mag.max(axis=1)  # peak strength across the window
    freqs = librosa.cqt_frequencies(96, fmin=40.0, bins_per_octave=12)
    for target_hz in (98.0, 196.0, 392.0, 330.0, 415.3, 494.0, 165.0):
        idx = int(np.argmin(np.abs(freqs - target_hz)))
        print(f"  {target_hz:6.1f}Hz -> peak CQT mag {profile[idx]:6.2f}")


if __name__ == '__main__':
    # bar 1 expected: at t=25.0+0.00→43(G2), +0.56→59(B3), +1.28→55(G3), +1.54→67(G4), +2.05→64(E4)
    for t in [25.0, 25.56, 26.28, 26.54, 27.05]:
        print(f"t={t}:")
        probe(t)
        print()
