"""High-resolution CQT probe to check whether the answer melody notes are really present."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.audio.loader import load_audio
from src.audio.librosa_compat import import_librosa


def main():
    audio = load_audio('/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3')
    librosa = import_librosa()
    w = np.asarray(audio.waveform, dtype=np.float32)
    if w.ndim == 2:
        w = w.mean(axis=1)
    sr = audio.sample_rate

    # For offset=25.0, bar 1 spans 25.0 to 28.33s. Bar 1 expects:
    #   t=25.00 E3(m43 on E string), t=25.33 D3f2 (m52), t=25.51 G3f0 (m55), t=25.56 B3f0 (m59), t=26.05 G3f0, t=26.42 B3f0,
    #   t=26.54 e3f3 (G4=67), t=27.05 e4f0 (E4=64), t=27.31 G3f2 (m57), t=27.50 B3f0, t=28.08 e4f3
    times_to_check = [
        (25.00, 43), (25.33, 52), (25.51, 55), (25.56, 59),
        (26.05, 55), (26.42, 59), (26.54, 67), (27.05, 64),
        (27.31, 57), (27.50, 59), (28.08, 67),
    ]
    # high-frequency check: kernel CQT around each note
    for t, midi in times_to_check:
        f_target = librosa.midi_to_hz(midi)
        # extract short window +/- 0.15s
        start = max(0, int((t - 0.15) * sr))
        end = min(len(w), int((t + 0.3) * sr))
        seg = w[start:end]
        # compute CQT with high time resolution
        cqt = librosa.cqt(seg, sr=sr, hop_length=128, fmin=librosa.note_to_hz('E1'),
                          n_bins=84, bins_per_octave=12)
        mag = np.abs(cqt)
        freqs = librosa.cqt_frequencies(84, fmin=librosa.note_to_hz('E1'), bins_per_octave=12)
        idx = np.argmin(np.abs(freqs - f_target))
        # Check peak vs neighbors in a sliding window
        mx = mag[idx, :].max()
        # compare to 2 octaves below/above
        idx_dn = idx - 24
        idx_up = idx + 24
        v_dn = mag[idx_dn, :].max() if 0 <= idx_dn else 0
        v_up = mag[idx_up, :].max() if idx_up < len(freqs) else 0
        v_oct_dn = mag[idx - 12, :].max() if idx - 12 >= 0 else 0
        v_oct_up = mag[idx + 12, :].max() if idx + 12 < len(freqs) else 0
        print(f"t={t:5.2f} expect m{midi:2d} ({f_target:6.1f}Hz): local={mx:.2f} | octave_dn={v_oct_dn:.2f} octave_up={v_oct_up:.2f} 2oct_dn={v_dn:.2f}")
        # ALSO print top-8 strongest in this segment
        top = np.argsort(mag.max(axis=1))[::-1][:8]
        print(f"    top-8 freqs in this window: {[(f'{freqs[i]:.0f}Hz', round(float(mag[i].max()),2)) for i in top]}")


if __name__ == '__main__':
    main()
