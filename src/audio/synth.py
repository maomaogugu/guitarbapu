"""Render a tablature back to audible audio so users can hear the result."""

from __future__ import annotations

import math

import numpy as np

from src.music.tab import Tablature


def synthesize_tablature(
    tablature: Tablature,
    *,
    sample_rate: int = 22050,
    default_duration: float = 0.35,
) -> np.ndarray:
    """Plucked-string style synthesis: decaying sine + a touch of 2nd harmonic.

    Returns a mono float32 waveform in [-1, 1).
    """

    events = tablature.events
    if not events:
        return np.zeros(0, dtype=np.float32)
    end_time = max(
        event.start + (event.duration if event.duration > 0 else default_duration)
        for event in events
    )
    total = int((end_time + 0.5) * sample_rate)
    out = np.zeros(total, dtype=np.float64)
    for event in events:
        midi = tablature.guitar.midi_at(event.string, event.fret)
        frequency = 440.0 * 2 ** ((midi - 69) / 12)
        duration = event.duration if event.duration > 0 else default_duration
        n = int(duration * sample_rate)
        if n <= 0:
            continue
        t = np.arange(n, dtype=np.float64) / sample_rate
        amplitude = (event.confidence if event.confidence is not None else 80 / 127)
        envelope = np.minimum(1.0, t / 0.005) * np.exp(-3.0 * t / max(duration, 0.1))
        tone = (
            np.sin(2 * math.pi * frequency * t)
            + 0.3 * np.sin(4 * math.pi * frequency * t)
        )
        start_idx = int(event.start * sample_rate)
        stop = min(total, start_idx + n)
        seg = tone[: stop - start_idx] * envelope[: stop - start_idx]
        out[start_idx:stop] += amplitude * seg
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out *= 0.89 / peak
    return out.astype(np.float32)
