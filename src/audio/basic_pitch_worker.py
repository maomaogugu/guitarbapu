"""Subprocess entry point for Basic Pitch inference.

Running the TensorFlow-backed model in a child process isolates the GUI from
known TF shutdown crashes (``std::terminate`` / recursive-mutex errors seen on
macOS arm64 builds of basic-pitch's bundled SavedModel).

Usage: ``python -m src.audio.basic_pitch_worker in.wav out.json \
    onset_threshold frame_threshold min_note_length_ms [min_freq] [max_freq]``
Writes ``[[start, end, midi, amplitude], ...]`` JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    (
        audio_path,
        out_path,
        onset_threshold,
        frame_threshold,
        minimum_note_length_ms,
        minimum_frequency,
        maximum_frequency,
    ) = argv[1:8]
    from .basic_pitch_backend import _load_model

    model = _load_model()
    from basic_pitch.inference import predict

    _, _, note_events = predict(
        Path(audio_path),
        model,
        onset_threshold=float(onset_threshold),
        frame_threshold=float(frame_threshold),
        minimum_note_length=float(minimum_note_length_ms),
        minimum_frequency=float(minimum_frequency) if minimum_frequency else None,
        maximum_frequency=float(maximum_frequency) if maximum_frequency else None,
    )
    payload = [
        [float(start), float(end), int(midi), float(amplitude)]
        for start, end, midi, amplitude, _bends in note_events
    ]
    Path(out_path).write_text(json.dumps(payload), encoding="utf-8")
    # Skip interpreter teardown: TF shutdown is where the crash lives.
    import os

    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
