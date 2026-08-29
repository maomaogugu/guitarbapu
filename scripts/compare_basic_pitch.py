"""Optional offline comparison against the Basic Pitch model.

Basic Pitch is intentionally NOT a runtime dependency of GuitarBapu. Use this
script in a dedicated evaluation environment to decide whether a stronger
polyphonic model is worth integrating.

Example environment:
    python3.12 -m venv .venv-eval
    .venv-eval/bin/pip install basic-pitch

Then:
    .venv-eval/bin/python scripts/compare_basic_pitch.py song.mp3 ref.txt
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.ascii_tab import flatten_midi_events, parse_ascii_tab


def _best_window_ratio(reference: tuple[int, ...], detected: tuple[int, ...]) -> float:
    if not reference or not detected or len(detected) < len(reference):
        return 0.0
    best = 0.0
    for start in range(0, len(detected) - len(reference) + 1):
        ratio = difflib.SequenceMatcher(
            None, reference, detected[start : start + len(reference)]
        ).ratio()
        best = max(best, ratio)
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("tab", type=Path, nargs="?", help="optional reference TAB")
    args = parser.parse_args()

    if importlib.util.find_spec("basic_pitch") is None:
        print(
            "Basic Pitch is not installed in this environment.\n"
            "Install it in a dedicated eval venv first, e.g.:\n"
            "  python3.12 -m venv .venv-eval\n"
            "  .venv-eval/bin/pip install basic-pitch",
            file=sys.stderr,
        )
        return 2

    from basic_pitch.inference import predict

    raw = predict(str(args.audio))
    note_events = None
    for item in raw if isinstance(raw, tuple) else (raw,):
        try:
            first = item[0]
        except (TypeError, IndexError, KeyError):
            continue
        if isinstance(first, (tuple, list)) and len(first) >= 3:
            note_events = item
            break
    if note_events is None:
        print("Could not locate note events in Basic Pitch output.", file=sys.stderr)
        return 3
    notes = tuple(
        sorted(
            (
                float(event[0]),
                float(event[1]),
                int(event[2]),
            )
            for event in note_events
        )
    )
    detected = tuple(pitch for _, _, pitch in notes)
    report: dict[str, object] = {
        "audio": str(args.audio),
        "note_events": len(detected),
        "midi_min": min(detected, default=None),
        "midi_max": max(detected, default=None),
    }
    if args.tab is not None:
        reference = flatten_midi_events(
            parse_ascii_tab(args.tab.read_text(encoding="utf-8"))
        )
        report.update(
            {
                "reference_events": len(reference),
                "best_window_ratio": _best_window_ratio(reference, detected),
            }
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
