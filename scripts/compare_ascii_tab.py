"""Compare a local ASCII reference TAB snippet with an audio file.

This is an evaluation helper. The reference TAB path stays local unless a
developer explicitly chooses to add a cleared/licensed fixture to the repo.
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.analyzer import AudioAnalyzer
from src.audio.loader import load_audio
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer
from src.eval.ascii_tab import flatten_midi_events, parse_ascii_tab


def _detected_midis(path: Path, mode: str) -> tuple[int, ...]:
    audio = load_audio(path)
    if mode == "polyphonic":
        analysis = PolyphonicAudioAnalyzer().analyze(audio)
    else:
        analysis = AudioAnalyzer().analyze(audio)

    # A reference ASCII snippet usually describes the audible top voice.  When
    # an analysis contains chord tones, compare against the highest pitch in
    # each short time neighborhood instead of every low accompaniment note.
    grouped: list[tuple[float, int]] = []
    for note in analysis.notes:
        if grouped and note.start - grouped[-1][0] <= 0.08:
            time, midi = grouped[-1]
            grouped[-1] = (time, max(midi, note.midi))
        else:
            grouped.append((note.start, note.midi))
    return tuple(midi for _, midi in grouped)


def _best_window_ratio(reference: tuple[int, ...], detected: tuple[int, ...]) -> dict:
    if not reference or not detected:
        return {"ratio": 0.0, "detected_start_index": None, "detected_end_index": None}
    best = {"ratio": 0.0, "detected_start_index": None, "detected_end_index": None}
    ref_len = len(reference)
    for start in range(0, max(0, len(detected) - ref_len) + 1):
        window = detected[start : start + ref_len]
        ratio = difflib.SequenceMatcher(None, reference, window).ratio()
        if ratio > best["ratio"]:
            best = {
                "ratio": ratio,
                "detected_start_index": start,
                "detected_end_index": start + len(window),
            }
    full_ratio = difflib.SequenceMatcher(None, reference, detected).ratio()
    return {
        "best_window": best,
        "full_ratio": full_ratio,
        "reference_length": len(reference),
        "detected_length": len(detected),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("tab", type=Path)
    parser.add_argument("--mode", choices=("monophonic", "polyphonic"), default="monophonic")
    args = parser.parse_args()

    reference = flatten_midi_events(
        parse_ascii_tab(args.tab.read_text(encoding="utf-8"))
    )
    detected = _detected_midis(args.audio, args.mode)
    report = _best_window_ratio(reference, detected)
    report["mode"] = args.mode
    report["reference_midi"] = reference
    report["detected_midi_prefix"] = detected[: max(20, len(reference))]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
