"""Grid-search helper for local reference TAB evaluation cases.

This script is deliberately report-only: it never rewrites source code or a
project file. Use it on licensed/owned reference snippets to choose candidate
analysis parameters before changing defaults.
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.loader import load_audio
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer
from src.eval.ascii_tab import flatten_midi_events, parse_ascii_tab


def _top_voice_midis(analysis) -> tuple[int, ...]:
    grouped: list[tuple[float, int]] = []
    for note in analysis.notes:
        if grouped and note.start - grouped[-1][0] <= 0.08:
            time, midi = grouped[-1]
            grouped[-1] = (time, max(midi, note.midi))
        else:
            grouped.append((note.start, note.midi))
    return tuple(midi for _, midi in grouped)


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


def evaluate_case(case: dict, grid: dict[str, list[float]]) -> dict:
    audio_path = Path(case["audio"]).expanduser()
    reference = flatten_midi_events(
        parse_ascii_tab(Path(case["tab"]).read_text(encoding="utf-8"))
    )
    audio = load_audio(audio_path)
    results: list[dict] = []
    for relative, harmonic, energy, attack_weight in itertools.product(
        grid["relative_pitch_threshold"],
        grid["harmonic_ratio"],
        grid["energy_threshold"],
        grid["attack_weight"],
    ):
        analyzer = PolyphonicAudioAnalyzer(
            relative_pitch_threshold=relative,
            harmonic_ratio=harmonic,
            energy_threshold=energy,
            attack_weight=attack_weight,
        )
        analysis = analyzer.analyze(audio)
        detected = _top_voice_midis(analysis)
        high_voice = sum(note.midi >= 64 for note in analysis.notes)
        results.append(
            {
                "relative_pitch_threshold": relative,
                "harmonic_ratio": harmonic,
                "energy_threshold": energy,
                "best_window_ratio": _best_window_ratio(reference, detected),
                "attack_weight": attack_weight,
                "notes": len(analysis.notes),
                "chords": len(analysis.chords),
                "high_voice_notes_e_string_range": high_voice,
            }
        )
    results.sort(key=lambda item: item["best_window_ratio"], reverse=True)
    return {
        "case": case.get("name", audio_path.name),
        "audio": str(audio_path),
        "reference_events": len(reference),
        "best": results[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path, help="JSON list with {name,audio,tab}")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    grid = {
        "relative_pitch_threshold": [0.12, 0.16, 0.20, 0.24],
        "harmonic_ratio": [0.45, 0.58, 0.70],
        "energy_threshold": [0.05, 0.08],
        "attack_weight": [0.0, 0.2, 0.35],
    }
    report = {
        "grid": grid,
        "cases": [evaluate_case(case, grid) for case in cases],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
