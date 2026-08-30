"""Score generated tablature against a hand-checked answer TAB in time domain.

The matcher auto-aligns the reference measures to the audio (the recording may
contain a spoken intro before bar 1), then reports how many answer events the
software reproduces on the correct string and fret at the right time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.loader import load_audio
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer
from src.audio.track_classifier import TrackClassifier
from src.audio.transcription_service import TranscriptionService
from src.eval.answer_tab import AnswerEvent, parse_answer_tab
from src.music.guitar import Guitar
from src.music.tab import Tablature


def _answer_targets(
    events: tuple[AnswerEvent, ...], bpm: float
) -> tuple[tuple[float, int, int, int, str | None], ...]:
    """Expand answer events to expected times assuming 4/4 at ``bpm``."""

    measures = sorted({event.measure for event in events})
    bar_length = 4.0 * 60.0 / bpm
    targets: list[tuple[float, int, int, int, str | None]] = []
    for measure in measures:
        bar_events = sorted(
            (event for event in events if event.measure == measure),
            key=lambda event: (event.order, event.string),
        )
        for event in bar_events:
            fraction = event.order / event.span if event.span > 0 else 0.0
            time = (measure - 1) * bar_length + fraction * bar_length
            targets.append(
                (time, event.string, event.fret, event.measure, event.technique)
            )
    return tuple(targets)


def _scan_offset(
    targets: tuple[tuple[float, int, int, int, str | None], ...],
    notes: tuple,
    midi_at,
    duration: float,
    bar_length: float,
) -> float:
    starts = np.asarray([note.start for note in notes], dtype=float)
    midis = np.asarray([note.midi for note in notes], dtype=float)
    if starts.size == 0:
        return 0.0
    span = targets[-1][0] + bar_length
    best_score = -1.0
    best_offset = 0.0
    for offset in np.arange(0.0, max(0.1, duration - span), 0.5):
        hits = 0
        for time, string, fret, _measure, _tech in targets:
            target = midi_at(string, fret)
            mask = np.abs(starts - (offset + time)) <= 0.30
            if mask.any() and np.isin(midis[mask], (target, target - 12, target + 12)).any():
                hits += 1
        score = hits / len(targets)
        if score > best_score:
            best_score, best_offset = score, float(offset)
    for offset in np.arange(max(0.0, best_offset - 0.6), best_offset + 0.61, 0.1):
        hits = 0
        for time, string, fret, _measure, _tech in targets:
            target = midi_at(string, fret)
            mask = np.abs(starts - (offset + time)) <= 0.225
            if mask.any() and np.isin(midis[mask], (target,)).any():
                hits += 1
        score = hits / len(targets)
        if score > best_score:
            best_score, best_offset = score, float(offset)
    return best_offset


def _strict_score(
    targets,
    tablature: Tablature,
    offset: float,
    midi_at,
    notes,
    tolerance: float = 0.3,
) -> dict:
    events = tuple(
        (event.start, event.string, event.fret) for event in tablature.events
    )
    note_list = tuple((note.start, note.midi) for note in notes)
    hits = 0
    midi_hits = 0
    misses: list[dict] = []
    for time, string, fret, measure, technique in targets:
        absolute = offset + time
        ok = any(
            abs(start - absolute) <= tolerance
            and detected_string == string
            and detected_fret == fret
            for start, detected_string, detected_fret in events
        )
        if ok:
            hits += 1
            midi_hits += 1
            continue
        target_midi = midi_at(string, fret)
        near = any(
            abs(start - absolute) <= tolerance
            and detected_midi in (target_midi, target_midi - 12, target_midi + 12)
            for start, detected_midi in note_list
        )
        if near:
            midi_hits += 1
        misses.append(
            {
                "measure": measure,
                "string": string,
                "fret": fret,
                "time": round(absolute, 3),
                "midi_detected": near,
            }
        )
    return {
        "recall": round(hits / len(targets), 4),
        "midi_recall": round(midi_hits / len(targets), 4),
        "hits": hits,
        "total": len(targets),
        "misses": misses[:40],
    }


def _sequence_score(targets, tablature: Tablature, offset: float) -> dict:
    """Order-based per-(measure, string) sequence match with difflib."""

    import difflib

    bar_length = 4.0 * 60.0 / 72.0
    cells: dict[tuple[int, int], list[int]] = {}
    detected_cells: dict[tuple[int, int], list[int]] = {}

    for time, string, fret, measure, _tech in targets:
        cells.setdefault((measure, string), []).append(fret)

    for event in tablature.events:
        measure_position = (event.start - offset) / bar_length
        measure = int(measure_position) + 1
        if 1 <= measure:
            detected_cells.setdefault((measure, event.string), []).append(
                event.fret
            )

    total = 0
    hits = 0
    details: list[dict] = []
    for (measure, string), expected in sorted(cells.items()):
        detected = detected_cells.get((measure, string), [])
        matcher = difflib.SequenceMatcher(None, expected, detected)
        matched_events = sum(
            block.size for block in matcher.get_matching_blocks()
        )
        hits += matched_events
        total += len(expected)
        details.append(
            {
                "measure": measure,
                "string": string,
                "expected": expected,
                "detected": detected,
                "matched": matched_events,
            }
        )
    return {
        "sequence_recall": round(hits / total, 4) if total else 0.0,
        "sequence_hits": hits,
        "sequence_total": total,
        "sequence_cells": details,
    }


def run(
    audio_path: Path,
    answer_path: Path,
    *,
    bars: int = 8,
    export_tab: Path | None = None,
    backend: str = "cqt",
    **analyzer_kwargs,
) -> dict:
    audio = load_audio(audio_path)
    events = parse_answer_tab(answer_path.read_text(encoding="utf-8"))
    guitar = Guitar.standard()
    if backend == "basic-pitch":
        from src.audio.basic_pitch_backend import BasicPitchAnalyzer

        analyzer = BasicPitchAnalyzer(**analyzer_kwargs)
    else:
        analyzer = PolyphonicAudioAnalyzer(**analyzer_kwargs)
    service = TranscriptionService(
        analyzer=analyzer,
        track_classifier=TrackClassifier(),
    )
    result = service.transcribe(audio_path, audio=audio)
    if export_tab is not None:
        from src.music.tab_renderer import TextTabRenderer

        export_tab.parent.mkdir(parents=True, exist_ok=True)
        export_tab.write_text(
            TextTabRenderer().render(result.tablature) + "\n",
            encoding="utf-8",
        )
    timing = result.analysis.rhythm.timing if result.analysis.rhythm else None
    bpm = 72.0
    bar_length = 4.0 * 60.0 / bpm
    targets = _answer_targets(events, bpm)
    offset = _scan_offset(targets, result.analysis.notes, guitar.midi_at, audio.duration, bar_length)
    report = _strict_score(
        targets,
        result.tablature,
        offset,
        guitar.midi_at,
        result.analysis.notes,
    )
    report.update(_sequence_score(targets, result.tablature, offset))
    report.update(
        {
            "offset_seconds": offset,
            "detected_bpm": timing.tempo_bpm if timing is not None else None,
            "tab_events": len(result.tablature.events),
            "notes": len(result.analysis.notes),
            "chords": len(result.analysis.chords),
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--answer", required=True, type=Path)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--attack-weight", type=float, default=0.0)
    parser.add_argument("--relative-threshold", type=float, default=0.24)
    parser.add_argument("--harmonic-ratio", type=float, default=0.58)
    parser.add_argument(
        "--octave-ratio",
        type=float,
        default=None,
        help="Suppression ratio for pure octaves (defaults to harmonic-ratio)",
    )
    parser.add_argument("--energy-threshold", type=float, default=0.08)
    parser.add_argument(
        "--log-compress",
        action="store_true",
        default=False,
        help="Enable the gentle fingerstyle log-compress boost",
    )
    parser.add_argument("--baseline-percentile", type=float, default=50.0)
    parser.add_argument("--novelty-weight", type=float, default=0.0)
    parser.add_argument("--freq-weight", type=float, default=0.0)
    parser.add_argument("--frontend", choices=("cqt", "stft"), default="cqt")
    parser.add_argument(
        "--backend",
        choices=("cqt", "basic-pitch"),
        default="cqt",
        help="Transcription engine: handcrafted CQT pipeline or Basic Pitch neural model",
    )
    parser.add_argument("--bp-onset-threshold", type=float, default=0.5)
    parser.add_argument("--bp-frame-threshold", type=float, default=0.3)
    parser.add_argument("--bp-min-note-length-ms", type=float, default=127.7)
    parser.add_argument("--export-tab", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.backend == "basic-pitch":
        report = run(
            args.audio,
            args.answer,
            bars=args.bars,
            export_tab=args.export_tab,
            backend="basic-pitch",
            onset_threshold=args.bp_onset_threshold,
            frame_threshold=args.bp_frame_threshold,
            minimum_note_length_ms=args.bp_min_note_length_ms,
        )
    else:
        report = run(
            args.audio,
            args.answer,
            bars=args.bars,
            export_tab=args.export_tab,
            attack_weight=args.attack_weight,
            relative_pitch_threshold=args.relative_threshold,
            harmonic_ratio=args.harmonic_ratio,
            octave_ratio=args.octave_ratio,
            energy_threshold=args.energy_threshold,
            log_compress=args.log_compress,
            baseline_percentile=args.baseline_percentile,
            novelty_weight=args.novelty_weight,
            freq_weight=args.freq_weight,
            frontend=args.frontend,
        )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
