"""JSON serialization for versioned GuitarBapu project files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from ..audio.analyzer import AudioAnalysis
from ..audio.rhythm import RhythmAnalysis
from ..music.chord import Chord
from ..music.guitar import Guitar, GuitarString
from ..music.note import Note
from ..music.tab import TabEvent, TabRest, Tablature, UnmappedNote
from ..music.technique import GuitarTechnique, TechniqueDetection
from ..music.timing import QuantizedNote, Rest, TimingInfo
from ..music.track import TrackRole
from ..utils.atomic import atomic_replace
from .model import TranscriptionProject
from .track import TranscriptionTrack


PROJECT_FORMAT = "guitarbapu-project"
CURRENT_SCHEMA_VERSION = 1


class ProjectFormatError(ValueError):
    """Raised when a project file is malformed or uses an unknown schema."""


def _note_to_dict(note: Note) -> dict[str, Any]:
    return {
        "midi": note.midi,
        "start": note.start,
        "duration": note.duration,
        "velocity": note.velocity,
        "frequency_hz": note.frequency_hz,
        "confidence": note.confidence,
    }


def _note_from_dict(data: Mapping[str, Any]) -> Note:
    return Note(
        midi=int(data["midi"]),
        start=float(data.get("start", 0.0)),
        duration=float(data.get("duration", 0.0)),
        velocity=int(data.get("velocity", 100)),
        frequency_hz=(
            None
            if data.get("frequency_hz") is None
            else float(data["frequency_hz"])
        ),
        confidence=(
            None if data.get("confidence") is None else float(data["confidence"])
        ),
    )


def _technique_to_dict(detection: TechniqueDetection) -> dict[str, Any]:
    return {
        "technique": detection.technique.value,
        "note": _note_to_dict(detection.note),
        "confidence": detection.confidence,
        "related_note": (
            _note_to_dict(detection.related_note)
            if detection.related_note is not None
            else None
        ),
        "pitch_change_semitones": detection.pitch_change_semitones,
    }


def _technique_from_dict(data: Mapping[str, Any]) -> TechniqueDetection:
    related = data.get("related_note")
    return TechniqueDetection(
        technique=GuitarTechnique(str(data["technique"])),
        note=_note_from_dict(data["note"]),
        confidence=float(data["confidence"]),
        related_note=_note_from_dict(related) if related is not None else None,
        pitch_change_semitones=(
            None
            if data.get("pitch_change_semitones") is None
            else float(data["pitch_change_semitones"])
        ),
    )


def _chord_to_dict(chord: Chord) -> dict[str, Any]:
    return {
        "midis": list(chord.midis),
        "start": chord.start,
        "duration": chord.duration,
        "root_pitch_class": chord.root_pitch_class,
        "quality": chord.quality,
        "confidence": chord.confidence,
    }


def _chord_from_dict(data: Mapping[str, Any]) -> Chord:
    return Chord(
        midis=tuple(int(value) for value in data["midis"]),
        start=float(data.get("start", 0.0)),
        duration=float(data.get("duration", 0.0)),
        root_pitch_class=(
            None
            if data.get("root_pitch_class") is None
            else int(data["root_pitch_class"])
        ),
        quality=(None if data.get("quality") is None else str(data["quality"])),
        confidence=(
            None if data.get("confidence") is None else float(data["confidence"])
        ),
    )


def _timing_to_dict(timing: TimingInfo) -> dict[str, Any]:
    return {
        "tempo_bpm": timing.tempo_bpm,
        "beat_times": list(timing.beat_times),
        "time_signature": (
            list(timing.time_signature) if timing.time_signature is not None else None
        ),
        "subdivision": timing.subdivision,
    }


def _timing_from_dict(data: Mapping[str, Any]) -> TimingInfo:
    signature = data.get("time_signature")
    return TimingInfo(
        tempo_bpm=(
            None if data.get("tempo_bpm") is None else float(data["tempo_bpm"])
        ),
        beat_times=tuple(float(value) for value in data.get("beat_times", ())),
        time_signature=(
            None
            if signature is None
            else (int(signature[0]), int(signature[1]))
        ),
        subdivision=int(data.get("subdivision", 4)),
    )


def _quantized_note_to_dict(item: QuantizedNote) -> dict[str, Any]:
    return {
        "source": _note_to_dict(item.source),
        "note": _note_to_dict(item.note),
        "start_beat": item.start_beat,
        "duration_beats": item.duration_beats,
        "tie_to_next": item.tie_to_next,
    }


def _quantized_note_from_dict(data: Mapping[str, Any]) -> QuantizedNote:
    return QuantizedNote(
        source=_note_from_dict(data["source"]),
        note=_note_from_dict(data["note"]),
        start_beat=(
            None if data.get("start_beat") is None else float(data["start_beat"])
        ),
        duration_beats=(
            None
            if data.get("duration_beats") is None
            else float(data["duration_beats"])
        ),
        tie_to_next=bool(data.get("tie_to_next", False)),
    )


def _rest_to_dict(rest: Rest) -> dict[str, Any]:
    return {
        "start": rest.start,
        "duration": rest.duration,
        "start_beat": rest.start_beat,
        "duration_beats": rest.duration_beats,
    }


def _rest_from_dict(data: Mapping[str, Any]) -> Rest:
    return Rest(
        start=float(data["start"]),
        duration=float(data["duration"]),
        start_beat=(
            None if data.get("start_beat") is None else float(data["start_beat"])
        ),
        duration_beats=(
            None
            if data.get("duration_beats") is None
            else float(data["duration_beats"])
        ),
    )


def _rhythm_to_dict(rhythm: RhythmAnalysis | None) -> dict[str, Any] | None:
    if rhythm is None:
        return None
    return {
        "timing": _timing_to_dict(rhythm.timing),
        "onset_times": list(rhythm.onset_times),
        "quantized_notes": [
            _quantized_note_to_dict(item) for item in rhythm.quantized_notes
        ],
        "rests": [_rest_to_dict(rest) for rest in rhythm.rests],
    }


def _rhythm_from_dict(data: Mapping[str, Any] | None) -> RhythmAnalysis | None:
    if data is None:
        return None
    return RhythmAnalysis(
        timing=_timing_from_dict(data.get("timing", {})),
        onset_times=tuple(float(value) for value in data.get("onset_times", ())),
        quantized_notes=tuple(
            _quantized_note_from_dict(item)
            for item in data.get("quantized_notes", ())
        ),
        rests=tuple(_rest_from_dict(item) for item in data.get("rests", ())),
    )


def _analysis_to_dict(analysis: AudioAnalysis) -> dict[str, Any]:
    return {
        "duration_seconds": analysis.duration_seconds,
        "sample_rate": analysis.sample_rate,
        "notes": [_note_to_dict(note) for note in analysis.notes],
        "raw_notes": [_note_to_dict(note) for note in analysis.raw_notes],
        "chords": [_chord_to_dict(chord) for chord in analysis.chords],
        "techniques": [
            _technique_to_dict(detection) for detection in analysis.techniques
        ],
        "rhythm": _rhythm_to_dict(analysis.rhythm),
    }


def _analysis_from_dict(data: Mapping[str, Any]) -> AudioAnalysis:
    return AudioAnalysis(
        duration_seconds=float(data["duration_seconds"]),
        sample_rate=int(data["sample_rate"]),
        # Large frame-level arrays are intentionally not persisted.
        features={},
        notes=tuple(_note_from_dict(item) for item in data.get("notes", ())),
        raw_notes=tuple(
            _note_from_dict(item) for item in data.get("raw_notes", ())
        ),
        rhythm=_rhythm_from_dict(data.get("rhythm")),
        chords=tuple(_chord_from_dict(item) for item in data.get("chords", ())),
        techniques=tuple(
            _technique_from_dict(item) for item in data.get("techniques", ())
        ),
    )


def _guitar_to_dict(guitar: Guitar) -> dict[str, Any]:
    return {
        "fret_count": guitar.fret_count,
        "capo": guitar.capo,
        "strings": [
            {
                "number": string.number,
                "name": string.name,
                "tuning_midi": string.tuning_midi,
            }
            for string in guitar.strings
        ],
    }


def _guitar_from_dict(data: Mapping[str, Any]) -> Guitar:
    return Guitar(
        strings=tuple(
            GuitarString(
                number=int(item["number"]),
                name=str(item["name"]),
                tuning_midi=int(item["tuning_midi"]),
            )
            for item in data["strings"]
        ),
        fret_count=int(data.get("fret_count", 24)),
        capo=int(data.get("capo", 0)),
    )


def _tab_event_to_dict(event: TabEvent) -> dict[str, Any]:
    return {
        "string": event.string,
        "fret": event.fret,
        "start": event.start,
        "duration": event.duration,
        "note": _note_to_dict(event.note) if event.note is not None else None,
        "start_beat": event.start_beat,
        "duration_beats": event.duration_beats,
        "measure": event.measure,
        "tie_to_next": event.tie_to_next,
        "technique": event.technique,
        "technique_confidence": event.technique_confidence,
        "confidence": event.confidence,
    }


def _tab_event_from_dict(data: Mapping[str, Any]) -> TabEvent:
    note_data = data.get("note")
    return TabEvent(
        string=int(data["string"]),
        fret=int(data["fret"]),
        start=float(data.get("start", 0.0)),
        duration=float(data.get("duration", 0.0)),
        note=_note_from_dict(note_data) if note_data is not None else None,
        start_beat=(
            None if data.get("start_beat") is None else float(data["start_beat"])
        ),
        duration_beats=(
            None
            if data.get("duration_beats") is None
            else float(data["duration_beats"])
        ),
        measure=int(data.get("measure", 1)),
        tie_to_next=bool(data.get("tie_to_next", False)),
        technique=data.get("technique"),
        technique_confidence=(
            None
            if data.get("technique_confidence") is None
            else float(data["technique_confidence"])
        ),
        confidence=(
            None if data.get("confidence") is None else float(data["confidence"])
        ),
    )


def _tab_rest_to_dict(rest: TabRest) -> dict[str, Any]:
    return {
        "start": rest.start,
        "duration": rest.duration,
        "start_beat": rest.start_beat,
        "duration_beats": rest.duration_beats,
        "measure": rest.measure,
    }


def _tab_rest_from_dict(data: Mapping[str, Any]) -> TabRest:
    return TabRest(
        start=float(data["start"]),
        duration=float(data["duration"]),
        start_beat=(
            None if data.get("start_beat") is None else float(data["start_beat"])
        ),
        duration_beats=(
            None
            if data.get("duration_beats") is None
            else float(data["duration_beats"])
        ),
        measure=int(data.get("measure", 1)),
    )


def _unmapped_to_dict(item: UnmappedNote) -> dict[str, Any]:
    return {
        "note": _note_to_dict(item.note),
        "reason": item.reason,
        "start_beat": item.start_beat,
        "measure": item.measure,
    }


def _unmapped_from_dict(data: Mapping[str, Any]) -> UnmappedNote:
    return UnmappedNote(
        note=_note_from_dict(data["note"]),
        reason=str(data["reason"]),
        start_beat=(
            None if data.get("start_beat") is None else float(data["start_beat"])
        ),
        measure=int(data.get("measure", 1)),
    )


def _tablature_to_dict(tablature: Tablature) -> dict[str, Any]:
    return {
        "guitar": _guitar_to_dict(tablature.guitar),
        "events": [_tab_event_to_dict(item) for item in tablature.events],
        "rests": [_tab_rest_to_dict(item) for item in tablature.rests],
        "unmapped_notes": [
            _unmapped_to_dict(item) for item in tablature.unmapped_notes
        ],
        "tempo_bpm": tablature.tempo_bpm,
        "time_signature": list(tablature.time_signature),
        "subdivision": tablature.subdivision,
        "measure_count": tablature.measure_count,
        "diagnostics": list(tablature.diagnostics),
    }


def _tablature_from_dict(data: Mapping[str, Any]) -> Tablature:
    signature = data.get("time_signature", (4, 4))
    return Tablature(
        guitar=_guitar_from_dict(data["guitar"]),
        events=tuple(_tab_event_from_dict(item) for item in data.get("events", ())),
        rests=tuple(_tab_rest_from_dict(item) for item in data.get("rests", ())),
        unmapped_notes=tuple(
            _unmapped_from_dict(item) for item in data.get("unmapped_notes", ())
        ),
        tempo_bpm=(
            None if data.get("tempo_bpm") is None else float(data["tempo_bpm"])
        ),
        time_signature=(int(signature[0]), int(signature[1])),
        subdivision=int(data.get("subdivision", 4)),
        measure_count=int(data.get("measure_count", 1)),
        diagnostics=tuple(str(item) for item in data.get("diagnostics", ())),
    )


def _track_to_dict(track: TranscriptionTrack) -> dict[str, Any]:
    metadata = dict(track.metadata)
    try:
        json.dumps(metadata)
    except (TypeError, ValueError) as exc:
        raise ValueError("track metadata must contain JSON values") from exc
    return {
        "track_id": track.track_id,
        "name": track.name,
        "role": track.role.value,
        "source_name": track.source_name,
        "confidence": track.confidence,
        "metadata": metadata,
        "analysis": _analysis_to_dict(track.analysis),
        "tablature": _tablature_to_dict(track.tablature),
    }


def _track_from_dict(data: Mapping[str, Any]) -> TranscriptionTrack:
    metadata = data.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("track metadata must be an object")
    return TranscriptionTrack(
        track_id=str(data["track_id"]),
        name=str(data["name"]),
        role=TrackRole(str(data.get("role", TrackRole.UNKNOWN.value))),
        source_name=str(data.get("source_name", "original")),
        confidence=(
            None if data.get("confidence") is None else float(data["confidence"])
        ),
        metadata=dict(metadata),
        analysis=_analysis_from_dict(data["analysis"]),
        tablature=_tablature_from_dict(data["tablature"]),
    )


def _stored_audio_path(
    audio_path: Path | None, project_path: Path | None
) -> dict[str, Any]:
    if audio_path is None:
        return {"path": None, "relative": False}
    resolved = audio_path.expanduser().resolve(strict=False)
    if project_path is None:
        return {"path": str(resolved), "relative": False}
    try:
        relative = os.path.relpath(resolved, project_path.parent.resolve(strict=False))
    except ValueError:
        # Windows cannot build a relative path across different drives.
        return {"path": str(resolved), "relative": False}
    return {"path": relative, "relative": True}


def project_to_dict(
    project: TranscriptionProject, *, project_path: str | Path | None = None
) -> dict[str, Any]:
    """Return the stable, JSON-compatible schema for ``project``."""

    target = Path(project_path) if project_path is not None else None
    parameters = dict(project.analysis_parameters)
    try:
        json.dumps(parameters)
    except (TypeError, ValueError) as exc:
        raise ValueError("analysis_parameters must contain JSON values") from exc
    return {
        "format": PROJECT_FORMAT,
        "schema_version": CURRENT_SCHEMA_VERSION,
        "audio": _stored_audio_path(project.audio_path, target),
        "analysis_parameters": parameters,
        "analysis": _analysis_to_dict(project.analysis),
        "tablature": _tablature_to_dict(project.tablature),
        "tracks": [_track_to_dict(track) for track in project.tracks],
        "active_track_id": project.active_track_id,
    }


def save_project(project: TranscriptionProject, path: str | Path) -> Path:
    """Write a UTF-8 project file and return its resolved path."""

    target = Path(path).expanduser().resolve(strict=False)
    payload = project_to_dict(project, project_path=target)

    def write(temporary: Path) -> None:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return atomic_replace(target, write)


def _resolved_audio_path(data: Mapping[str, Any], project_path: Path) -> Path | None:
    value = data.get("path")
    if value is None:
        return None
    candidate = Path(str(value)).expanduser()
    if bool(data.get("relative", False)):
        candidate = project_path.parent / candidate
    return candidate.resolve(strict=False)


def load_project(path: str | Path) -> TranscriptionProject:
    """Load and validate a GuitarBapu JSON project file."""

    source = Path(path).expanduser().resolve(strict=False)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProjectFormatError(f"无法读取项目文件：{exc}") from exc
    if not isinstance(payload, Mapping):
        raise ProjectFormatError("项目文件根节点必须是 JSON 对象")
    if payload.get("format") != PROJECT_FORMAT:
        raise ProjectFormatError("这不是 GuitarBapu 项目文件")
    if payload.get("schema_version") != CURRENT_SCHEMA_VERSION:
        raise ProjectFormatError(
            f"不支持的项目版本：{payload.get('schema_version')!r}"
        )

    try:
        analysis = _analysis_from_dict(payload["analysis"])
        parameters = payload.get("analysis_parameters", {})
        if not isinstance(parameters, Mapping):
            raise TypeError("analysis_parameters must be an object")
        tracks_data = payload.get("tracks", [])
        if not isinstance(tracks_data, list):
            raise TypeError("tracks must be an array")
        return TranscriptionProject(
            audio_path=_resolved_audio_path(payload.get("audio", {}), source),
            analysis=analysis,
            tablature=_tablature_from_dict(payload["tablature"]),
            analysis_parameters=dict(parameters),
            tracks=tuple(_track_from_dict(item) for item in tracks_data),
            active_track_id=(
                None
                if payload.get("active_track_id") is None
                else str(payload["active_track_id"])
            ),
        )
    except (AttributeError, KeyError, TypeError, ValueError, IndexError) as exc:
        raise ProjectFormatError(f"项目内容无效：{exc}") from exc


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "PROJECT_FORMAT",
    "ProjectFormatError",
    "load_project",
    "project_to_dict",
    "save_project",
]
