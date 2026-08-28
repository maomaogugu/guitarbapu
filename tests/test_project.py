"""Round-trip tests for versioned GuitarBapu JSON projects."""

import json

import numpy as np
import pytest

from src.audio.analyzer import AudioAnalysis
from src.audio.rhythm import RhythmAnalysis
from src.music.chord import Chord
from src.music.guitar import Guitar
from src.music.note import Note
from src.music.tab import TabEvent, TabRest, Tablature, UnmappedNote
from src.music.timing import QuantizedNote, Rest, TimingInfo
from src.project import (
    CURRENT_SCHEMA_VERSION,
    ProjectFormatError,
    TranscriptionProject,
    load_project,
    save_project,
)


def _project(audio_path):
    source = Note(
        64,
        start=0.5,
        duration=0.5,
        frequency_hz=329.6,
        confidence=0.92,
    )
    quantized = Note(64, start=0.5, duration=0.5)
    timing = TimingInfo(
        tempo_bpm=120.0,
        beat_times=(0.0, 0.5, 1.0),
        time_signature=(4, 4),
        subdivision=4,
    )
    analysis = AudioAnalysis(
        duration_seconds=2.0,
        sample_rate=44_100,
        features={"pitch_hz": np.asarray([329.6]), "tempo_bpm": 120.0},
        notes=(source,),
        raw_notes=(source,),
        rhythm=RhythmAnalysis(
            timing=timing,
            onset_times=(0.5,),
            quantized_notes=(
                QuantizedNote(
                    source=source,
                    note=quantized,
                    start_beat=1.0,
                    duration_beats=1.0,
                ),
            ),
            rests=(Rest(0.0, 0.5, 0.0, 1.0),),
        ),
        chords=(Chord.from_midis((48, 52, 55), start=0.5, duration=0.5),),
    )
    tablature = Tablature(
        guitar=Guitar.standard(capo=1),
        events=(
            TabEvent(
                string=2,
                fret=4,
                start=0.5,
                duration=0.5,
                note=source,
                start_beat=1.0,
                duration_beats=1.0,
                confidence=0.92,
            ),
        ),
        rests=(TabRest(0.0, 0.5, 0.0, 1.0),),
        unmapped_notes=(UnmappedNote(Note(20), "超出音域", 3.0, 1),),
        tempo_bpm=120.0,
        time_signature=(4, 4),
        subdivision=4,
        measure_count=1,
        diagnostics=("测试诊断",),
    )
    return TranscriptionProject(
        audio_path=audio_path,
        analysis=analysis,
        tablature=tablature,
        analysis_parameters={"fmin_hz": 65.41, "beat_subdivision": 4},
    )


def test_project_json_round_trip_uses_relative_audio_reference(tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    project_path = tmp_path / "song.guitarbapu.json"

    save_project(_project(audio_path), project_path)
    loaded = load_project(project_path)

    payload = json.loads(project_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CURRENT_SCHEMA_VERSION
    assert payload["audio"] == {"path": "audio.wav", "relative": True}
    assert "pitch_hz" not in project_path.read_text(encoding="utf-8")
    assert loaded.audio_path == audio_path.resolve()
    assert loaded.analysis.notes == _project(audio_path).analysis.notes
    assert loaded.analysis.rhythm == _project(audio_path).analysis.rhythm
    assert loaded.analysis.chords == _project(audio_path).analysis.chords
    assert loaded.analysis.features == {}
    assert loaded.tablature == _project(audio_path).tablature
    assert loaded.analysis_parameters["fmin_hz"] == 65.41


def test_project_can_open_when_referenced_audio_is_missing(tmp_path):
    missing_audio = tmp_path / "missing.wav"
    project_path = tmp_path / "missing-audio.json"
    save_project(_project(missing_audio), project_path)

    loaded = load_project(project_path)

    assert loaded.audio_path == missing_audio.resolve()
    assert not loaded.audio_path.exists()
    assert loaded.tablature.events[0].fret == 4


def test_project_without_chords_field_remains_backward_compatible(tmp_path):
    project_path = tmp_path / "legacy-project.json"
    save_project(_project(None), project_path)
    payload = json.loads(project_path.read_text(encoding="utf-8"))
    payload["analysis"].pop("chords")
    project_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_project(project_path)

    assert loaded.analysis.chords == ()
    assert loaded.analysis.notes


def test_project_rejects_unknown_schema_version(tmp_path):
    project_path = tmp_path / "future.json"
    project_path.write_text(
        json.dumps(
            {
                "format": "guitarbapu-project",
                "schema_version": 999,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProjectFormatError, match="不支持的项目版本"):
        load_project(project_path)


def test_project_rejects_invalid_json(tmp_path):
    project_path = tmp_path / "broken.json"
    project_path.write_text("not json", encoding="utf-8")

    with pytest.raises(ProjectFormatError, match="无法读取项目文件"):
        load_project(project_path)


def test_project_wraps_invalid_nested_content_as_format_error(tmp_path):
    project_path = tmp_path / "invalid-content.json"
    project_path.write_text(
        json.dumps(
            {
                "format": "guitarbapu-project",
                "schema_version": CURRENT_SCHEMA_VERSION,
                "audio": [],
                "analysis": [],
                "tablature": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProjectFormatError, match="项目内容无效"):
        load_project(project_path)
