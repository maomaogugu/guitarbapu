"""Tests for text, MIDI, and MusicXML exports."""

from music21 import articulations, converter

from src.exporters import (
    export_midi,
    export_musicxml,
    export_text_tab,
    tablature_to_score,
)
from src.music.guitar import Guitar
from src.music.note import Note
from src.music.tab import TabEvent, Tablature
from src.music.tab_renderer import TextTabRenderer


def _tablature():
    first = Note(64, start=0.0, duration=2 / 3, velocity=90)
    second = Note(67, start=2 / 3, duration=1 / 3, velocity=100)
    return Tablature(
        guitar=Guitar.standard(),
        events=(
            TabEvent(
                string=1,
                fret=0,
                note=first,
                start=first.start,
                duration=first.duration,
                start_beat=0.0,
                duration_beats=1.0,
            ),
            TabEvent(
                string=1,
                fret=3,
                note=second,
                start=second.start,
                duration=second.duration,
                start_beat=1.0,
                duration_beats=0.5,
                technique="slide",
                technique_confidence=0.82,
            ),
        ),
        tempo_bpm=90.0,
        time_signature=(4, 4),
        subdivision=4,
        measure_count=1,
    )


def test_text_export_matches_gui_renderer(tmp_path):
    tablature = _tablature()
    output = tmp_path / "song.txt"

    returned = export_text_tab(tablature, output)

    assert returned == output.resolve()
    assert output.read_text(encoding="utf-8").rstrip() == TextTabRenderer().render(
        tablature
    )


def test_score_preserves_pitch_timing_velocity_and_fret_metadata():
    score = tablature_to_score(_tablature())
    notes = list(score.recurse().notes)

    assert [item.pitch.midi for item in notes] == [64, 67]
    assert [float(item.offset) for item in notes] == [0.0, 1.0]
    assert [float(item.duration.quarterLength) for item in notes] == [1.0, 0.5]
    assert [item.volume.velocity for item in notes] == [90, 100]
    assert isinstance(notes[0].articulations[0], articulations.StringIndication)
    assert notes[0].articulations[0].number == 1
    assert isinstance(notes[0].articulations[1], articulations.FretIndication)
    assert notes[0].articulations[1].number == 0


def test_midi_export_can_be_read_back(tmp_path):
    output = tmp_path / "song.mid"

    export_midi(_tablature(), output)
    parsed = converter.parse(output)

    assert output.read_bytes().startswith(b"MThd")
    assert [item.pitch.midi for item in parsed.recurse().notes] == [64, 67]


def test_musicxml_export_can_be_read_back_with_timing(tmp_path):
    output = tmp_path / "song.musicxml"

    export_musicxml(_tablature(), output)
    parsed = converter.parse(output)
    parsed_notes = list(parsed.recurse().notes)

    xml = output.read_text(encoding="utf-8")
    assert xml.lstrip().startswith("<?xml")
    assert "<string>1</string>" in xml
    assert "<fret>0</fret>" in xml
    assert "slide" in xml
    assert [item.pitch.midi for item in parsed_notes] == [64, 67]
    assert [float(item.duration.quarterLength) for item in parsed_notes] == [1.0, 0.5]
