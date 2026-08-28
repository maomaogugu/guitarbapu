"""Tests for guitar tuning, fretboard mapping, and fingering selection."""

from src.music.fretboard import FretPosition, Fretboard
from src.music.guitar import Guitar
from src.music.note import Note


def test_standard_guitar_open_strings_and_fret_range():
    guitar = Guitar.standard()

    assert [string.tuning_midi for string in guitar.strings] == [64, 59, 55, 50, 45, 40]
    assert guitar.fret_count == 24
    assert guitar.midi_at(6, 0) == 40
    assert guitar.midi_at(1, 24) == 88


def test_find_positions_includes_open_string():
    positions = Fretboard().find_positions(Note(midi=40))

    assert positions == (FretPosition(string=6, fret=0),)


def test_find_positions_includes_twelfth_fret_octave():
    positions = Fretboard().find_positions(Note(midi=52))

    assert FretPosition(string=6, fret=12) in positions
    assert Fretboard().position_midi(FretPosition(string=6, fret=12)) == 52


def test_find_positions_returns_all_playable_locations_in_low_fret_order():
    positions = Fretboard().find_positions(Note(midi=64))

    assert positions[:3] == (
        FretPosition(string=1, fret=0),
        FretPosition(string=2, fret=5),
        FretPosition(string=3, fret=9),
    )
    assert len(positions) == 6


def test_unplayable_note_returns_no_position():
    fretboard = Fretboard()

    assert fretboard.find_positions(Note(midi=39)) == ()
    assert fretboard.choose_position(Note(midi=39)) is None


def test_map_notes_prefers_low_fret_and_smooth_movement():
    fretboard = Fretboard()
    selected = fretboard.map_notes((Note(midi=64), Note(midi=67), Note(midi=69)))

    assert selected == (
        FretPosition(string=1, fret=0),
        FretPosition(string=1, fret=3),
        FretPosition(string=1, fret=5),
    )


def test_capo_changes_sounding_pitch():
    guitar = Guitar.standard(capo=2)

    assert guitar.midi_at(6, 0) == 42
    assert Fretboard(guitar).find_positions(Note(midi=42)) == (
        FretPosition(string=6, fret=0),
    )
