"""Tests for Phase 4 note-event cleanup."""

import pytest

from src.music.note import Note
from src.music.note_processor import NoteProcessor


def test_empty_notes_remain_empty():
    assert NoteProcessor().process(()) == ()


def test_short_pitch_glitch_is_filtered():
    notes = (Note(60, start=0.0, duration=0.01),)

    assert NoteProcessor(min_note_duration=0.08).process(notes) == ()


def test_normal_note_is_preserved():
    note = Note(64, start=0.25, duration=0.5)

    assert NoteProcessor().process((note,)) == (note,)


def test_same_pitch_across_short_gap_is_merged():
    notes = (
        Note(64, start=0.0, duration=0.4),
        Note(64, start=0.45, duration=0.35),
    )

    result = NoteProcessor(merge_gap=0.08).process(notes)

    assert len(result) == 1
    assert result[0].start == 0
    assert result[0].duration == pytest.approx(0.8)


def test_different_pitches_are_not_merged():
    notes = (
        Note(64, start=0.0, duration=0.4),
        Note(67, start=0.42, duration=0.4),
    )

    assert NoteProcessor().process(notes) == notes


def test_tiny_wrong_pitch_between_matching_notes_is_removed():
    notes = (
        Note(64, start=0.0, duration=0.4),
        Note(65, start=0.4, duration=0.02),
        Note(64, start=0.42, duration=0.4),
    )

    result = NoteProcessor().process(notes)

    assert len(result) == 1
    assert result[0].midi == 64
    assert result[0].duration == pytest.approx(0.82)


def test_onset_splits_a_repeated_sustained_pitch():
    note = Note(64, start=0.0, duration=1.0)

    result = NoteProcessor().process((note,), onset_times=(0.5,))

    assert [(item.start, item.duration) for item in result] == [
        (0.0, 0.5),
        (0.5, 0.5),
    ]


def test_different_overlapping_pitches_are_made_monophonic():
    notes = (
        Note(64, start=0.0, duration=0.6),
        Note(67, start=0.4, duration=0.5),
    )

    result = NoteProcessor().process(notes)

    assert result[0].duration == pytest.approx(0.4)
    assert result[1] == notes[1]
