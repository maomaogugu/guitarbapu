"""Tests for the note simplifier that humanizes dense detection output."""

from src.music.note import Note
from src.music.note_filter import simplify_notes


def n(midi, start, duration=0.2, confidence=None):
    return Note(midi=midi, start=start, duration=duration, confidence=confidence)


def test_merges_duplicate_attacks_of_same_pitch():
    notes = [n(64, 1.00), n(64, 1.01), n(64, 1.02, duration=0.4)]
    out = simplify_notes(notes)
    assert len(out) == 1
    assert out[0].start == 1.00
    assert out[0].duration >= 0.42 - 1e-9


def test_drops_micro_notes_shorter_than_min_duration():
    notes = [n(64, 0.0, duration=0.02), n(65, 0.5, duration=0.3)]
    out = simplify_notes(notes)
    assert [note.midi for note in out] == [65]


def test_caps_simultaneous_notes_by_confidence():
    notes = [
        n(48, 1.0, confidence=0.9),
        n(52, 1.0, confidence=0.8),
        n(55, 1.0, confidence=0.7),
        n(59, 1.0, confidence=0.6),
        n(64, 1.0, confidence=0.5),
        n(67, 1.0, confidence=0.4),
    ]
    out = simplify_notes(notes, max_simultaneous=4)
    assert len(out) == 4
    assert {note.midi for note in out} == {48, 52, 55, 59}


def test_density_cap_limits_notes_per_second():
    notes = [n(60 + (i % 5), i * 0.02, duration=0.1) for i in range(60)]
    out = simplify_notes(notes, max_notes_per_second=10)
    # 60 notes over 1.2s unfiltered; cap must cut it down substantially
    assert len(out) < len(notes) // 2
    assert len(out) <= 20  # generous bound around the steady 10/s rate


def test_preserves_sorted_order_and_empty_input():
    assert simplify_notes([]) == ()
    notes = [n(64, 0.2), n(59, 0.1)]
    out = simplify_notes(notes)
    assert [note.midi for note in out] == [59, 64]
