"""Tests for structured chord identity and validation."""

import pytest

from src.music.chord import Chord, infer_chord_identity


@pytest.mark.parametrize(
    ("midis", "expected"),
    [
        ((48, 52, 55), (0, "major")),
        ((52, 55, 59), (4, "minor")),
        ((40, 47), (4, "power")),
        ((43, 45, 50), (7, "sus2")),
    ],
)
def test_infer_supported_chord_identities(midis, expected):
    assert infer_chord_identity(midis) == expected


def test_chord_normalizes_pitches_and_creates_note_events():
    chord = Chord.from_midis(
        (55, 48, 52, 48),
        start=0.5,
        duration=1.0,
        confidence=0.8,
    )

    assert chord.midis == (48, 52, 55)
    assert chord.name == "C"
    assert [(note.midi, note.start, note.duration) for note in chord.notes()] == [
        (48, 0.5, 1.0),
        (52, 0.5, 1.0),
        (55, 0.5, 1.0),
    ]


def test_unknown_pitch_set_uses_readable_note_names():
    chord = Chord.from_midis((48, 49, 55))

    assert chord.root_pitch_class is None
    assert chord.quality is None
    assert chord.name == "C3/C#3/G3"


def test_chord_rejects_single_pitch_and_more_than_six_pitches():
    with pytest.raises(ValueError, match="at least two"):
        Chord.from_midis((60,))
    with pytest.raises(ValueError, match="more than six"):
        Chord.from_midis(range(60, 67))
