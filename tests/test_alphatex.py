"""Tests for the Tablature -> alphaTex converter."""

from src.exporters.alphatex import to_alphatex
from src.music.guitar import Guitar
from src.music.note import Note
from src.music.tab import TabEvent, Tablature


def _tablature(events, **kwargs):
    return Tablature(guitar=Guitar.standard(), events=tuple(events), **kwargs)


def test_header_contains_tempo_time_and_capo():
    tab = _tablature(
        (), measure_count=1, tempo_bpm=72.0, time_signature=(4, 4)
    )
    text = to_alphatex(tab)
    assert '\\title "Transcription"' in text
    assert "\\tempo 72" in text
    assert "\\ts 4 4" in text
    assert "\\staff{tabs}" in text


def test_capo_header_only_when_set():
    plain = to_alphatex(_tablature((), measure_count=1))
    assert "\\capo" not in plain
    with_capo = to_alphatex(
        Tablature(guitar=Guitar.standard(capo=3), events=(), measure_count=1)
    )
    assert "\\capo 3" in with_capo


def test_single_notes_and_chords_map_to_alphatex():
    tab = _tablature(
        [
            TabEvent(1, 3, start=0.0, duration=0.25, start_beat=0.0, duration_beats=1.0, note=Note(67)),
            TabEvent(2, 0, start=0.25, duration=0.25, start_beat=1.0, duration_beats=1.0, note=Note(59)),
            # chord of two notes at beat 2
            TabEvent(1, 0, start=0.5, duration=0.25, start_beat=2.0, duration_beats=1.0, note=Note(64)),
            TabEvent(3, 0, start=0.5, duration=0.25, start_beat=2.0, duration_beats=1.0, note=Note(55)),
        ],
        measure_count=1,
        tempo_bpm=120.0,
    )
    text = to_alphatex(tab)
    assert "3.1.4" in text
    assert "0.2.4" in text
    assert "(0.1 0.3).4" in text


def test_rests_fill_gaps():
    tab = _tablature(
        [TabEvent(1, 3, start=0.0, duration=0.25, start_beat=0.0, duration_beats=1.0, note=Note(67))],
        measure_count=1,
        tempo_bpm=120.0,
    )
    text = to_alphatex(tab)
    assert "r.2 d" in text or "r.2" in text or "r.1" in text


def test_measure_barlines_split_events():
    tab = _tablature(
        [
            TabEvent(1, 3, start=0.0, duration=0.25, start_beat=0.0, duration_beats=1.0, note=Note(67)),
            TabEvent(1, 0, start=0.9, duration=0.25, start_beat=4.0, duration_beats=1.0, note=Note(64)),
        ],
        measure_count=2,
        tempo_bpm=120.0,
    )
    text = to_alphatex(tab)
    bars = [line for line in text.splitlines() if line.rstrip().endswith("|")]
    assert len(bars) == 2
