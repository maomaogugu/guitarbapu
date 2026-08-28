"""Tests for converting Phase 4 analysis into structured tablature."""

from src.audio.analyzer import AudioAnalysis
from src.audio.rhythm import RhythmAnalysis
from src.music.note import Note
from src.music.tab_generator import TabGenerator
from src.music.timing import QuantizedNote, Rest, TimingInfo


def _analysis(items, *, tempo=120.0, rests=()):
    notes = tuple(item.source for item in items)
    return AudioAnalysis(
        duration_seconds=4.0,
        sample_rate=44_100,
        notes=notes,
        raw_notes=notes,
        rhythm=RhythmAnalysis(
            timing=TimingInfo(
                tempo_bpm=tempo,
                time_signature=(4, 4),
                subdivision=4,
            ),
            quantized_notes=tuple(items),
            rests=tuple(rests),
        ),
    )


def _item(midi, beat, duration=1.0, *, tie=False):
    source = Note(midi, start=beat * 0.5, duration=duration * 0.5)
    return QuantizedNote(
        source=source,
        note=source,
        start_beat=beat,
        duration_beats=duration,
        tie_to_next=tie,
    )


def test_open_string_tuning_sequence_maps_to_six_strings():
    items = tuple(
        _item(midi, beat)
        for beat, midi in enumerate((40, 45, 50, 55, 59, 64))
    )

    tab = TabGenerator().generate(_analysis(items))

    assert [(event.string, event.fret) for event in tab.events] == [
        (6, 0),
        (5, 0),
        (4, 0),
        (3, 0),
        (2, 0),
        (1, 0),
    ]
    assert tab.measure_count == 2


def test_generator_retains_rest_and_tie_information():
    tab = TabGenerator().generate(
        _analysis(
            (_item(64, 3.5, duration=1.0, tie=True),),
            rests=(Rest(0.0, 0.5, 0.0, 1.0),),
        )
    )

    assert tab.events[0].measure == 1
    assert tab.events[0].tie_to_next is True
    assert tab.rests[0].measure == 1
    assert tab.measure_count == 2


def test_generator_marks_cross_measure_tie_even_when_source_does_not():
    tab = TabGenerator().generate(
        _analysis((_item(64, 3.5, duration=1.0, tie=False),))
    )

    assert tab.events[0].tie_to_next is True


def test_unplayable_note_is_retained_as_diagnostic():
    tab = TabGenerator().generate(_analysis((_item(39, 0),)))

    assert tab.events == ()
    assert len(tab.unmapped_notes) == 1
    assert "超出" in tab.unmapped_notes[0].reason


def test_simultaneous_notes_create_distinct_string_events():
    tab = TabGenerator().generate(
        _analysis((_item(64, 0), _item(67, 0), _item(71, 0)))
    )

    assert len(tab.events) == 3
    assert len({event.string for event in tab.events}) == 3
