"""Tests for Phase 4 beat quantization and rest construction."""

import pytest

from src.music.note import Note
from src.music.timing import TimingInfo, find_rests, quantize_notes


def test_quantize_notes_to_sixteenth_note_grid():
    timing = TimingInfo(tempo_bpm=120, subdivision=4)
    source = Note(64, start=0.26, duration=0.24)

    result = quantize_notes((source,), timing)[0]

    assert result.source is source
    assert result.start_beat == pytest.approx(0.5)
    assert result.duration_beats == pytest.approx(0.5)
    assert result.note.start == pytest.approx(0.25)
    assert result.note.duration == pytest.approx(0.25)


def test_unknown_tempo_preserves_second_timing():
    source = Note(64, start=0.26, duration=0.24)

    result = quantize_notes((source,), TimingInfo())[0]

    assert result.note is source
    assert result.start_beat is None
    assert result.duration_beats is None


def test_note_crossing_measure_is_marked_for_tie():
    timing = TimingInfo(tempo_bpm=120, time_signature=(4, 4), subdivision=4)
    source = Note(64, start=1.75, duration=0.5)

    result = quantize_notes((source,), timing)[0]

    assert result.start_beat == pytest.approx(3.5)
    assert result.duration_beats == pytest.approx(1.0)
    assert result.tie_to_next is True


def test_find_rests_includes_leading_internal_and_trailing_silence():
    notes = (
        Note(64, start=0.2, duration=0.3),
        Note(67, start=0.8, duration=0.2),
    )

    rests = find_rests(
        notes,
        total_duration=1.3,
        timing=TimingInfo(tempo_bpm=120),
        min_duration=0.1,
    )

    assert len(rests) == 3
    assert rests[0].start == pytest.approx(0.0)
    assert rests[0].duration == pytest.approx(0.2)
    assert rests[1].start == pytest.approx(0.5)
    assert rests[1].duration == pytest.approx(0.3)
    assert rests[2].start == pytest.approx(1.0)
    assert rests[2].duration == pytest.approx(0.3)
    assert rests[0].duration_beats == pytest.approx(0.5)
