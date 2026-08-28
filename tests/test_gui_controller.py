"""Tests for widget-independent TAB editing state."""

import pytest

from src.gui.controller import TabEditController, TabEditError
from src.music.fretboard import FretPosition
from src.music.note import Note
from src.music.tab import TabEvent, Tablature


def _tablature():
    note = Note(64, start=0.0, duration=0.5, confidence=0.8)
    return Tablature(
        events=(
            TabEvent(
                string=1,
                fret=0,
                note=note,
                start=0.0,
                duration=0.5,
                start_beat=0.0,
                duration_beats=1.0,
                confidence=0.8,
            ),
        ),
        tempo_bpm=120.0,
        time_signature=(4, 4),
        subdivision=4,
        measure_count=1,
    )


def test_pitch_change_chooses_nearby_playable_position():
    controller = TabEditController(_tablature())

    index = controller.change_pitch(0, 67)
    event = controller.tablature.events[index]

    assert event.note is not None and event.note.midi == 67
    assert (event.string, event.fret) == (1, 3)
    assert event.confidence is None
    assert controller.dirty is True


def test_position_change_only_accepts_same_pitch():
    controller = TabEditController(_tablature())

    index = controller.change_position(0, FretPosition(2, 5))

    assert (controller.tablature.events[index].string, controller.tablature.events[index].fret) == (2, 5)
    with pytest.raises(TabEditError, match="不一致"):
        controller.change_position(index, FretPosition(2, 6))


def test_timing_edit_recalculates_seconds_measure_and_tie():
    controller = TabEditController(_tablature())

    index = controller.update_event(
        0,
        midi=64,
        string=1,
        fret=0,
        start_beat=3.5,
        duration_beats=1.0,
        technique="slide",
    )
    event = controller.tablature.events[index]

    assert event.start == 1.75
    assert event.duration == 0.5
    assert event.measure == 1
    assert event.tie_to_next is True
    assert event.technique == "slide"
    assert controller.tablature.measure_count == 2


def test_manual_technique_change_clears_automatic_confidence():
    source = Note(64, start=0.0, duration=0.5)
    tablature = Tablature(
        events=(
            TabEvent(
                string=1,
                fret=0,
                note=source,
                start_beat=0.0,
                duration_beats=1.0,
                technique="vibrato",
                technique_confidence=0.8,
            ),
        ),
        tempo_bpm=120.0,
    )
    controller = TabEditController(tablature)

    changed = controller.update_event(
        0,
        midi=64,
        string=1,
        fret=0,
        start_beat=0.0,
        duration_beats=1.0,
        technique="bend",
    )

    assert controller.tablature.events[changed].technique == "bend"
    assert controller.tablature.events[changed].technique_confidence is None


def test_insert_delete_undo_redo_and_saved_state():
    controller = TabEditController(_tablature())

    inserted_index = controller.insert_event(
        midi=67,
        start_beat=1.0,
        duration_beats=0.5,
    )
    assert len(controller.tablature.events) == 2
    assert controller.can_undo is True
    controller.mark_saved()
    assert controller.dirty is False

    controller.delete_event(inserted_index)
    assert len(controller.tablature.events) == 1
    assert controller.dirty is True
    controller.undo()
    assert len(controller.tablature.events) == 2
    assert controller.dirty is False
    controller.redo()
    assert len(controller.tablature.events) == 1


def test_unplayable_pitch_is_rejected_without_history_change():
    controller = TabEditController(_tablature())

    with pytest.raises(TabEditError, match="超出"):
        controller.change_pitch(0, 20)

    assert controller.tablature == _tablature()
    assert controller.can_undo is False


def test_insert_avoids_same_string_overlap_when_another_position_exists():
    controller = TabEditController(_tablature())

    index = controller.insert_event(
        midi=67,
        start_beat=0.0,
        duration_beats=1.0,
    )
    inserted = controller.tablature.events[index]

    assert inserted.string != 1
    assert inserted.note is not None and inserted.note.midi == 67


def test_explicit_overlapping_string_position_is_rejected():
    controller = TabEditController(_tablature())

    with pytest.raises(TabEditError, match="已有其他音符"):
        controller.insert_event(
            midi=67,
            position=FretPosition(1, 3),
            start_beat=0.0,
            duration_beats=1.0,
        )
