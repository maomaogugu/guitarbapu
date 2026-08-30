"""Tests for global fingering optimization."""

from src.music.fingering import FingeringOptimizer
from src.music.fretboard import FretPosition
from src.music.note import Note


def test_optimizer_keeps_simple_melody_on_a_smooth_position():
    positions = FingeringOptimizer().optimize(
        (Note(64), Note(67), Note(69))
    )

    assert positions == (
        FretPosition(string=1, fret=0),
        FretPosition(string=1, fret=3),
        FretPosition(string=1, fret=5),
    )


def test_optimizer_preserves_unplayable_notes():
    positions = FingeringOptimizer().optimize((Note(39), Note(40)))

    assert positions == (None, FretPosition(string=6, fret=0))


def test_chord_group_uses_distinct_strings():
    result = FingeringOptimizer().optimize_groups(
        ((Note(64), Note(67), Note(71)),)
    )[0]

    strings = [position.string for position in result if position is not None]
    assert len(strings) == 3
    assert len(set(strings)) == 3


def test_chord_conflict_maps_what_is_playable_and_marks_one_missing():
    result = FingeringOptimizer().optimize_groups(
        ((Note(40), Note(40)),)
    )[0]

    assert sum(position is not None for position in result) == 1
    assert sum(position is None for position in result) == 1


def test_chord_candidates_are_pruned_before_sequence_optimization():
    optimizer = FingeringOptimizer(max_group_candidates=8)

    assignments = optimizer._group_assignments(
        tuple(Note(midi) for midi in (48, 52, 55, 60, 64, 67))
    )

    assert 1 <= len(assignments) <= 8


def test_dense_neural_cluster_never_explodes_and_fits_six_strings():
    import time

    optimizer = FingeringOptimizer()
    cluster = tuple(
        Note(midi, start=0.0) for midi in range(48, 72)
    )
    started = time.perf_counter()
    (assignment,) = optimizer.optimize_groups((cluster,))
    elapsed = time.perf_counter() - started

    assert len(assignment) == 24
    assert elapsed < 5.0
    assert sum(position is not None for position in assignment) <= 6
