"""Global fingering selection for monophonic guitar note sequences."""

from dataclasses import dataclass
from itertools import product
from typing import Iterable

from .fretboard import FretPosition, Fretboard
from .guitar import Guitar
from .note import Note


@dataclass(frozen=True)
class FingeringWeights:
    """Tunable costs used by the dynamic-programming optimizer."""

    fret_height: float = 0.35
    fret_movement: float = 2.0
    string_movement: float = 0.6
    large_shift: float = 2.5
    open_string_bonus: float = 0.75


class FingeringOptimizer:
    """Choose a playable sequence with minimum total movement cost."""

    def __init__(
        self,
        guitar: Guitar | None = None,
        *,
        weights: FingeringWeights | None = None,
        max_group_candidates: int = 96,
    ) -> None:
        if max_group_candidates < 1:
            raise ValueError("max_group_candidates must be positive")
        self.guitar = guitar or Guitar.standard()
        self.fretboard = Fretboard(self.guitar)
        self.weights = weights or FingeringWeights()
        self.max_group_candidates = int(max_group_candidates)

    def _base_cost(self, position: FretPosition) -> float:
        cost = position.fret * self.weights.fret_height
        if position.fret == 0:
            cost -= self.weights.open_string_bonus
        return cost

    def _transition_cost(
        self, previous: FretPosition, current: FretPosition
    ) -> float:
        fret_delta = abs(current.fret - previous.fret)
        string_delta = abs(current.string - previous.string)
        cost = (
            fret_delta * self.weights.fret_movement
            + string_delta * self.weights.string_movement
        )
        if fret_delta > 4:
            cost += (fret_delta - 4) * self.weights.large_shift
        return cost

    def optimize(
        self, notes: Iterable[Note]
    ) -> tuple[FretPosition | None, ...]:
        """Return one position per note while preserving unplayable entries."""

        note_tuple = tuple(notes)
        if not note_tuple:
            return ()

        result: list[FretPosition | None] = [None] * len(note_tuple)
        playable_indices: list[int] = []
        candidate_sets: list[tuple[FretPosition, ...]] = []
        for index, note in enumerate(note_tuple):
            candidates = self.fretboard.find_positions(note)
            if candidates:
                playable_indices.append(index)
                candidate_sets.append(candidates)

        if not candidate_sets:
            return tuple(result)

        costs: dict[FretPosition, float] = {
            candidate: self._base_cost(candidate)
            for candidate in candidate_sets[0]
        }
        paths: dict[FretPosition, tuple[FretPosition, ...]] = {
            candidate: (candidate,) for candidate in candidate_sets[0]
        }

        for candidates in candidate_sets[1:]:
            next_costs: dict[FretPosition, float] = {}
            next_paths: dict[FretPosition, tuple[FretPosition, ...]] = {}
            for candidate in candidates:
                best_previous = min(
                    costs,
                    key=lambda previous: (
                        costs[previous]
                        + self._transition_cost(previous, candidate)
                        + self._base_cost(candidate),
                        candidate.fret,
                        candidate.string,
                    ),
                )
                next_costs[candidate] = (
                    costs[best_previous]
                    + self._transition_cost(best_previous, candidate)
                    + self._base_cost(candidate)
                )
                next_paths[candidate] = paths[best_previous] + (candidate,)
            costs = next_costs
            paths = next_paths

        final_position = min(
            costs,
            key=lambda position: (costs[position], position.fret, position.string),
        )
        selected = paths[final_position]
        for index, position in zip(playable_indices, selected):
            result[index] = position
        return tuple(result)

    @staticmethod
    def _anchor(
        assignment: tuple[FretPosition | None, ...]
    ) -> tuple[float, float]:
        positions = [position for position in assignment if position is not None]
        if not positions:
            return (0.0, 0.0)
        return (
            sum(position.fret for position in positions) / len(positions),
            sum(position.string for position in positions) / len(positions),
        )

    def _group_cost(
        self, assignment: tuple[FretPosition | None, ...]
    ) -> float:
        positions = [position for position in assignment if position is not None]
        missing_cost = sum(position is None for position in assignment) * 100.0
        if not positions:
            return missing_cost
        frets = [position.fret for position in positions if position.fret > 0]
        spread = max(frets) - min(frets) if len(frets) > 1 else 0
        return (
            sum(self._base_cost(position) for position in positions)
            + spread * 1.5
            + missing_cost
        )

    def _group_transition(
        self,
        previous: tuple[FretPosition | None, ...],
        current: tuple[FretPosition | None, ...],
    ) -> float:
        previous_fret, previous_string = self._anchor(previous)
        current_fret, current_string = self._anchor(current)
        fret_delta = abs(current_fret - previous_fret)
        string_delta = abs(current_string - previous_string)
        cost = (
            fret_delta * self.weights.fret_movement
            + string_delta * self.weights.string_movement
        )
        if fret_delta > 4:
            cost += (fret_delta - 4) * self.weights.large_shift
        return cost

    def _group_assignments(
        self, notes: tuple[Note, ...]
    ) -> tuple[tuple[FretPosition | None, ...], ...]:
        candidates = []
        for note in notes:
            positions = self.fretboard.find_positions(note)
            candidates.append(positions + (None,) if positions else (None,))
        assignments: list[tuple[FretPosition | None, ...]] = []
        for assignment in product(*candidates):
            strings = [
                position.string for position in assignment if position is not None
            ]
            if len(strings) == len(set(strings)):
                assignments.append(tuple(assignment))
        if assignments:
            assignments.sort(
                key=lambda assignment: (
                    self._group_cost(assignment),
                    tuple(
                        (position.fret, position.string)
                        if position is not None
                        else (999, 999)
                        for position in assignment
                    ),
                )
            )
            return tuple(assignments[: self.max_group_candidates])
        return (tuple(None for _ in notes),)

    def optimize_groups(
        self, groups: Iterable[Iterable[Note]]
    ) -> tuple[tuple[FretPosition | None, ...], ...]:
        """Optimize sequential note/chord groups with distinct chord strings."""

        group_tuple = tuple(tuple(group) for group in groups)
        if not group_tuple:
            return ()
        assignment_sets = [self._group_assignments(group) for group in group_tuple]

        costs = {
            assignment: self._group_cost(assignment)
            for assignment in assignment_sets[0]
        }
        paths = {assignment: (assignment,) for assignment in assignment_sets[0]}
        for assignments in assignment_sets[1:]:
            next_costs = {}
            next_paths = {}
            for assignment in assignments:
                previous = min(
                    costs,
                    key=lambda candidate: (
                        costs[candidate]
                        + self._group_transition(candidate, assignment)
                        + self._group_cost(assignment),
                        self._group_cost(assignment),
                    ),
                )
                next_costs[assignment] = (
                    costs[previous]
                    + self._group_transition(previous, assignment)
                    + self._group_cost(assignment)
                )
                next_paths[assignment] = paths[previous] + (assignment,)
            costs = next_costs
            paths = next_paths

        final = min(costs, key=costs.__getitem__)
        return paths[final]


__all__ = ["FingeringOptimizer", "FingeringWeights"]
