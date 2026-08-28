"""Map musical notes to playable guitar string and fret positions."""

from dataclasses import dataclass
from typing import Iterable

from .guitar import Guitar
from .note import Note


@dataclass(frozen=True, order=True)
class FretPosition:
    """A playable position identified by its string number and fret."""

    string: int
    fret: int

    def __post_init__(self) -> None:
        if self.string < 1:
            raise ValueError("string must be positive")
        if self.fret < 0:
            raise ValueError("fret must be non-negative")


class Fretboard:
    """Enumerate positions and choose a low-movement fingering."""

    def __init__(self, guitar: Guitar | None = None) -> None:
        self.guitar = guitar or Guitar.standard()

    @staticmethod
    def _midi(note: Note | int) -> int:
        midi = note.midi if isinstance(note, Note) else note
        if not isinstance(midi, int) or not 0 <= midi <= 127:
            raise ValueError("note must contain a MIDI value between 0 and 127")
        return midi

    def find_positions(self, note: Note | int) -> tuple[FretPosition, ...]:
        """Return all positions that produce ``note``, ordered by low fret."""

        midi = self._midi(note)
        positions: list[FretPosition] = []
        for guitar_string in self.guitar.strings:
            fret = midi - guitar_string.tuning_midi - self.guitar.capo
            if 0 <= fret <= self.guitar.fret_count:
                positions.append(FretPosition(string=guitar_string.number, fret=fret))
        return tuple(sorted(positions, key=lambda position: (position.fret, position.string)))

    def position_midi(self, position: FretPosition) -> int:
        """Return the sounding MIDI value for a validated position."""

        return self.guitar.midi_at(position.string, position.fret)

    def score_position(
        self,
        position: FretPosition,
        previous: FretPosition | None = None,
    ) -> tuple[float, int, int]:
        """Return a sortable cost: fret height first, movement second."""

        if previous is None:
            return (float(position.fret), position.fret, position.string)
        fret_motion = abs(position.fret - previous.fret)
        string_motion = abs(position.string - previous.string)
        return (
            float(position.fret) + 2.0 * fret_motion + 0.5 * string_motion,
            fret_motion,
            string_motion,
        )

    def choose_position(
        self,
        note: Note | int,
        previous: FretPosition | None = None,
    ) -> FretPosition | None:
        """Choose the lowest-cost position, or ``None`` if it is unplayable."""

        positions = self.find_positions(note)
        if not positions:
            return None
        return min(positions, key=lambda position: self.score_position(position, previous))

    def map_notes(self, notes: Iterable[Note]) -> tuple[FretPosition | None, ...]:
        """Choose positions for a sequence while minimizing hand movement."""

        selected: list[FretPosition | None] = []
        previous: FretPosition | None = None
        for note in notes:
            position = self.choose_position(note, previous)
            selected.append(position)
            if position is not None:
                previous = position
        return tuple(selected)


__all__ = ["FretPosition", "Fretboard"]
