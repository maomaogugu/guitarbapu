"""Guitar instrument configuration and string/fret domain objects."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GuitarString:
    """One guitar string, numbered from the highest-pitched string as 1."""

    number: int
    name: str
    tuning_midi: int

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError("string number must be positive")
        if not 0 <= self.tuning_midi <= 127:
            raise ValueError("tuning_midi must be between 0 and 127")
        if not self.name.strip():
            raise ValueError("string name must not be empty")


@dataclass(frozen=True)
class Guitar:
    """A fretted guitar definition used when mapping notes to tablature."""

    strings: tuple[GuitarString, ...] = field(default_factory=tuple)
    fret_count: int = 24
    capo: int = 0

    @classmethod
    def standard(cls, *, fret_count: int = 24, capo: int = 0) -> "Guitar":
        """Return a six-string guitar in standard E tuning."""

        strings = (
            GuitarString(1, "high E", 64),
            GuitarString(2, "B", 59),
            GuitarString(3, "G", 55),
            GuitarString(4, "D", 50),
            GuitarString(5, "A", 45),
            GuitarString(6, "low E", 40),
        )
        return cls(strings=strings, fret_count=fret_count, capo=capo)

    def midi_at(self, string: int, fret: int) -> int:
        """Return the sounding MIDI pitch at a string/fret position."""

        if not 1 <= string <= len(self.strings):
            raise ValueError("string is outside this guitar's string range")
        if not 0 <= fret <= self.fret_count:
            raise ValueError("fret is outside this guitar's fret range")
        guitar_string = next(item for item in self.strings if item.number == string)
        return guitar_string.tuning_midi + self.capo + fret

    def __post_init__(self) -> None:
        if not self.strings:
            raise ValueError("a guitar must define at least one string")
        if self.fret_count < 1:
            raise ValueError("fret_count must be positive")
        if self.capo < 0:
            raise ValueError("capo must be non-negative")
        numbers = [guitar_string.number for guitar_string in self.strings]
        if len(set(numbers)) != len(numbers):
            raise ValueError("guitar string numbers must be unique")
