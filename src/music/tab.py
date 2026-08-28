"""Guitar tablature data structures.

Rendering and automatic note-to-string mapping are intentionally deferred.
"""

from dataclasses import dataclass, field

from .guitar import Guitar


@dataclass(frozen=True)
class TabEvent:
    """A note placed on one string and fret."""

    string: int
    fret: int
    start: float = 0.0
    duration: float = 0.0


@dataclass(frozen=True)
class Tablature:
    """A collection of tab events for a particular guitar definition."""

    guitar: Guitar = field(default_factory=Guitar.standard)
    events: tuple[TabEvent, ...] = field(default_factory=tuple)
