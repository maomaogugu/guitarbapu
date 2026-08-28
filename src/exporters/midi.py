"""Standard MIDI export for generated tablature."""

from pathlib import Path

from ..music.tab import Tablature
from .score import tablature_to_score


def export_midi(tablature: Tablature, path: str | Path) -> Path:
    """Write pitches and beat timing as a Standard MIDI File."""

    target = Path(path).expanduser().resolve(strict=False)
    tablature_to_score(tablature).write("midi", fp=str(target))
    return target


__all__ = ["export_midi"]
