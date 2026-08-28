"""Standard MIDI export for generated tablature."""

from pathlib import Path

from ..music.tab import Tablature
from ..utils.atomic import atomic_replace
from .score import tablature_to_score


def export_midi(tablature: Tablature, path: str | Path) -> Path:
    """Write pitches and beat timing as a Standard MIDI File."""

    target = Path(path).expanduser().resolve(strict=False)
    return atomic_replace(
        target,
        lambda temporary: tablature_to_score(tablature).write(
            "midi", fp=str(temporary)
        ),
    )


__all__ = ["export_midi"]
