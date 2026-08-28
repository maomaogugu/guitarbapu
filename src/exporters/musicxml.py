"""MusicXML export for notation programs such as MuseScore."""

from pathlib import Path

from ..music.tab import Tablature
from .score import tablature_to_score


def export_musicxml(tablature: Tablature, path: str | Path) -> Path:
    """Write standard notation plus string/fret technical indications."""

    target = Path(path).expanduser().resolve(strict=False)
    tablature_to_score(tablature).write("musicxml", fp=str(target))
    return target


__all__ = ["export_musicxml"]
