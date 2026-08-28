"""Plain-text guitar tablature export."""

from pathlib import Path

from ..music.tab import Tablature
from ..music.tab_renderer import TextTabRenderer
from ..utils.atomic import atomic_replace


def export_text_tab(
    tablature: Tablature,
    path: str | Path,
    *,
    renderer: TextTabRenderer | None = None,
) -> Path:
    """Write the same deterministic TAB text shown by the GUI."""

    target = Path(path).expanduser().resolve(strict=False)
    content = (renderer or TextTabRenderer()).render(tablature)
    return atomic_replace(
        target,
        lambda temporary: temporary.write_text(content + "\n", encoding="utf-8"),
    )


__all__ = ["export_text_tab"]
