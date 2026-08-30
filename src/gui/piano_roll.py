"""Interactive piano-roll visualisation for transcribed notes.

Renders note events as horizontal blocks over a piano-key background,
Songscription-style, so users can see exactly what was detected even if they
do not read TAB.
"""

from __future__ import annotations

from collections.abc import Iterable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QScrollArea, QWidget

_BLACK_KEYS = {1, 3, 6, 8, 10}

_NOTE_COLOR = QColor(79, 130, 229)
_NOTE_BORDER = QColor(45, 90, 180)
_WHITE_ROW = QColor(250, 250, 250)
_BLACK_ROW = QColor(232, 234, 238)
_GRID = QColor(210, 210, 210)
_TEXT = QColor(90, 90, 90)


class PianoRollWidget(QWidget):
    """A single piano-roll canvas; place inside ``QScrollArea`` to navigate."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        pixels_per_second: float = 60.0,
        row_height: int = 12,
    ) -> None:
        super().__init__(parent)
        self.pixels_per_second = float(pixels_per_second)
        self.row_height = int(row_height)
        self._notes: list[tuple[float, float, int]] = []
        self._min_midi = 21
        self._max_midi = 108
        self._duration = 0.0
        self._apply_size()

    def set_notes(
        self, notes: Iterable[tuple[float, float, int]]
    ) -> None:
        """Accept ``(start_seconds, duration_seconds, midi)`` triples."""

        self._notes = [
            (float(start), max(0.05, float(duration)), int(midi))
            for start, duration, midi in notes
        ]
        if self._notes:
            midis = [midi for _s, _d, midi in self._notes]
            self._min_midi = max(21, min(midis) - 3)
            self._max_midi = min(108, max(midis) + 3)
            self._duration = max(start + duration for start, duration, _ in self._notes)
        else:
            self._min_midi, self._max_midi, self._duration = 21, 108, 0.0
        self._apply_size()
        self.update()

    def _apply_size(self) -> None:
        width = int(max(400.0, (self._duration + 1.0) * self.pixels_per_second))
        height = (self._max_midi - self._min_midi + 1) * self.row_height
        self.setMinimumSize(width, height)
        self.resize(width, height)

    def _row_for(self, midi: int) -> int:
        return (self._max_midi - midi) * self.row_height

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        try:
            for midi in range(self._min_midi, self._max_midi + 1):
                top = self._row_for(midi)
                color = _BLACK_ROW if midi % 12 in _BLACK_KEYS else _WHITE_ROW
                painter.fillRect(0, top, self.width(), self.row_height, color)
                painter.setPen(QPen(_GRID))
                painter.drawLine(0, top + self.row_height, self.width(), top + self.row_height)
                if midi % 12 == 0:
                    painter.setPen(QPen(_TEXT))
                    painter.drawText(
                        4, top + self.row_height - 2, f"C{midi // 12 - 1}"
                    )
            if self._duration > 0:
                second = 0
                painter.setPen(QPen(_GRID))
                while second <= self._duration:
                    x = int(second * self.pixels_per_second)
                    painter.drawLine(x, 0, x, self.height())
                    second += 1
            for start, duration, midi in self._notes:
                if not self._min_midi <= midi <= self._max_midi:
                    continue
                x = int(start * self.pixels_per_second)
                w = max(3, int(duration * self.pixels_per_second))
                y = self._row_for(midi) + 1
                painter.fillRect(x, y, w, self.row_height - 2, _NOTE_COLOR)
                painter.setPen(QPen(_NOTE_BORDER))
                painter.drawRect(x, y, w, self.row_height - 2)
        finally:
            painter.end()


class PianoRollView(QScrollArea):
    """Scrollable container hosting the piano-roll canvas."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.canvas = PianoRollWidget(self)
        self.setWidget(self.canvas)
        self.setWidgetResizable(False)
        self.setMinimumHeight(220)

    def set_notes(self, notes: Iterable[tuple[float, float, int]]) -> None:
        self.canvas.set_notes(notes)
        self.horizontalScrollBar().setValue(0)
