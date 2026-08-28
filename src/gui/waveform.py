"""Dependency-free Qt waveform view with cursor and loop selection."""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget


class WaveformWidget(QWidget):
    """Render a compact waveform and emit seek/selection times in seconds."""

    position_requested = pyqtSignal(float)
    selection_changed = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setMouseTracking(True)
        self._minimums = np.empty(0, dtype=np.float32)
        self._maximums = np.empty(0, dtype=np.float32)
        self._duration = 0.0
        self._cursor = 0.0
        self._selection: tuple[float, float] | None = None
        self._event_times: tuple[float, ...] = ()
        self._drag_anchor: float | None = None
        self._dragging = False

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def selection(self) -> tuple[float, float] | None:
        return self._selection

    def sizeHint(self) -> QSize:
        return QSize(700, 150)

    def clear(self) -> None:
        self._minimums = np.empty(0, dtype=np.float32)
        self._maximums = np.empty(0, dtype=np.float32)
        self._duration = 0.0
        self._cursor = 0.0
        self._selection = None
        self._event_times = ()
        self.update()

    def set_audio(
        self,
        waveform,
        sample_rate: int,
        *,
        max_columns: int = 4000,
    ) -> None:
        samples = np.asarray(waveform, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        if samples.ndim != 1:
            raise ValueError("waveform must be one- or two-dimensional")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if samples.size == 0:
            self.clear()
            return
        self._duration = samples.size / sample_rate
        bucket_size = max(1, math.ceil(samples.size / max_columns))
        padded_size = math.ceil(samples.size / bucket_size) * bucket_size
        if padded_size != samples.size:
            samples = np.pad(samples, (0, padded_size - samples.size))
        buckets = samples.reshape(-1, bucket_size)
        minimums = buckets.min(axis=1)
        maximums = buckets.max(axis=1)
        peak = float(max(np.max(np.abs(minimums)), np.max(np.abs(maximums))))
        if math.isfinite(peak) and peak > 0:
            minimums = minimums / peak
            maximums = maximums / peak
        self._minimums = minimums.astype(np.float32)
        self._maximums = maximums.astype(np.float32)
        self._cursor = min(self._cursor, self._duration)
        self._selection = None
        self.update()

    def set_event_times(self, event_times) -> None:
        self._event_times = tuple(
            max(0.0, min(float(value), self._duration)) for value in event_times
        )
        self.update()

    def set_cursor(self, seconds: float) -> None:
        self._cursor = max(0.0, min(float(seconds), self._duration))
        self.update()

    def set_selection(self, start: float, end: float) -> None:
        first = max(0.0, min(float(start), self._duration))
        second = max(0.0, min(float(end), self._duration))
        if abs(second - first) < 1e-6:
            self.clear_selection()
            return
        self._selection = (min(first, second), max(first, second))
        self.selection_changed.emit(*self._selection)
        self.update()

    def clear_selection(self) -> None:
        self._selection = None
        self.selection_changed.emit(0.0, 0.0)
        self.update()

    def seconds_at_x(self, x: float) -> float:
        if self._duration <= 0 or self.width() <= 0:
            return 0.0
        return max(0.0, min(float(x) / self.width(), 1.0)) * self._duration

    def _x_at_seconds(self, seconds: float) -> float:
        if self._duration <= 0:
            return 0.0
        return max(0.0, min(seconds / self._duration, 1.0)) * self.width()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._duration <= 0:
            super().mousePressEvent(event)
            return
        self._drag_anchor = self.seconds_at_x(event.position().x())
        self._dragging = False
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_anchor is None:
            super().mouseMoveEvent(event)
            return
        current = self.seconds_at_x(event.position().x())
        if abs(current - self._drag_anchor) >= max(0.02, self._duration / 500):
            self._dragging = True
            self.set_selection(self._drag_anchor, current)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._drag_anchor is None:
            super().mouseReleaseEvent(event)
            return
        current = self.seconds_at_x(event.position().x())
        if self._dragging:
            self.set_selection(self._drag_anchor, current)
        else:
            self.clear_selection()
            self.position_requested.emit(current)
        self._drag_anchor = None
        self._dragging = False
        event.accept()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        painter.setPen(QPen(QColor("#d1d5db"), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        middle = self.height() / 2
        painter.drawLine(0, round(middle), self.width(), round(middle))

        if self._selection is not None:
            start, end = self._selection
            selection_rect = QRectF(
                self._x_at_seconds(start),
                0,
                self._x_at_seconds(end) - self._x_at_seconds(start),
                self.height(),
            )
            painter.fillRect(selection_rect, QColor(59, 130, 246, 45))

        if self._minimums.size:
            painter.setPen(QPen(QColor("#2563eb"), 1))
            column_count = self._minimums.size
            for pixel_x in range(self.width()):
                index = min(
                    column_count - 1,
                    int(pixel_x / max(1, self.width()) * column_count),
                )
                top = middle - float(self._maximums[index]) * (middle - 8)
                bottom = middle - float(self._minimums[index]) * (middle - 8)
                painter.drawLine(pixel_x, round(top), pixel_x, round(bottom))

        painter.setPen(QPen(QColor(16, 185, 129, 150), 1))
        painter.setBrush(QColor(16, 185, 129, 180))
        for seconds in self._event_times:
            x = self._x_at_seconds(seconds)
            painter.drawLine(round(x), 0, round(x), 7)
            painter.drawPolygon(
                QPolygonF(
                    (QPointF(x - 3, 0), QPointF(x + 3, 0), QPointF(x, 6))
                )
            )

        if self._duration > 0:
            cursor_x = round(self._x_at_seconds(self._cursor))
            painter.setPen(QPen(QColor("#dc2626"), 2))
            painter.drawLine(cursor_x, 0, cursor_x, self.height())


__all__ = ["WaveformWidget"]
