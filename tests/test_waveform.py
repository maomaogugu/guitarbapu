"""Offscreen checks for playback formatting and waveform interactions."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from src.gui.audio_player import format_seconds
from src.gui.waveform import WaveformWidget


def test_format_seconds_is_stable():
    assert format_seconds(0) == "00:00.000"
    assert format_seconds(61.234) == "01:01.234"
    assert format_seconds(-1) == "00:00.000"


def test_waveform_accepts_stereo_and_computes_duration():
    app = QApplication.instance() or QApplication([])
    widget = WaveformWidget()
    stereo = np.column_stack(
        (np.linspace(-1, 1, 1000), np.linspace(1, -1, 1000))
    )

    widget.set_audio(stereo, 1000)

    assert widget.duration == 1.0
    widget.close()
    app.processEvents()


def test_waveform_click_requests_matching_time():
    app = QApplication.instance() or QApplication([])
    widget = WaveformWidget()
    widget.resize(400, 120)
    widget.set_audio(np.ones(4000), 1000)
    requested = []
    widget.position_requested.connect(requested.append)
    widget.show()
    app.processEvents()

    QTest.mouseClick(
        widget,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QPoint(200, 60),
    )

    assert requested and requested[-1] == 2.0
    assert widget.selection is None
    widget.close()
    app.processEvents()


def test_waveform_selection_is_normalized_and_clamped():
    app = QApplication.instance() or QApplication([])
    widget = WaveformWidget()
    widget.set_audio(np.ones(1000), 1000)
    selections = []
    widget.selection_changed.connect(lambda start, end: selections.append((start, end)))

    widget.set_selection(0.8, 0.2)

    assert widget.selection == (0.2, 0.8)
    assert selections[-1] == (0.2, 0.8)
    widget.set_cursor(5.0)
    widget.clear_selection()
    assert widget.selection is None
    widget.close()
    app.processEvents()
