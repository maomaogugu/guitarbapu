"""Lightweight GUI checks for displaying Phase 4 analysis results."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.audio.analyzer import AudioAnalysis
from src.audio.rhythm import RhythmAnalysis
from src.gui.app import MainWindow
from src.music.note import Note
from src.music.timing import QuantizedNote, Rest, TimingInfo


def test_main_window_displays_cleaned_notes_and_tempo():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    source = Note(64, start=0.2, duration=0.5, confidence=0.9)
    quantized = QuantizedNote(
        source=source,
        note=Note(64, start=0.25, duration=0.5),
        start_beat=0.5,
        duration_beats=1.0,
    )
    analysis = AudioAnalysis(
        duration_seconds=1.0,
        sample_rate=44_100,
        notes=(source,),
        raw_notes=(Note(65, duration=0.01), source),
        rhythm=RhythmAnalysis(
            timing=TimingInfo(tempo_bpm=120),
            quantized_notes=(quantized,),
            rests=(Rest(0.0, 0.2),),
        ),
    )

    window._show_analysis(analysis)

    text = window.tab_output.toPlainText()
    assert "原始音符：2  清理后：1" in text
    assert "120.0 BPM" in text
    assert "E4" in text
    assert "时值=1.00拍" in text
    assert "清理为 1 个音符" in window.status_label.text()
    window.close()
    app.processEvents()
