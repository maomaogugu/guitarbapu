"""Offscreen checks for the interactive desktop workflow."""

from concurrent.futures import Future
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import soundfile as sf
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.audio.analyzer import AudioAnalysis
from src.audio.loader import load_audio
from src.audio.rhythm import RhythmAnalysis
from src.gui.app import MainWindow
from src.music.note import Note
from src.music.timing import QuantizedNote, Rest, TimingInfo
from src.project import load_project


def _sample_analysis():
    source = Note(64, start=0.2, duration=0.5, confidence=0.9)
    quantized = QuantizedNote(
        source=source,
        note=Note(64, start=0.25, duration=0.5),
        start_beat=0.5,
        duration_beats=1.0,
    )
    return AudioAnalysis(
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


def test_main_window_displays_cleaned_notes_and_tempo():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    analysis = _sample_analysis()

    window._show_analysis(analysis)

    text = window.tab_output.toPlainText()
    assert "原始音符：2  清理后：1" in text
    assert "120.0 BPM" in text
    assert "E4" in text
    assert "时值=1.00拍" in text
    assert "Tuning: E A D G B E" in text
    assert "Mapped: 1" in text
    assert "清理为 1 个音符" in window.status_label.text()
    window.close()
    app.processEvents()


def test_result_actions_enable_only_after_analysis():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    buttons = (
        window.save_project_button,
        window.export_text_button,
        window.export_midi_button,
        window.export_musicxml_button,
    )

    assert all(not button.isEnabled() for button in buttons)
    window._show_analysis(_sample_analysis())
    assert all(button.isEnabled() for button in buttons)
    assert window.event_table.rowCount() == 1
    assert window.insert_event_button.isEnabled()

    window.close()
    app.processEvents()


def test_gui_saves_and_reopens_project_without_audio(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    project_path = tmp_path / "saved-project.guitarbapu.json"
    missing_audio = tmp_path / "missing.wav"
    window = MainWindow()
    window.selected_file = missing_audio
    window._show_analysis(_sample_analysis())
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(project_path), ""),
    )

    window._save_project()

    assert project_path.exists()
    assert "项目已保存" in window.status_label.text()

    reopened = MainWindow()
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(project_path), ""),
    )
    reopened._open_project()

    assert reopened.project_path == project_path.resolve()
    assert reopened.audio is None
    assert reopened.tablature is not None
    assert "文件已移动或缺失" in reopened.file_label.text()
    assert "项目已打开" in reopened.status_label.text()
    assert "E4" in reopened.tab_output.toPlainText()

    window.close()
    reopened.close()
    app.processEvents()


def test_gui_text_export_adds_default_suffix(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._show_analysis(_sample_analysis())
    output_without_suffix = tmp_path / "exported-tab"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(output_without_suffix), ""),
    )

    window._export_text()

    output = Path(str(output_without_suffix) + ".txt")
    assert output.exists()
    assert "Tuning: E A D G B E" in output.read_text(encoding="utf-8")
    assert "导出成功" in window.status_label.text()
    window.close()
    app.processEvents()


def test_gui_edit_refreshes_table_and_supports_undo():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._show_analysis(_sample_analysis())
    assert window.edit_controller is not None

    index = window.edit_controller.change_pitch(0, 67)
    window._edited(index, "测试修改")

    assert window.event_table.item(index, 1).text() == "G4"
    assert "G4" in window.tab_output.toPlainText()
    assert "未保存" in window.status_label.text()
    assert window.undo_button.isEnabled()

    window._undo_edit()
    assert window.event_table.item(0, 1).text() == "E4"
    assert not window.edit_controller.dirty
    window.close()
    app.processEvents()


def test_gui_safe_cancel_discards_completed_result():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    future = Future()
    future.set_result(_sample_analysis())
    window.analysis_future = future
    window.analysis_cancel_requested = True
    window.analysis_progress.setVisible(True)

    window._poll_analysis()

    assert window.analysis is None
    assert window.analysis_future is None
    assert window.status_label.text() == "分析已取消"
    assert not window.analysis_progress.isVisible()
    window.close()
    app.processEvents()


def test_gui_loads_waveform_and_enables_playback_for_audio(tmp_path):
    app = QApplication.instance() or QApplication([])
    audio_path = tmp_path / "short.wav"
    sf.write(audio_path, np.zeros(1000, dtype=np.float32), 1000)
    window = MainWindow()

    window_audio = load_audio(audio_path)
    window._set_audio_source(audio_path, window_audio)

    assert window_audio.duration == 1.0
    assert window.waveform.duration == 1.0
    assert window.audio_player.has_source
    assert window.play_button.isEnabled()
    window.close()
    app.processEvents()


def test_unsaved_prompt_honors_cancel_and_discard(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._show_analysis(_sample_analysis())
    assert window.edit_controller is not None
    window.edit_controller.change_pitch(0, 67)

    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: QMessageBox.StandardButton.Cancel,
    )
    assert window._confirm_discard_changes() is False

    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: QMessageBox.StandardButton.Discard,
    )
    assert window._confirm_discard_changes() is True
    window.edit_controller.undo()
    window.close()
    app.processEvents()


def test_edited_tablature_is_saved_while_analysis_source_is_preserved(
    monkeypatch, tmp_path
):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._show_analysis(_sample_analysis())
    assert window.edit_controller is not None
    index = window.edit_controller.change_pitch(0, 67)
    window._edited(index, "修改")
    project_path = tmp_path / "edited.guitarbapu.json"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(project_path), ""),
    )

    assert window._save_project() is True
    loaded = load_project(project_path)

    assert loaded.analysis.notes[0].midi == 64
    assert loaded.tablature.events[0].note is not None
    assert loaded.tablature.events[0].note.midi == 67
    assert window.edit_controller.dirty is False
    window.close()
    app.processEvents()
