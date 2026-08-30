"""Offscreen checks for the interactive desktop workflow."""

from concurrent.futures import Future
import os
from pathlib import Path
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import soundfile as sf
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.audio.analyzer import AudioAnalysis
from src.audio.demucs_separator import DemucsSeparator
from src.audio.loader import load_audio
from src.audio.rhythm import RhythmAnalysis
from src.audio.separator import (
    SeparationError,
    SeparationProgress,
    SeparationResult,
    Stem,
)
from src.audio.transcription_service import TranscriptionResult
from src.audio.track_classifier import TrackClassifier
from src.gui.app import MainWindow
from src.gui.diagnostics_dialog import DiagnosticsDialog
from src.music.chord import Chord
from src.music.note import Note
from src.music.tab_generator import TabGenerator
from src.music.technique import GuitarTechnique, TechniqueDetection
from src.music.timing import QuantizedNote, Rest, TimingInfo
from src.music.track import TrackRole
from src.project import TranscriptionTrack, load_project
from src.utils.diagnostics import collect_diagnostics
from src.utils.model_manager import HTDEMUCS_6S, ModelStatus


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


def _mixed_analysis():
    lead = Note(69, start=0.0, duration=0.5, confidence=0.9)
    chord_notes = tuple(
        Note(midi, start=1.0, duration=1.0, confidence=0.8)
        for midi in (48, 52, 55)
    )
    notes = (lead,) + chord_notes
    timing = TimingInfo(tempo_bpm=120.0, time_signature=(4, 4))
    return AudioAnalysis(
        duration_seconds=2.5,
        sample_rate=44_100,
        notes=notes,
        raw_notes=notes,
        rhythm=RhythmAnalysis(
            timing=timing,
            quantized_notes=tuple(
                QuantizedNote(
                    source=note,
                    note=note,
                    start_beat=note.start * 2,
                    duration_beats=note.duration * 2,
                )
                for note in notes
            ),
        ),
        chords=(Chord.from_midis((48, 52, 55), start=1.0, duration=1.0),),
    )


def _mixed_result():
    analysis = _mixed_analysis()
    generator = TabGenerator()
    tracks = tuple(
        TranscriptionTrack(
            track_id=f"logical-{candidate.role.value}",
            name=candidate.name,
            role=candidate.role,
            analysis=candidate.analysis,
            tablature=generator.generate(candidate.analysis),
            confidence=candidate.confidence,
            metadata={"logical": True, "independent_audio": False},
        )
        for candidate in TrackClassifier().classify(analysis)
    )
    return TranscriptionResult(
        source_audio_path=Path("source.wav"),
        analyzed_audio_path=Path("source.wav"),
        analysis=analysis,
        tablature=generator.generate(analysis),
        tracks=tracks,
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


def test_main_window_displays_automatic_technique_confidence():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    analysis = _sample_analysis()
    source = analysis.notes[0]
    analysis = AudioAnalysis(
        duration_seconds=analysis.duration_seconds,
        sample_rate=analysis.sample_rate,
        notes=analysis.notes,
        raw_notes=analysis.raw_notes,
        rhythm=analysis.rhythm,
        techniques=(
            TechniqueDetection(
                GuitarTechnique.VIBRATO,
                source,
                0.84,
            ),
        ),
    )

    window._show_analysis(analysis)

    assert window.event_table.item(0, 8).text() == "vibrato (84%)"
    assert "技巧候选：1" in window.tab_output.toPlainText()
    assert "技巧=vibrato(84%)" in window.tab_output.toPlainText()
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


def test_help_menu_exposes_product_support_actions():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert window.diagnostics_action.text() == "系统诊断…"
    assert window.prepare_model_action.text() == "准备 Demucs 模型…"
    assert window.open_logs_action.text() == "打开日志目录"

    window.close()
    app.processEvents()


def test_diagnostics_dialog_can_save_report(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])

    class Manager:
        def status(self):
            return ModelStatus(
                HTDEMUCS_6S,
                True,
                True,
                HTDEMUCS_6S.required_files,
            )

    diagnostics = collect_diagnostics(
        model_manager=Manager(),
        paths=(tmp_path,),
    )
    target = tmp_path / "diagnostics.txt"
    dialog = DiagnosticsDialog(diagnostics, log_dir=tmp_path / "logs")
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(target), ""),
    )

    dialog._save_report()

    assert target.exists()
    assert "系统诊断" in target.read_text(encoding="utf-8")
    dialog.close()
    app.processEvents()


def test_gui_prepares_optional_model_in_background(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    missing = ModelStatus(HTDEMUCS_6S, True, False)
    ready = ModelStatus(
        HTDEMUCS_6S,
        True,
        True,
        HTDEMUCS_6S.required_files,
    )

    class Manager:
        def status(self):
            return missing

        def prepare(self):
            return ready

    window.model_manager = Manager()
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)

    window._prepare_demucs_model()
    assert window.model_future is not None
    window.model_future.result(timeout=2)
    window._poll_model_preparation()

    assert window.model_future is None
    assert window.prepare_model_action.isEnabled()
    assert "已准备" in window.status_label.text()
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


def test_demucs_option_is_disabled_when_optional_backend_is_missing(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(DemucsSeparator, "is_available", staticmethod(lambda: False))

    window = MainWindow()

    assert not window.separate_guitar_checkbox.isEnabled()
    assert "requirements-separation.txt" in window.separate_guitar_checkbox.toolTip()
    window.close()
    app.processEvents()


def test_gui_displays_separation_progress():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.analysis_future = Future()
    window.analysis_progress_state = SeparationProgress(
        "separating", 0.5, "正在分离吉他音轨…"
    )

    window._poll_analysis()

    assert window.analysis_progress.maximum() == 100
    assert window.analysis_progress.value() == 50
    assert window.status_label.text() == "正在分离吉他音轨…"
    window.close()
    app.processEvents()


def test_gui_adds_guitar_stem_without_replacing_original_project_audio(tmp_path):
    app = QApplication.instance() or QApplication([])
    original_path = tmp_path / "original.wav"
    guitar_path = tmp_path / "guitar.wav"
    sf.write(original_path, np.zeros(1000, dtype=np.float32), 1000)
    sf.write(guitar_path, np.zeros(1000, dtype=np.float32), 1000)
    window = MainWindow()
    window._set_audio_source(original_path, load_audio(original_path))
    analysis = _sample_analysis()
    separation = SeparationResult(
        source_path=original_path,
        model_name="htdemucs_6s",
        device="cpu",
        cache_key="a" * 64,
        stems=(Stem("guitar", guitar_path, 1000, 1, 1.0),),
    )
    result = TranscriptionResult(
        source_audio_path=original_path,
        analyzed_audio_path=guitar_path,
        analysis=analysis,
        tablature=TabGenerator().generate(analysis),
        separation=separation,
    )

    window._show_transcription_result(result)

    assert window.selected_file == original_path
    assert window.audio is not None
    assert window.playback_source_combo.currentText() == "吉他分离轨"
    assert set(window.playback_sources) == {"原音频", "吉他分离轨"}
    assert window.analysis_parameters["separation"]["cache_key"] == "a" * 64
    window.close()
    app.processEvents()


def test_gui_cancel_sets_cooperative_event():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.analysis_future = Future()
    window.analysis_cancel_event = Event()

    window._cancel_analysis()

    assert window.analysis_cancel_event.is_set()
    assert window.analysis_cancel_requested
    window.close()
    app.processEvents()


def test_gui_separation_error_recommends_original_audio_fallback():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    future = Future()
    future.set_exception(SeparationError("模型失败"))
    window.analysis_future = future

    window._poll_analysis()

    assert "可取消勾选后分析原音频" in window.status_label.text()
    window.close()
    app.processEvents()


def test_project_save_keeps_original_audio_after_stem_playback(
    monkeypatch, tmp_path
):
    app = QApplication.instance() or QApplication([])
    original_path = tmp_path / "original.wav"
    guitar_path = tmp_path / "guitar.wav"
    project_path = tmp_path / "separated.guitarbapu.json"
    sf.write(original_path, np.zeros(1000, dtype=np.float32), 1000)
    sf.write(guitar_path, np.zeros(1000, dtype=np.float32), 1000)
    window = MainWindow()
    window._set_audio_source(original_path, load_audio(original_path))
    window._show_analysis(_sample_analysis())
    window._set_playback_sources(
        {
            "原音频": (original_path, load_audio(original_path)),
            "吉他分离轨": (guitar_path, load_audio(guitar_path)),
        },
        selected="吉他分离轨",
    )
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(project_path), ""),
    )

    assert window._save_project()
    assert load_project(project_path).audio_path == original_path
    window.close()
    app.processEvents()


def test_gui_displays_experimental_chord_summary():
    app = QApplication.instance() or QApplication([])
    notes = tuple(Note(midi, start=0.0, duration=1.0) for midi in (48, 52, 55))
    analysis = AudioAnalysis(
        duration_seconds=1.0,
        sample_rate=44_100,
        notes=notes,
        raw_notes=notes,
        chords=(Chord.from_midis((48, 52, 55), duration=1.0),),
    )
    window = MainWindow()

    window._show_analysis(analysis)

    assert "1 个和弦" in window.status_label.text()
    assert "和弦检测（实验）" in window.tab_output.toPlainText()
    assert "C" in window.tab_output.toPlainText()
    assert window.event_table.rowCount() == 3
    window.close()
    app.processEvents()


def test_gui_polyphonic_mode_builds_polyphonic_analyzer(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    audio_path = tmp_path / "chord.wav"
    sf.write(audio_path, np.zeros(1000, dtype=np.float32), 1000)
    captured = {}
    window = MainWindow()
    window._set_audio_source(audio_path, load_audio(audio_path))
    window.analysis_mode_combo.setCurrentIndex(
        window.analysis_mode_combo.findData("polyphonic")
    )

    def fake_submit(function, *args, **kwargs):
        captured["analyzer"] = function.__self__.analyzer
        return Future()

    monkeypatch.setattr(window.analysis_executor, "submit", fake_submit)
    window._start_analysis()

    assert captured["analyzer"].__class__.__name__ == "PolyphonicAudioAnalyzer"
    assert window.analysis_parameters["analysis_mode"] == "polyphonic"
    assert window.analysis_parameters["max_polyphony"] == 6
    window.close()
    app.processEvents()


def test_gui_capo_spinbox_flows_into_tab_generator(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    audio_path = tmp_path / "capo.wav"
    sf.write(audio_path, np.zeros(1000, dtype=np.float32), 1000)
    captured = {}
    window = MainWindow()
    window._set_audio_source(audio_path, load_audio(audio_path))
    window.capo_spinbox.setValue(3)

    def fake_submit(function, *args, **kwargs):
        captured["service"] = function.__self__
        return Future()

    monkeypatch.setattr(window.analysis_executor, "submit", fake_submit)
    window._start_analysis()

    service = captured["service"]
    assert service.tab_generator.guitar.capo == 3
    assert window.analysis_parameters["capo"] == 3
    window.close()
    app.processEvents()


def test_gui_basic_pitch_mode_builds_basic_pitch_analyzer(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    audio_path = tmp_path / "chord.wav"
    sf.write(audio_path, np.zeros(1000, dtype=np.float32), 1000)
    captured = {}
    window = MainWindow()
    window._set_audio_source(audio_path, load_audio(audio_path))
    window.analysis_mode_combo.setCurrentIndex(
        window.analysis_mode_combo.findData("basic_pitch")
    )

    def fake_submit(function, *args, **kwargs):
        captured["analyzer"] = function.__self__.analyzer
        return Future()

    monkeypatch.setattr(window.analysis_executor, "submit", fake_submit)
    window._start_analysis()

    assert captured["analyzer"].__class__.__name__ == "BasicPitchAnalyzer"
    assert window.analysis_parameters["analysis_mode"] == "basic_pitch"
    assert window.analysis_parameters["onset_threshold"] == 0.3
    window.close()
    app.processEvents()


def test_gui_switches_logical_tracks_and_preserves_independent_edits():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._show_transcription_result(_mixed_result())

    # Default view is the fullest texture (rhythm); switch to lead explicitly.
    assert window.track_combo.count() == 2
    assert window.active_track_id == "logical-rhythm"
    assert window.event_table.rowCount() == 3
    assert "逻辑轨共享同一音频" in window.track_info_label.text()

    window.track_combo.setCurrentIndex(window.track_combo.findData("logical-lead"))
    assert window.edit_controller is not None
    changed_index = window.edit_controller.change_pitch(0, 71)
    window._edited(changed_index, "修改主音")
    window.track_combo.setCurrentIndex(
        window.track_combo.findData("logical-rhythm")
    )
    assert window.event_table.rowCount() == 3
    assert {window.event_table.item(row, 2).text() for row in range(3)} == {
        "48",
        "52",
        "55",
    }

    window.track_combo.setCurrentIndex(window.track_combo.findData("logical-lead"))
    assert window.event_table.item(0, 2).text() == "71"
    for controller in window.track_controllers.values():
        controller.mark_saved()
    window.close()
    app.processEvents()


def test_gui_saves_track_roles_and_independent_tablatures(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    project_path = tmp_path / "tracks.guitarbapu.json"
    window = MainWindow()
    window.selected_file = tmp_path / "missing.wav"
    window._show_transcription_result(_mixed_result())
    window.track_combo.setCurrentIndex(window.track_combo.findData("logical-lead"))
    assert window.edit_controller is not None
    changed_index = window.edit_controller.change_pitch(0, 71)
    window._edited(changed_index, "修改")
    window.track_role_combo.setCurrentIndex(
        window.track_role_combo.findData(TrackRole.SOLO.value)
    )
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(project_path), ""),
    )

    assert window._save_project()
    loaded = load_project(project_path)

    assert len(loaded.tracks) == 2
    assert loaded.active_track_id == "logical-lead"
    assert loaded.active_track is not None
    assert loaded.active_track.role is TrackRole.SOLO
    assert loaded.active_track.tablature.events[0].note is not None
    assert loaded.active_track.tablature.events[0].note.midi == 71
    assert loaded.analysis.notes[0].midi == 69
    assert not window._dirty_controllers()
    assert window.track_metadata_dirty is False
    window.close()
    app.processEvents()


def test_gui_exports_only_the_selected_logical_track(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    output = tmp_path / "rhythm-track.txt"
    window = MainWindow()
    window._show_transcription_result(_mixed_result())
    window.track_combo.setCurrentIndex(
        window.track_combo.findData("logical-rhythm")
    )
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(output), ""),
    )

    window._export_text()

    content = output.read_text(encoding="utf-8")
    assert "Mapped: 3" in content
    assert "Mapped: 4" not in content
    assert "当前轨道" in window.status_label.text()
    window.close()
    app.processEvents()
