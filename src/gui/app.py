"""Interactive desktop workflow for GuitarBapu transcription projects."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from pathlib import Path
import sys
from threading import Event

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.audio.analyzer import AudioAnalysis, AudioAnalyzer
from src.audio.demucs_separator import DemucsConfig, DemucsSeparator
from src.audio.loader import AudioData, load_audio
from src.audio.separation_cache import SeparationCache
from src.audio.separator import (
    SeparationCancelled,
    SeparationError,
    SeparationProgress,
    SeparationResult,
)
from src.audio.transcription_service import (
    TranscriptionResult,
    TranscriptionService,
)
from src.exporters import export_midi, export_musicxml, export_text_tab
from src.gui.audio_player import AudioPlayer, format_seconds
from src.gui.controller import TabEditController, TabEditError
from src.gui.event_editor import EventEditorDialog
from src.gui.waveform import WaveformWidget
from src.music.tab import TabEvent, Tablature
from src.music.tab_generator import TabGenerator
from src.music.tab_renderer import TextTabRenderer
from src.project import (
    ProjectFormatError,
    TranscriptionProject,
    load_project,
    save_project,
)


class MainWindow(QMainWindow):
    """Top-level import, analysis, playback, editing, and export workflow."""

    AUDIO_FILTER = "Audio files (*.mp3 *.wav *.flac)"
    EVENT_HEADERS = (
        "#",
        "音符",
        "MIDI",
        "弦",
        "品",
        "开始拍",
        "时值",
        "小节",
        "技巧",
        "可信度",
    )

    def __init__(self) -> None:
        super().__init__()
        self.selected_file: Path | None = None
        self.project_path: Path | None = None
        self.audio: AudioData | None = None
        self.analysis: AudioAnalysis | None = None
        self.tablature: Tablature | None = None
        self.edit_controller: TabEditController | None = None
        self.analysis_parameters: dict[str, object] = {}
        self.analysis_executor = ThreadPoolExecutor(max_workers=1)
        self.analysis_future: Future | None = None
        self.analysis_cancel_requested = False
        self.analysis_cancel_event: Event | None = None
        self.analysis_progress_state: SeparationProgress | None = None
        self.last_separation_result: SeparationResult | None = None
        self.playback_sources: dict[str, tuple[Path, AudioData]] = {}
        self.analysis_timer = QTimer(self)
        self.analysis_timer.setInterval(100)
        self.analysis_timer.timeout.connect(self._poll_analysis)
        self.audio_player = AudioPlayer(self)
        self._loop_selection: tuple[float, float] | None = None
        self._build_ui()
        self._connect_playback()

    def _build_ui(self) -> None:
        self.setWindowTitle("GuitarBapu AI Transcriber")
        self.setMinimumSize(900, 700)

        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        title = QLabel("GuitarBapu AI Transcriber")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        file_group = QGroupBox("音频与项目")
        file_layout = QVBoxLayout(file_group)
        file_buttons = QHBoxLayout()
        self.import_button = QPushButton("导入音频")
        self.import_button.clicked.connect(self._choose_audio_file)
        file_buttons.addWidget(self.import_button)
        self.open_project_button = QPushButton("打开项目")
        self.open_project_button.clicked.connect(self._open_project)
        file_buttons.addWidget(self.open_project_button)
        file_layout.addLayout(file_buttons)
        self.file_label = QLabel("尚未选择音频文件")
        self.file_label.setObjectName("fileLabel")
        self.file_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.file_label.setMinimumHeight(34)
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        layout.addWidget(file_group)

        playback_group = QGroupBox("音频播放与波形")
        playback_layout = QVBoxLayout(playback_group)
        playback_controls = QHBoxLayout()
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.audio_player.toggle)
        playback_controls.addWidget(self.play_button)
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.audio_player.stop)
        playback_controls.addWidget(self.stop_button)
        self.loop_checkbox = QCheckBox("循环选区")
        self.loop_checkbox.toggled.connect(self._loop_toggled)
        playback_controls.addWidget(self.loop_checkbox)
        self.playback_source_combo = QComboBox()
        self.playback_source_combo.setMinimumWidth(120)
        self.playback_source_combo.currentTextChanged.connect(
            self._playback_source_selected
        )
        playback_controls.addWidget(self.playback_source_combo)
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(
            lambda value: self.audio_player.seek(value / 1000.0)
        )
        playback_controls.addWidget(self.position_slider, stretch=1)
        self.time_label = QLabel("00:00.000 / 00:00.000")
        playback_controls.addWidget(self.time_label)
        playback_layout.addLayout(playback_controls)
        self.waveform = WaveformWidget()
        playback_layout.addWidget(self.waveform)
        layout.addWidget(playback_group)

        analysis_row = QHBoxLayout()
        self.separate_guitar_checkbox = QCheckBox("先分离吉他（Demucs）")
        demucs_available = DemucsSeparator.is_available()
        self.separate_guitar_checkbox.setEnabled(demucs_available)
        if demucs_available:
            device = DemucsSeparator.available_device().upper()
            self.separate_guitar_checkbox.setToolTip(
                f"模型：htdemucs_6s；设备：{device}；"
                "首次使用可能下载约 52 MB；分离结果会自动缓存"
            )
        else:
            self.separate_guitar_checkbox.setToolTip(
                "请在 Python 3.12 .venv 中安装 requirements-separation.txt"
            )
        analysis_row.addWidget(self.separate_guitar_checkbox)
        self.analyze_button = QPushButton("开始分析")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self._start_analysis)
        analysis_row.addWidget(self.analyze_button)
        self.cancel_analysis_button = QPushButton("取消分析")
        self.cancel_analysis_button.setEnabled(False)
        self.cancel_analysis_button.clicked.connect(self._cancel_analysis)
        analysis_row.addWidget(self.cancel_analysis_button)
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 0)
        self.analysis_progress.setVisible(False)
        analysis_row.addWidget(self.analysis_progress, stretch=1)
        layout.addLayout(analysis_row)

        status_group = QGroupBox("状态")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("等待导入音频")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        layout.addWidget(status_group)

        export_group = QGroupBox("项目保存与导出")
        export_layout = QHBoxLayout(export_group)
        self.save_project_button = QPushButton("保存项目")
        self.save_project_button.clicked.connect(self._save_project)
        export_layout.addWidget(self.save_project_button)
        self.export_text_button = QPushButton("导出 TAB")
        self.export_text_button.clicked.connect(self._export_text)
        export_layout.addWidget(self.export_text_button)
        self.export_midi_button = QPushButton("导出 MIDI")
        self.export_midi_button.clicked.connect(self._export_midi)
        export_layout.addWidget(self.export_midi_button)
        self.export_musicxml_button = QPushButton("导出 MusicXML")
        self.export_musicxml_button.clicked.connect(self._export_musicxml)
        export_layout.addWidget(self.export_musicxml_button)
        layout.addWidget(export_group)

        splitter = QSplitter(Qt.Orientation.Vertical)
        tab_group = QGroupBox("六线谱 TAB 与事件详情")
        tab_layout = QVBoxLayout(tab_group)
        self.tab_output = QPlainTextEdit()
        self.tab_output.setReadOnly(True)
        self.tab_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.tab_output.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        )
        self.tab_output.setPlaceholderText("分析或打开项目后，六线谱将在这里显示")
        tab_layout.addWidget(self.tab_output)
        splitter.addWidget(tab_group)

        editor_group = QGroupBox("TAB 事件编辑")
        editor_layout = QVBoxLayout(editor_group)
        self.event_table = QTableWidget(0, len(self.EVENT_HEADERS))
        self.event_table.setHorizontalHeaderLabels(self.EVENT_HEADERS)
        self.event_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.event_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.event_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.event_table.verticalHeader().setVisible(False)
        self.event_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.event_table.horizontalHeader().setStretchLastSection(True)
        self.event_table.itemSelectionChanged.connect(
            self._event_selection_changed
        )
        editor_layout.addWidget(self.event_table)
        editor_buttons = QHBoxLayout()
        self.insert_event_button = QPushButton("插入")
        self.insert_event_button.clicked.connect(self._insert_event)
        editor_buttons.addWidget(self.insert_event_button)
        self.edit_event_button = QPushButton("编辑")
        self.edit_event_button.clicked.connect(self._edit_event)
        editor_buttons.addWidget(self.edit_event_button)
        self.delete_event_button = QPushButton("删除")
        self.delete_event_button.clicked.connect(self._delete_event)
        editor_buttons.addWidget(self.delete_event_button)
        self.undo_button = QPushButton("撤销")
        self.undo_button.clicked.connect(self._undo_edit)
        editor_buttons.addWidget(self.undo_button)
        self.redo_button = QPushButton("重做")
        self.redo_button.clicked.connect(self._redo_edit)
        editor_buttons.addWidget(self.redo_button)
        editor_layout.addLayout(editor_buttons)
        splitter.addWidget(editor_group)
        splitter.setSizes((360, 240))
        layout.addWidget(splitter, stretch=1)

        self.setCentralWidget(central_widget)
        self._set_result_actions_enabled(False)
        self._set_playback_enabled(False)
        self._update_editor_actions()

    def _connect_playback(self) -> None:
        self.audio_player.position_changed.connect(self._playback_position_changed)
        self.audio_player.duration_changed.connect(self._playback_duration_changed)
        self.audio_player.playing_changed.connect(self._playback_state_changed)
        self.audio_player.error_occurred.connect(self._playback_error)
        self.waveform.position_requested.connect(self.audio_player.seek)
        self.waveform.selection_changed.connect(self._selection_changed)

    def _set_result_actions_enabled(self, enabled: bool) -> None:
        for button in (
            self.save_project_button,
            self.export_text_button,
            self.export_midi_button,
            self.export_musicxml_button,
        ):
            button.setEnabled(enabled)

    def _set_playback_enabled(self, enabled: bool) -> None:
        for widget in (
            self.play_button,
            self.stop_button,
            self.loop_checkbox,
            self.playback_source_combo,
            self.position_slider,
            self.waveform,
        ):
            widget.setEnabled(enabled)
        if not enabled:
            self.loop_checkbox.setChecked(False)

    def _clear_result(self, *, keep_project_path: bool = False) -> None:
        self.analysis = None
        self.tablature = None
        self.edit_controller = None
        self.last_separation_result = None
        if not keep_project_path:
            self.project_path = None
        self.analysis_parameters = {}
        self.tab_output.clear()
        self.event_table.setRowCount(0)
        self.waveform.set_event_times(())
        self._set_result_actions_enabled(False)
        self._update_editor_actions()

    def _clear_audio_view(self) -> None:
        self.audio_player.set_source(None)
        self.playback_sources = {}
        self.playback_source_combo.blockSignals(True)
        self.playback_source_combo.clear()
        self.playback_source_combo.blockSignals(False)
        self.waveform.clear()
        self.position_slider.setRange(0, 0)
        self.time_label.setText("00:00.000 / 00:00.000")
        self._set_playback_enabled(False)

    def _set_audio_source(self, path: Path, audio: AudioData) -> None:
        self.audio = audio
        self.selected_file = path
        self._set_playback_sources({"原音频": (path, audio)}, selected="原音频")

    def _set_playback_sources(
        self,
        sources: dict[str, tuple[Path, AudioData]],
        *,
        selected: str,
    ) -> None:
        self.playback_sources = dict(sources)
        self.playback_source_combo.blockSignals(True)
        self.playback_source_combo.clear()
        self.playback_source_combo.addItems(self.playback_sources)
        self.playback_source_combo.setCurrentText(selected)
        self.playback_source_combo.blockSignals(False)
        self._activate_playback_source(selected)

    def _activate_playback_source(self, name: str) -> None:
        source = self.playback_sources.get(name)
        if source is None:
            return
        path, audio = source
        self.audio_player.stop()
        self.waveform.set_audio(audio.waveform, audio.sample_rate)
        self._playback_duration_changed(audio.duration)
        if self.tablature is not None:
            self.waveform.set_event_times(
                event.start for event in self.tablature.events
            )
        try:
            self.audio_player.set_source(path)
        except FileNotFoundError as error:
            self._set_playback_enabled(False)
            self.status_label.setText(str(error))
        else:
            self._set_playback_enabled(True)

    def _playback_source_selected(self, name: str) -> None:
        if name:
            self._activate_playback_source(name)

    def _choose_audio_file(self) -> None:
        if not self._confirm_discard_changes():
            return
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            "",
            self.AUDIO_FILTER,
        )
        if not file_name:
            return
        path = Path(file_name).resolve(strict=False)
        try:
            audio = load_audio(path)
        except (OSError, ValueError, RuntimeError) as error:
            self.status_label.setText(f"音频读取失败：{error}")
            return

        self.audio_player.stop()
        self._clear_result()
        self._set_audio_source(path, audio)
        self.file_label.setText(
            f"文件名：{path.name}\n"
            f"时长：{audio.duration:.2f} 秒\n"
            f"采样率：{audio.sample_rate} Hz"
        )
        self.status_label.setText("音频已导入，可以播放或开始分析")
        self.analyze_button.setEnabled(True)

    def _open_project(self) -> None:
        if not self._confirm_discard_changes():
            return
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "打开 GuitarBapu 项目",
            "",
            "GuitarBapu project (*.guitarbapu.json *.json)",
        )
        if not file_name:
            return
        try:
            project = load_project(file_name)
        except ProjectFormatError as error:
            self.status_label.setText(f"项目打开失败：{error}")
            return

        self.audio_player.stop()
        self.project_path = Path(file_name).resolve(strict=False)
        self.selected_file = project.audio_path
        self.analysis = project.analysis
        self.tablature = project.tablature
        self.edit_controller = TabEditController(project.tablature)
        self.analysis_parameters = dict(project.analysis_parameters)

        audio_text = "未记录原音频"
        self.audio = None
        self._clear_audio_view()
        if project.audio_path is not None:
            if project.audio_path.is_file():
                try:
                    audio = load_audio(project.audio_path)
                except (OSError, ValueError, RuntimeError) as error:
                    audio_text = f"{project.audio_path}（读取失败：{error}）"
                else:
                    self._set_audio_source(project.audio_path, audio)
                    audio_text = f"{project.audio_path}（可用）"
            else:
                audio_text = f"{project.audio_path}（文件已移动或缺失）"

        self._restore_cached_separation_playback()

        self.file_label.setText(
            f"项目：{self.project_path.name}\n原音频：{audio_text}"
        )
        self.analyze_button.setEnabled(self.audio is not None)
        self._apply_tablature(project.tablature)
        self.status_label.setText(
            f"项目已打开：{len(project.analysis.notes)} 个分析音符，"
            f"{len(project.tablature.events)} 个 TAB 事件"
        )

    def _start_analysis(self) -> None:
        if self.audio is None:
            self.status_label.setText("请先导入可用的音频文件")
            return
        if self.analysis_future is not None:
            return
        if not self._confirm_discard_changes():
            return

        self.audio_player.pause()
        self.analyze_button.setEnabled(False)
        self.import_button.setEnabled(False)
        self.open_project_button.setEnabled(False)
        self.cancel_analysis_button.setEnabled(True)
        self.analysis_progress.setVisible(True)
        self.analysis_progress.setRange(0, 0)
        self._clear_result(keep_project_path=True)
        use_separation = self.separate_guitar_checkbox.isChecked()
        if self.selected_file is not None:
            self._set_playback_sources(
                {"原音频": (self.selected_file, self.audio)},
                selected="原音频",
            )
        self.status_label.setText(
            "正在准备吉他分离；首次使用可能下载约 52 MB 模型…"
            if use_separation
            else "正在检测音高、清理音符并分析节奏，请稍候…"
        )
        analyzer = AudioAnalyzer()
        self.analysis_parameters = {
            "fmin_hz": analyzer.fmin_hz,
            "fmax_hz": analyzer.fmax_hz,
            "frame_length": analyzer.frame_length,
            "hop_length": analyzer.hop_length,
            "energy_threshold": analyzer.energy_threshold,
            "beat_subdivision": analyzer.rhythm_analyzer.subdivision,
            "use_separation": use_separation,
        }
        separator = None
        if use_separation:
            separator = DemucsSeparator(DemucsConfig(model_name="htdemucs_6s"))
            self.analysis_parameters["separation_model"] = "htdemucs_6s"
        self.analysis_cancel_requested = False
        self.analysis_cancel_event = Event()
        self.analysis_progress_state = None
        service = TranscriptionService(
            analyzer=analyzer,
            tab_generator=TabGenerator(),
            separator=separator,
        )

        def progress_callback(progress: SeparationProgress) -> None:
            self.analysis_progress_state = progress

        self.analysis_future = self.analysis_executor.submit(
            service.transcribe,
            self.selected_file,
            audio=None if use_separation else self.audio,
            use_separation=use_separation,
            progress_callback=progress_callback,
            cancel_event=self.analysis_cancel_event,
        )
        self.analysis_timer.start()

    def _cancel_analysis(self) -> None:
        if self.analysis_future is None:
            return
        self.analysis_cancel_requested = True
        if self.analysis_cancel_event is not None:
            self.analysis_cancel_event.set()
        self.analysis_future.cancel()
        self.cancel_analysis_button.setEnabled(False)
        self.status_label.setText(
            "已请求取消；正在等待当前 Demucs/librosa 计算安全结束…"
        )

    def _poll_analysis(self) -> None:
        future = self.analysis_future
        if future is None:
            return
        progress = self.analysis_progress_state
        if progress is not None and not self.analysis_cancel_requested:
            if progress.fraction is None:
                self.analysis_progress.setRange(0, 0)
            else:
                self.analysis_progress.setRange(0, 100)
                self.analysis_progress.setValue(round(progress.fraction * 100))
            self.status_label.setText(progress.message)
        if not future.done():
            return
        self.analysis_timer.stop()
        try:
            if self.analysis_cancel_requested or future.cancelled():
                self.status_label.setText("分析已取消")
                return
            result = future.result()
        except (CancelledError, SeparationCancelled):
            self.status_label.setText("分析已取消")
        except SeparationError as error:
            self.status_label.setText(f"吉他分离失败：{error}；可取消勾选后分析原音频")
        except Exception as error:  # GUI boundary: report instead of crashing
            self.status_label.setText(f"分析失败：{error}")
        else:
            self._show_transcription_result(result)
        finally:
            self._analysis_finished()

    def _analysis_finished(self) -> None:
        self.analyze_button.setEnabled(self.audio is not None)
        self.import_button.setEnabled(True)
        self.open_project_button.setEnabled(True)
        self.cancel_analysis_button.setEnabled(False)
        self.analysis_progress.setVisible(False)
        self.analysis_future = None
        self.analysis_cancel_requested = False
        self.analysis_cancel_event = None
        self.analysis_progress_state = None

    def _show_transcription_result(self, result: TranscriptionResult) -> None:
        self.last_separation_result = result.separation
        self._show_analysis(result.analysis, tablature=result.tablature)
        if result.separation is None:
            return
        separation = result.separation
        cache_text = "复用缓存" if separation.from_cache else "新生成"
        self.analysis_parameters["separation"] = {
            "model_name": separation.model_name,
            "device": separation.device,
            "cache_key": separation.cache_key,
            "stem": "guitar",
        }
        try:
            guitar_audio = load_audio(result.analyzed_audio_path)
        except (OSError, ValueError, RuntimeError) as error:
            self.status_label.setText(
                self.status_label.text() + f"；吉他 stem 播放加载失败：{error}"
            )
        else:
            sources = dict(self.playback_sources)
            sources["吉他分离轨"] = (result.analyzed_audio_path, guitar_audio)
            self._set_playback_sources(sources, selected="吉他分离轨")
        if self.selected_file is not None and self.audio is not None:
            self.file_label.setText(
                f"文件名：{self.selected_file.name}\n"
                f"原音频：{self.audio.duration:.2f} 秒 / {self.audio.sample_rate} Hz\n"
                f"分析源：guitar stem（{separation.model_name}，"
                f"{separation.device.upper()}，{cache_text}）"
            )
        self.status_label.setText(
            self.status_label.text() + f"；吉他分离轨已就绪（{cache_text}）"
        )

    def _restore_cached_separation_playback(self) -> None:
        separation = self.analysis_parameters.get("separation")
        if not isinstance(separation, dict):
            return
        cache_key = separation.get("cache_key")
        if not isinstance(cache_key, str):
            return
        try:
            cached = SeparationCache().load(cache_key)
            if cached is None:
                return
            guitar_path = cached.stem("guitar").path
            guitar_audio = load_audio(guitar_path)
        except (OSError, ValueError, RuntimeError, SeparationError):
            return
        sources = dict(self.playback_sources)
        sources["吉他分离轨"] = (guitar_path, guitar_audio)
        self._set_playback_sources(
            sources,
            selected="原音频" if "原音频" in sources else "吉他分离轨",
        )

    def _show_analysis(
        self,
        analysis: AudioAnalysis,
        *,
        tablature: Tablature | None = None,
    ) -> None:
        if not analysis.notes:
            self.status_label.setText("分析完成：未检测到可用的单音音符")
            self.tab_output.setPlainText(
                "未检测到音符。请尝试单音、较清晰的吉他录音。"
            )
            return
        if tablature is None:
            try:
                tablature = TabGenerator().generate(analysis)
            except (RuntimeError, ValueError) as error:
                self.status_label.setText(f"分析失败：TAB 生成失败：{error}")
                return
        self.analysis = analysis
        self.tablature = tablature
        self.edit_controller = TabEditController(tablature)
        self._apply_tablature(tablature)
        self.status_label.setText(
            f"分析完成：{len(analysis.raw_notes)} 个原始事件清理为 "
            f"{len(analysis.notes)} 个音符；TAB 映射 {len(tablature.events)} 个，"
            f"未映射 {len(tablature.unmapped_notes)} 个"
        )

    def _apply_tablature(
        self, tablature: Tablature, *, selected_index: int | None = None
    ) -> None:
        self.tablature = tablature
        if self.analysis is not None:
            self._display_results(self.analysis, tablature)
        self._refresh_event_table(selected_index=selected_index)
        self.waveform.set_event_times(event.start for event in tablature.events)
        self._set_result_actions_enabled(True)
        self._update_editor_actions()

    def _display_results(
        self, analysis: AudioAnalysis, tablature: Tablature
    ) -> None:
        rendered_tab = TextTabRenderer().render(tablature)
        rhythm = analysis.rhythm
        tempo = rhythm.timing.tempo_bpm if rhythm is not None else None
        tempo_text = f"{tempo:.1f} BPM" if tempo is not None else "未检测到稳定 BPM"
        detail_lines = [
            "TAB 事件详情",
            f"原始音符：{len(analysis.raw_notes)}  清理后：{len(analysis.notes)}",
            f"节拍：{tempo_text}",
            "",
        ]
        for event in tablature.events:
            midi = self._event_midi(event, tablature)
            name = event.note.name if event.note is not None else f"MIDI {midi}"
            confidence = (
                f" 可信度={event.confidence:.0%}"
                if event.confidence is not None
                else ""
            )
            technique = f" 技巧={event.technique}" if event.technique else ""
            detail_lines.append(
                f"{name:<4} MIDI={midi:<3} 弦={event.string} 品={event.fret:<2} "
                f"拍={event.start_beat or 0.0:.2f} "
                f"时值={event.duration_beats or 0.0:.2f}拍"
                f"{technique}{confidence}"
            )
        self.tab_output.setPlainText(
            rendered_tab + "\n\n" + "\n".join(detail_lines)
        )

    @staticmethod
    def _event_midi(event: TabEvent, tablature: Tablature) -> int:
        if event.note is not None:
            return event.note.midi
        return tablature.guitar.midi_at(event.string, event.fret)

    def _refresh_event_table(self, *, selected_index: int | None = None) -> None:
        tablature = self.tablature
        self.event_table.blockSignals(True)
        self.event_table.setRowCount(0 if tablature is None else len(tablature.events))
        if tablature is not None:
            for row, event in enumerate(tablature.events):
                midi = self._event_midi(event, tablature)
                values = (
                    str(row + 1),
                    event.note.name if event.note is not None else str(midi),
                    str(midi),
                    str(event.string),
                    str(event.fret),
                    f"{event.start_beat or 0.0:.3f}",
                    f"{event.duration_beats or 0.0:.3f}",
                    str(event.measure),
                    event.technique or "",
                    (
                        f"{event.confidence:.0%}"
                        if event.confidence is not None
                        else ""
                    ),
                )
                for column, value in enumerate(values):
                    self.event_table.setItem(row, column, QTableWidgetItem(value))
        self.event_table.blockSignals(False)
        if (
            selected_index is not None
            and tablature is not None
            and 0 <= selected_index < len(tablature.events)
        ):
            self.event_table.selectRow(selected_index)
        else:
            self.event_table.clearSelection()
        self._update_editor_actions()

    def _selected_event_index(self) -> int | None:
        rows = self.event_table.selectionModel().selectedRows()
        return rows[0].row() if rows else None

    def _event_selection_changed(self) -> None:
        self._update_editor_actions()
        index = self._selected_event_index()
        if index is None or self.tablature is None:
            return
        event = self.tablature.events[index]
        self.audio_player.seek(event.start)
        end = event.start + max(event.duration, 0.01)
        if self.waveform.duration > 0:
            self.waveform.set_selection(event.start, end)

    def _update_editor_actions(self) -> None:
        has_controller = self.edit_controller is not None
        has_selection = self._selected_event_index() is not None
        self.insert_event_button.setEnabled(has_controller)
        self.edit_event_button.setEnabled(has_controller and has_selection)
        self.delete_event_button.setEnabled(has_controller and has_selection)
        self.undo_button.setEnabled(
            has_controller and bool(self.edit_controller and self.edit_controller.can_undo)
        )
        self.redo_button.setEnabled(
            has_controller and bool(self.edit_controller and self.edit_controller.can_redo)
        )

    def _insert_event(self) -> None:
        controller = self.edit_controller
        if controller is None:
            return
        selected = self._selected_event_index()
        if selected is not None:
            event = controller.tablature.events[selected]
            default_midi = self._event_midi(event, controller.tablature)
            default_start = (event.start_beat or 0.0) + (
                event.duration_beats or 0.0
            )
        elif controller.tablature.events:
            event = controller.tablature.events[-1]
            default_midi = self._event_midi(event, controller.tablature)
            default_start = (event.start_beat or 0.0) + (
                event.duration_beats or 0.0
            )
        else:
            default_midi = 64
            default_start = 0.0
        dialog = EventEditorDialog(
            controller.tablature,
            default_midi=default_midi,
            default_start_beat=default_start,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        try:
            index = controller.insert_event(
                midi=values.midi,
                position=values.position,
                start_beat=values.start_beat,
                duration_beats=values.duration_beats,
                technique=values.technique,
            )
        except TabEditError as error:
            self.status_label.setText(f"插入失败：{error}")
            return
        self._edited(index, "已插入 TAB 事件")

    def _edit_event(self) -> None:
        controller = self.edit_controller
        index = self._selected_event_index()
        if controller is None or index is None:
            return
        event = controller.tablature.events[index]
        dialog = EventEditorDialog(
            controller.tablature,
            event=event,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        try:
            new_index = controller.update_event(
                index,
                midi=values.midi,
                string=values.position.string,
                fret=values.position.fret,
                start_beat=values.start_beat,
                duration_beats=values.duration_beats,
                technique=values.technique,
            )
        except TabEditError as error:
            self.status_label.setText(f"编辑失败：{error}")
            return
        self._edited(new_index, "TAB 事件已修改")

    def _delete_event(self) -> None:
        controller = self.edit_controller
        index = self._selected_event_index()
        if controller is None or index is None:
            return
        controller.delete_event(index)
        next_index = min(index, len(controller.tablature.events) - 1)
        self._edited(
            next_index if next_index >= 0 else None,
            "TAB 事件已删除，可撤销",
        )

    def _undo_edit(self) -> None:
        if self.edit_controller is None:
            return
        self.edit_controller.undo()
        self._edited(None, "已撤销")

    def _redo_edit(self) -> None:
        if self.edit_controller is None:
            return
        self.edit_controller.redo()
        self._edited(None, "已重做")

    def _edited(self, selected_index: int | None, message: str) -> None:
        if self.edit_controller is None:
            return
        self._apply_tablature(
            self.edit_controller.tablature,
            selected_index=selected_index,
        )
        dirty_text = "（未保存）" if self.edit_controller.dirty else ""
        self.status_label.setText(message + dirty_text)

    def _playback_position_changed(self, seconds: float) -> None:
        if (
            self.loop_checkbox.isChecked()
            and self._loop_selection is not None
            and self.audio_player.is_playing
        ):
            start, end = self._loop_selection
            if end - start >= 0.02 and seconds >= end - 0.01:
                self.audio_player.seek(start)
                return
        self.position_slider.setValue(round(seconds * 1000))
        self.waveform.set_cursor(seconds)
        self.time_label.setText(
            f"{format_seconds(seconds)} / {format_seconds(self.waveform.duration)}"
        )

    def _playback_duration_changed(self, seconds: float) -> None:
        duration = max(0.0, float(seconds), self.waveform.duration)
        self.position_slider.setRange(0, round(duration * 1000))
        self.time_label.setText(
            f"{format_seconds(self.audio_player.position)} / {format_seconds(duration)}"
        )

    def _playback_state_changed(self, playing: bool) -> None:
        self.play_button.setText("暂停" if playing else "播放")

    def _playback_error(self, error: str) -> None:
        self.status_label.setText(f"音频播放失败：{error}")

    def _selection_changed(self, start: float, end: float) -> None:
        self._loop_selection = (start, end) if end - start >= 0.02 else None

    def _loop_toggled(self, enabled: bool) -> None:
        if not enabled:
            return
        if self._loop_selection is None:
            self.loop_checkbox.setChecked(False)
            self.status_label.setText("请先在波形上拖动，或选择一个 TAB 事件")
            return
        start, _ = self._loop_selection
        self.audio_player.seek(start)

    def _default_export_name(self, suffix: str) -> str:
        if self.project_path is not None:
            name = self.project_path.name
            if name.endswith(".guitarbapu.json"):
                name = name[: -len(".guitarbapu.json")]
            else:
                name = self.project_path.stem
        elif self.selected_file is not None:
            name = self.selected_file.stem
        else:
            name = "transcription"
        return name + suffix

    @staticmethod
    def _with_suffix(path: str, suffixes: tuple[str, ...], default: str) -> Path:
        target = Path(path).expanduser()
        if not target.name.lower().endswith(suffixes):
            target = target.with_name(target.name + default)
        return target

    def _save_project(self) -> bool:
        if self.analysis is None or self.tablature is None:
            self.status_label.setText("没有可以保存的分析结果")
            return False
        default_name = (
            str(self.project_path)
            if self.project_path is not None
            else self._default_export_name(".guitarbapu.json")
        )
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "保存 GuitarBapu 项目",
            default_name,
            "GuitarBapu project (*.guitarbapu.json *.json)",
        )
        if not file_name:
            return False
        target = self._with_suffix(file_name, (".json",), ".guitarbapu.json")
        project = TranscriptionProject(
            audio_path=self.selected_file,
            analysis=self.analysis,
            tablature=self.tablature,
            analysis_parameters=self.analysis_parameters,
        )
        try:
            self.project_path = save_project(project, target)
        except (OSError, TypeError, ValueError) as error:
            self.status_label.setText(f"项目保存失败：{error}")
            return False
        if self.edit_controller is not None:
            self.edit_controller.mark_saved()
        self._update_editor_actions()
        self.status_label.setText(f"项目已保存：{self.project_path}")
        return True

    def _export_result(
        self,
        *,
        title: str,
        file_filter: str,
        suffixes: tuple[str, ...],
        default_suffix: str,
        exporter,
    ) -> None:
        if self.tablature is None:
            self.status_label.setText("没有可以导出的 TAB")
            return
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            title,
            self._default_export_name(default_suffix),
            file_filter,
        )
        if not file_name:
            return
        target = self._with_suffix(file_name, suffixes, default_suffix)
        try:
            exported = exporter(self.tablature, target)
        except Exception as error:  # GUI boundary: third-party exporters vary
            self.status_label.setText(f"导出失败：{error}")
            return
        self.status_label.setText(f"导出成功：{exported}")

    def _export_text(self) -> None:
        self._export_result(
            title="导出文本 TAB",
            file_filter="Text TAB (*.txt)",
            suffixes=(".txt",),
            default_suffix=".txt",
            exporter=export_text_tab,
        )

    def _export_midi(self) -> None:
        self._export_result(
            title="导出 MIDI",
            file_filter="MIDI file (*.mid *.midi)",
            suffixes=(".mid", ".midi"),
            default_suffix=".mid",
            exporter=export_midi,
        )

    def _export_musicxml(self) -> None:
        self._export_result(
            title="导出 MusicXML",
            file_filter="MusicXML (*.musicxml *.xml)",
            suffixes=(".musicxml", ".xml"),
            default_suffix=".musicxml",
            exporter=export_musicxml,
        )

    def _confirm_discard_changes(self) -> bool:
        if self.edit_controller is None or not self.edit_controller.dirty:
            return True
        choice = QMessageBox.warning(
            self,
            "未保存的修改",
            "当前 TAB 有未保存的修改。是否先保存项目？",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if choice == QMessageBox.StandardButton.Save:
            return self._save_project()
        return choice == QMessageBox.StandardButton.Discard

    def closeEvent(self, event) -> None:
        if not self._confirm_discard_changes():
            event.ignore()
            return
        self.analysis_timer.stop()
        self.analysis_cancel_requested = True
        if self.analysis_cancel_event is not None:
            self.analysis_cancel_event.set()
        if self.analysis_future is not None:
            self.analysis_future.cancel()
        self.audio_player.stop()
        self.analysis_executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
