"""Main window for the GuitarBapu desktop application.

This module owns only presentation and user interaction. It calls the public
audio services but does not contain decoding or pitch-analysis logic.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import sys
from pathlib import Path

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.audio.analyzer import AudioAnalysis, AudioAnalyzer
from src.audio.loader import AudioData, load_audio
from src.exporters import export_midi, export_musicxml, export_text_tab
from src.music.tab import Tablature
from src.music.tab_generator import TabGenerator
from src.music.tab_renderer import TextTabRenderer
from src.project import (
    ProjectFormatError,
    TranscriptionProject,
    load_project,
    save_project,
)


class MainWindow(QMainWindow):
    """Top-level window for importing and preparing an audio transcription."""

    AUDIO_FILTER = "Audio files (*.mp3 *.wav *.flac)"

    def __init__(self) -> None:
        super().__init__()
        self.selected_file: Path | None = None
        self.project_path: Path | None = None
        self.audio: AudioData | None = None
        self.analysis: AudioAnalysis | None = None
        self.tablature: Tablature | None = None
        self.analysis_parameters: dict[str, object] = {}
        self.analysis_executor = ThreadPoolExecutor(max_workers=1)
        self.analysis_future: Future | None = None
        self.analysis_timer = QTimer(self)
        self.analysis_timer.setInterval(100)
        self.analysis_timer.timeout.connect(self._poll_analysis)
        self._build_ui()

    def _build_ui(self) -> None:
        """Create the fixed skeleton UI and connect its controls."""

        self.setWindowTitle("GuitarBapu AI Transcriber")
        self.setMinimumSize(640, 480)

        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("GuitarBapu AI Transcriber")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        file_group = QGroupBox("音频文件")
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
        self.file_label.setMinimumHeight(36)
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        layout.addWidget(file_group)

        self.analyze_button = QPushButton("开始分析")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self._start_analysis)
        layout.addWidget(self.analyze_button)

        status_group = QGroupBox("分析状态")
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
        self._set_result_actions_enabled(False)

        tab_group = QGroupBox("六线谱 TAB 与音符详情")
        tab_layout = QVBoxLayout(tab_group)
        self.tab_output = QPlainTextEdit()
        self.tab_output.setReadOnly(True)
        self.tab_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.tab_output.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        )
        self.tab_output.setPlaceholderText("分析完成后，六线谱将在这里显示")
        tab_layout.addWidget(self.tab_output)
        layout.addWidget(tab_group, stretch=1)

        self.setCentralWidget(central_widget)

    def _set_result_actions_enabled(self, enabled: bool) -> None:
        """Enable save/export actions only when a result is available."""

        for button in (
            self.save_project_button,
            self.export_text_button,
            self.export_midi_button,
            self.export_musicxml_button,
        ):
            button.setEnabled(enabled)

    def _clear_result(self) -> None:
        self.analysis = None
        self.tablature = None
        self.project_path = None
        self.analysis_parameters = {}
        self.tab_output.clear()
        self._set_result_actions_enabled(False)

    def _choose_audio_file(self) -> None:
        """Open a picker limited to the audio formats supported by the UI."""

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            "",
            self.AUDIO_FILTER,
        )
        if not file_name:
            return

        self.selected_file = Path(file_name)
        try:
            audio = load_audio(self.selected_file)
        except (OSError, ValueError, RuntimeError) as error:
            self.selected_file = None
            self.audio = None
            self._clear_result()
            self.file_label.setText("尚未选择音频文件")
            self.status_label.setText(f"音频读取失败：{error}")
            self.analyze_button.setEnabled(False)
            return

        self.audio = audio
        self._clear_result()
        self.file_label.setText(
            f"文件名：{self.selected_file.name}\n"
            f"时长：{audio.duration:.2f} 秒\n"
            f"采样率：{audio.sample_rate} Hz"
        )
        self.status_label.setText("音频已导入，可以开始分析")
        self.analyze_button.setEnabled(True)

    def _open_project(self) -> None:
        """Open a saved project without requiring the original audio file."""

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

        self.project_path = Path(file_name).resolve(strict=False)
        self.selected_file = project.audio_path
        self.audio = None
        self.analysis = project.analysis
        self.tablature = project.tablature
        self.analysis_parameters = dict(project.analysis_parameters)
        audio_text = "未记录原音频"
        if project.audio_path is not None:
            availability = "可用" if project.audio_path.exists() else "文件已移动或缺失"
            audio_text = f"{project.audio_path}（{availability}）"
        self.file_label.setText(
            f"项目：{self.project_path.name}\n原音频：{audio_text}"
        )
        self.analyze_button.setEnabled(False)
        self._display_results(project.analysis, project.tablature)
        self._set_result_actions_enabled(True)
        self.status_label.setText(
            f"项目已打开：{len(project.analysis.notes)} 个音符，"
            f"{len(project.tablature.events)} 个 TAB 事件"
        )

    def _start_analysis(self) -> None:
        """Start the transcription pipeline in a worker thread."""

        if self.audio is None:
            self.status_label.setText("请先导入音频文件")
            return
        if self.analysis_future is not None:
            return

        self.analyze_button.setEnabled(False)
        self.import_button.setEnabled(False)
        self.open_project_button.setEnabled(False)
        self._clear_result()
        self.status_label.setText("正在检测音高、清理音符并分析节奏，请稍候…")
        analyzer = AudioAnalyzer()
        self.analysis_parameters = {
            "fmin_hz": analyzer.fmin_hz,
            "fmax_hz": analyzer.fmax_hz,
            "frame_length": analyzer.frame_length,
            "hop_length": analyzer.hop_length,
            "energy_threshold": analyzer.energy_threshold,
            "beat_subdivision": analyzer.rhythm_analyzer.subdivision,
        }
        self.analysis_future = self.analysis_executor.submit(
            analyzer.analyze, self.audio
        )
        self.analysis_timer.start()

    def _poll_analysis(self) -> None:
        future = self.analysis_future
        if future is None or not future.done():
            return
        self.analysis_timer.stop()
        try:
            analysis = future.result()
        except Exception as error:  # GUI boundary: report instead of crashing
            self._show_analysis_error(str(error))
        else:
            self._show_analysis(analysis)
        finally:
            self._analysis_finished()

    def _show_analysis_error(self, error: str) -> None:
        self.status_label.setText(f"分析失败：{error}")

    def _analysis_finished(self) -> None:
        self.analyze_button.setEnabled(self.audio is not None)
        self.import_button.setEnabled(True)
        self.open_project_button.setEnabled(True)
        self.analysis_future = None

    def _show_analysis(self, analysis: AudioAnalysis) -> None:
        """Render a completed AudioAnalysis without performing audio work."""

        notes = analysis.notes

        if not notes:
            self.status_label.setText("分析完成：未检测到可用的单音音符")
            self.tab_output.setPlainText("未检测到音符。请尝试单音、较清晰的吉他录音。")
            return

        try:
            tablature = TabGenerator().generate(analysis)
        except (RuntimeError, ValueError) as error:
            self._show_analysis_error(f"TAB 生成失败：{error}")
            return
        self.analysis = analysis
        self.tablature = tablature
        self._display_results(analysis, tablature)
        self._set_result_actions_enabled(True)
        self.status_label.setText(
            f"分析完成：{len(analysis.raw_notes)} 个原始事件清理为 "
            f"{len(notes)} 个音符；TAB 映射 {len(tablature.events)} 个，"
            f"未映射 {len(tablature.unmapped_notes)} 个"
        )

    def _display_results(
        self, analysis: AudioAnalysis, tablature: Tablature
    ) -> None:
        """Display already computed data without rerunning any algorithm."""

        rendered_tab = TextTabRenderer().render(tablature)
        notes = analysis.notes

        rhythm = analysis.rhythm
        tempo = rhythm.timing.tempo_bpm if rhythm is not None else None
        tempo_text = f"{tempo:.1f} BPM" if tempo is not None else "未检测到稳定 BPM"
        detail_lines = [
            "音符详情",
            f"原始音符：{len(analysis.raw_notes)}  清理后：{len(notes)}",
            f"节拍：{tempo_text}",
            "",
        ]
        quantized = rhythm.quantized_notes if rhythm is not None else ()
        if quantized:
            for item in quantized:
                note = item.note
                beat_text = ""
                if item.start_beat is not None and item.duration_beats is not None:
                    beat_text = (
                        f" 拍={item.start_beat:.2f} 时值={item.duration_beats:.2f}拍"
                    )
                confidence = (
                    f" 可信度={item.source.confidence:.0%}"
                    if item.source.confidence is not None
                    else ""
                )
                detail_lines.append(
                    f"{note.name:<4} MIDI={note.midi:<3} "
                    f"开始={note.start:.2f}s 时长={note.duration:.2f}s"
                    f"{beat_text}{confidence}"
                )
        else:
            detail_lines.extend(
                f"{note.name:<4} MIDI={note.midi:<3} "
                f"开始={note.start:.2f}s 时长={note.duration:.2f}s"
                for note in notes
            )
        self.tab_output.setPlainText(
            rendered_tab + "\n\n" + "\n".join(detail_lines)
        )

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

    def _save_project(self) -> None:
        if self.analysis is None or self.tablature is None:
            self.status_label.setText("没有可以保存的分析结果")
            return
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
            return
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
            return
        self.status_label.setText(f"项目已保存：{self.project_path}")

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

    def closeEvent(self, event) -> None:
        """Stop polling and release the background executor on window close."""

        self.analysis_timer.stop()
        self.analysis_executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)


def main() -> int:
    """Create the Qt application and run the event loop."""

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
