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
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.audio.analyzer import AudioAnalysis, AudioAnalyzer
from src.audio.loader import AudioData, load_audio
from src.music.tab import Tablature
from src.music.tab_generator import TabGenerator
from src.music.tab_renderer import TextTabRenderer


class MainWindow(QMainWindow):
    """Top-level window for importing and preparing an audio transcription."""

    AUDIO_FILTER = "Audio files (*.mp3 *.wav *.flac)"

    def __init__(self) -> None:
        super().__init__()
        self.selected_file: Path | None = None
        self.audio: AudioData | None = None
        self.tablature: Tablature | None = None
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
        self.import_button = QPushButton("导入音频")
        self.import_button.clicked.connect(self._choose_audio_file)
        file_layout.addWidget(self.import_button)

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
            self.file_label.setText("尚未选择音频文件")
            self.status_label.setText(f"音频读取失败：{error}")
            self.analyze_button.setEnabled(False)
            return

        self.audio = audio
        self.tablature = None
        self.file_label.setText(
            f"文件名：{self.selected_file.name}\n"
            f"时长：{audio.duration:.2f} 秒\n"
            f"采样率：{audio.sample_rate} Hz"
        )
        self.status_label.setText("音频已导入，可以开始分析")
        self.analyze_button.setEnabled(True)

    def _start_analysis(self) -> None:
        """Start Phase 4 analysis in a worker thread."""

        if self.audio is None:
            self.status_label.setText("请先导入音频文件")
            return
        if self.analysis_future is not None:
            return

        self.analyze_button.setEnabled(False)
        self.import_button.setEnabled(False)
        self.tab_output.clear()
        self.status_label.setText("正在检测音高、清理音符并分析节奏，请稍候…")
        self.analysis_future = self.analysis_executor.submit(
            AudioAnalyzer().analyze,
            self.audio,
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
            rendered_tab = TextTabRenderer().render(tablature)
        except (RuntimeError, ValueError) as error:
            self._show_analysis_error(f"TAB 生成失败：{error}")
            return
        self.tablature = tablature

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
        self.status_label.setText(
            f"分析完成：{len(analysis.raw_notes)} 个原始事件清理为 "
            f"{len(notes)} 个音符；TAB 映射 {len(tablature.events)} 个，"
            f"未映射 {len(tablature.unmapped_notes)} 个"
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
