"""Main window for the GuitarBapu desktop application.

This module owns only presentation and user interaction. It calls the public
audio services but does not contain decoding or pitch-analysis logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import Qt
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

from src.audio.analyzer import AudioAnalyzer
from src.audio.loader import AudioData, load_audio


class MainWindow(QMainWindow):
    """Top-level window for importing and preparing an audio transcription."""

    AUDIO_FILTER = "Audio files (*.mp3 *.wav *.flac)"

    def __init__(self) -> None:
        super().__init__()
        self.selected_file: Path | None = None
        self.audio: AudioData | None = None
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

        tab_group = QGroupBox("检测到的音符（基础版）")
        tab_layout = QVBoxLayout(tab_group)
        self.tab_output = QPlainTextEdit()
        self.tab_output.setReadOnly(True)
        self.tab_output.setPlaceholderText("分析完成后，识别到的音符将在这里显示")
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
        self.file_label.setText(
            f"文件名：{self.selected_file.name}\n"
            f"时长：{audio.duration:.2f} 秒\n"
            f"采样率：{audio.sample_rate} Hz"
        )
        self.status_label.setText("音频已导入，可以开始分析")
        self.analyze_button.setEnabled(True)

    def _start_analysis(self) -> None:
        """Run the Phase 2 monophonic analyzer and show detected note events."""

        if self.audio is None:
            self.status_label.setText("请先导入音频文件")
            return

        self.analyze_button.setEnabled(False)
        self.tab_output.clear()
        self.status_label.setText("正在进行基础单音音高检测，请稍候…")
        QApplication.processEvents()
        try:
            analysis = AudioAnalyzer().analyze(self.audio)
        except (RuntimeError, ValueError) as error:
            self.status_label.setText(f"音高检测失败：{error}")
            return
        finally:
            self.analyze_button.setEnabled(True)

        if not analysis.notes:
            self.status_label.setText("分析完成：未检测到可用的单音音符")
            self.tab_output.setPlainText("未检测到音符。请尝试单音、较清晰的吉他录音。")
            return

        lines = [
            f"{note.name:<4} MIDI={note.midi:<3} 开始={note.start:.2f}s 时长={note.duration:.2f}s"
            for note in analysis.notes
        ]
        self.tab_output.setPlainText("\n".join(lines))
        self.status_label.setText(
            f"分析完成：检测到 {len(analysis.notes)} 个基础音符事件"
        )


def main() -> int:
    """Create the Qt application and run the event loop."""

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
