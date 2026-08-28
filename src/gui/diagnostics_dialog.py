"""GUI presentation for read-only runtime diagnostics and support reports."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from src.utils.diagnostics import RuntimeDiagnostics, save_diagnostic_report
from src.utils.paths import user_log_dir


class DiagnosticsDialog(QDialog):
    """Show a copyable report without exposing environment variables."""

    def __init__(
        self,
        diagnostics: RuntimeDiagnostics,
        parent=None,
        *,
        log_dir: str | Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.diagnostics = diagnostics
        self.log_dir = Path(log_dir) if log_dir is not None else user_log_dir()
        self.setWindowTitle("GuitarBapu 系统诊断")
        self.resize(760, 520)

        layout = QVBoxLayout(self)
        self.report_text = QPlainTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setPlainText(diagnostics.render())
        layout.addWidget(self.report_text)

        actions = QHBoxLayout()
        self.copy_button = QPushButton("复制报告")
        self.copy_button.clicked.connect(self._copy_report)
        actions.addWidget(self.copy_button)
        self.save_button = QPushButton("保存报告")
        self.save_button.clicked.connect(self._save_report)
        actions.addWidget(self.save_button)
        self.open_logs_button = QPushButton("打开日志目录")
        self.open_logs_button.clicked.connect(self._open_logs)
        actions.addWidget(self.open_logs_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _copy_report(self) -> None:
        QApplication.clipboard().setText(self.diagnostics.render())

    def _save_report(self) -> None:
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "保存系统诊断",
            "guitarbapu-diagnostics.txt",
            "Text files (*.txt)",
        )
        if not file_name:
            return
        try:
            save_diagnostic_report(self.diagnostics, file_name)
        except OSError as error:
            QMessageBox.warning(self, "保存失败", str(error))

    def _open_logs(self) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            QMessageBox.warning(self, "无法打开日志目录", str(error))
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.log_dir)))


__all__ = ["DiagnosticsDialog"]
