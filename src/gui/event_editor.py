"""Dialog for atomically editing one playable TAB event."""

from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.music.fretboard import FretPosition, Fretboard
from src.music.note import Note
from src.music.tab import TabEvent, Tablature


@dataclass(frozen=True)
class EventEditValues:
    midi: int
    position: FretPosition
    start_beat: float
    duration_beats: float
    technique: str | None


class EventEditorDialog(QDialog):
    """Edit pitch, legal position, beat timing, and technique together."""

    TECHNIQUES = (
        "",
        "slide",
        "hammer-on",
        "pull-off",
        "bend",
        "vibrato",
    )

    def __init__(
        self,
        tablature: Tablature,
        *,
        event: TabEvent | None = None,
        default_midi: int = 64,
        default_start_beat: float = 0.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.tablature = tablature
        self.fretboard = Fretboard(tablature.guitar)
        self.event = event
        self.setWindowTitle("编辑 TAB 事件" if event is not None else "插入 TAB 事件")

        if event is not None:
            default_midi = (
                event.note.midi
                if event.note is not None
                else tablature.guitar.midi_at(event.string, event.fret)
            )
            default_start_beat = event.start_beat or 0.0
            default_duration = event.duration_beats or 1.0 / tablature.subdivision
            default_technique = event.technique or ""
            self._preferred_position = FretPosition(event.string, event.fret)
        else:
            default_duration = 1.0
            default_technique = ""
            self._preferred_position = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.midi_spin = QSpinBox()
        self.midi_spin.setRange(0, 127)
        self.midi_spin.setValue(default_midi)
        form.addRow("MIDI 音高", self.midi_spin)
        self.note_label = QLabel()
        form.addRow("音符", self.note_label)
        self.position_combo = QComboBox()
        form.addRow("琴弦 / 品位", self.position_combo)
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0.0, 1_000_000.0)
        self.start_spin.setDecimals(3)
        self.start_spin.setSingleStep(0.25)
        self.start_spin.setValue(default_start_beat)
        form.addRow("开始拍", self.start_spin)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.001, 1_000_000.0)
        self.duration_spin.setDecimals(3)
        self.duration_spin.setSingleStep(0.25)
        self.duration_spin.setValue(default_duration)
        form.addRow("时值（拍）", self.duration_spin)
        self.technique_combo = QComboBox()
        self.technique_combo.setEditable(True)
        self.technique_combo.addItems(self.TECHNIQUES)
        self.technique_combo.setCurrentText(default_technique)
        form.addRow("演奏技巧", self.technique_combo)
        layout.addLayout(form)

        self.validation_label = QLabel()
        self.validation_label.setWordWrap(True)
        layout.addWidget(self.validation_label)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.midi_spin.valueChanged.connect(self._refresh_positions)
        self._refresh_positions(default_midi)

    def _refresh_positions(self, midi: int) -> None:
        self.note_label.setText(Note(midi).name)
        positions = self.fretboard.find_positions(midi)
        self.position_combo.clear()
        selected_index = 0
        for index, position in enumerate(positions):
            self.position_combo.addItem(
                f"{position.string} 弦 / {position.fret} 品",
                (position.string, position.fret),
            )
            if position == self._preferred_position:
                selected_index = index
        if positions:
            self.position_combo.setCurrentIndex(selected_index)
            self.validation_label.setText("")
        else:
            self.validation_label.setText("该音高超出当前吉他的可演奏范围")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            bool(positions)
        )

    def values(self) -> EventEditValues:
        data = self.position_combo.currentData()
        if data is None:
            raise ValueError("没有可用的琴弦/品位")
        technique = self.technique_combo.currentText().strip() or None
        return EventEditValues(
            midi=self.midi_spin.value(),
            position=FretPosition(int(data[0]), int(data[1])),
            start_beat=self.start_spin.value(),
            duration_beats=self.duration_spin.value(),
            technique=technique,
        )


__all__ = ["EventEditValues", "EventEditorDialog"]
