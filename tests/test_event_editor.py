"""Offscreen tests for legal pitch/position choices in the event dialog."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QDialogButtonBox

from src.gui.event_editor import EventEditorDialog
from src.music.note import Note
from src.music.tab import TabEvent, Tablature


def test_editor_keeps_existing_legal_position():
    app = QApplication.instance() or QApplication([])
    event = TabEvent(
        string=2,
        fret=5,
        note=Note(64),
        start_beat=1.0,
        duration_beats=0.5,
    )
    dialog = EventEditorDialog(Tablature(events=(event,)), event=event)

    values = dialog.values()

    assert values.midi == 64
    assert (values.position.string, values.position.fret) == (2, 5)
    assert values.start_beat == 1.0
    assert values.duration_beats == 0.5
    dialog.close()
    app.processEvents()


def test_editor_disables_accept_for_unplayable_pitch():
    app = QApplication.instance() or QApplication([])
    dialog = EventEditorDialog(Tablature(), default_midi=64)

    dialog.midi_spin.setValue(20)

    assert dialog.position_combo.count() == 0
    assert not dialog.buttons.button(
        QDialogButtonBox.StandardButton.Ok
    ).isEnabled()
    assert "超出" in dialog.validation_label.text()
    dialog.close()
    app.processEvents()
