"""Tests for TAB playback synthesis."""

import numpy as np

from src.audio.synth import synthesize_tablature
from src.music.guitar import Guitar
from src.music.note import Note
from src.music.tab import TabEvent, Tablature


def _tablature(events):
    return Tablature(guitar=Guitar.standard(), events=tuple(events), measure_count=1)


def test_synthesize_single_note_produces_sound():
    tab = _tablature([TabEvent(1, 0, start=0.0, duration=0.5, note=Note(64))])
    audio = synthesize_tablature(tab)

    assert audio.dtype == np.float32
    assert audio.size > 0.5 * 22050
    assert np.max(np.abs(audio)) > 0.1


def test_synthesize_empty_tablature_is_silent():
    tab = _tablature(())
    audio = synthesize_tablature(tab)

    assert audio.size == 0


def test_synthesis_respects_start_offsets():
    early = _tablature([TabEvent(1, 0, start=0.0, duration=0.2, note=Note(64))])
    late = _tablature([TabEvent(1, 0, start=1.0, duration=0.2, note=Note(64))])

    early_audio = synthesize_tablature(early)
    late_audio = synthesize_tablature(late)

    assert np.argmax(np.abs(early_audio)) < 0.1 * 22050
    assert np.argmax(np.abs(late_audio)) > 0.95 * 22050
