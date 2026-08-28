"""Snapshot-like checks for deterministic text TAB rendering."""

from src.music.guitar import Guitar
from src.music.note import Note
from src.music.tab import TabEvent, Tablature, UnmappedNote
from src.music.tab_renderer import TextTabRenderer


def test_renderer_aligns_single_and_double_digit_frets():
    tab = Tablature(
        guitar=Guitar.standard(),
        events=(
            TabEvent(1, 3, note=Note(67), start_beat=0, duration_beats=1),
            TabEvent(1, 12, note=Note(76), start_beat=1, duration_beats=1),
        ),
        tempo_bpm=120,
    )

    rendered = TextTabRenderer().render(tab)

    high_e = next(line for line in rendered.splitlines() if line.startswith("   e "))
    assert "3--" in high_e
    assert "12-" in high_e
    assert "Tempo: 120.0 BPM" in rendered


def test_renderer_wraps_after_four_measures():
    tab = Tablature(measure_count=5)

    rendered = TextTabRenderer(measures_per_line=4).render(tab)

    assert "M1" in rendered
    assert "M4" in rendered
    assert "M5" in rendered
    assert rendered.count("beat ") == 2


def test_renderer_shows_x_and_warning_for_unmapped_note():
    note = Note(39, start=0.5, duration=0.5)
    tab = Tablature(
        unmapped_notes=(
            UnmappedNote(note, "音符超出范围", start_beat=1.0, measure=1),
        )
    )

    rendered = TextTabRenderer().render(tab)

    assert "x--" in rendered
    assert "警告：" in rendered
    assert "音符超出范围" in rendered


def test_renderer_lists_detected_technique_with_confidence():
    tab = Tablature(
        events=(
            TabEvent(
                1,
                3,
                start=0.5,
                note=Note(67),
                start_beat=1.0,
                duration_beats=1.0,
                technique="slide",
                technique_confidence=0.82,
            ),
        )
    )

    rendered = TextTabRenderer().render(tab)

    assert "技巧候选（请试听确认）" in rendered
    assert "slide" in rendered
    assert "82%" in rendered
