"""Tests for the hand-checked answer TAB parser."""

from src.eval.answer_tab import expected_strings_by_measure, parse_answer_tab


def test_answer_tab_parser_handles_inline_segments_and_unicode_dashes():
    text = (
        "1小节 e|——3-0—| B|—0——0——|\n"
        "D|–2———| A|————| E|3————|\n"
    )
    events = parse_answer_tab(text)
    grouped = expected_strings_by_measure(events)

    assert grouped[(1, 1)] == (3, 0)
    assert grouped[(1, 2)] == (0, 0)
    assert grouped[(1, 4)] == (2,)
    assert grouped[(1, 6)] == (3,)


def test_answer_tab_parser_marks_hammer_and_pulloff_on_second_note():
    events = parse_answer_tab("8小节 e|0-0h1-0| B|--2--|\n")
    by_fret = {(event.measure, event.order): event for event in events}

    hammered = next(event for event in events if (event.string, event.fret) == (1, 1))
    assert hammered.technique == "h"
