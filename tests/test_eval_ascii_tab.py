"""Tests for the local ASCII TAB comparison helper."""

from src.eval.ascii_tab import flatten_midi_events, parse_ascii_tab


def test_ascii_tab_parser_reads_high_string_events():
    events = parse_ascii_tab(
        "e|--2-0-----3-0-------|\n"
        "B|--0-0---3h0-2--3-0--|\n"
        "G|--------------------|\n"
        "D|--------------------|\n"
        "A|--------------------|\n"
        "E|--------------------|\n"
    )

    assert events[0].midis == (59, 66)
    assert events[1].midis == (59, 64)
    flattened = flatten_midi_events(events)
    assert flattened[:2] == (66, 64)
    assert 62 in flattened
    assert 67 in flattened


def test_ascii_tab_parser_accepts_circled_digits_and_ignores_symbols():
    events = parse_ascii_tab(
        "e|--③---0-♪--0-0------|\n"
        "B|----2-3-0-2-1-0-2-0-|\n"
    )

    assert 67 in events[0].midis
    assert flatten_midi_events(events)[0] == 67
