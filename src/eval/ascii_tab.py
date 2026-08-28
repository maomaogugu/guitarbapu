"""Parse simple ASCII guitar TAB into MIDI event groups for evaluation.

This is intentionally small and conservative: it is a comparison aid for
user-provided reference snippets, not a MusicXML/TAB interchange format.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_STRING_MIDI = {
    "e": 64,
    "B": 59,
    "G": 55,
    "D": 50,
    "A": 45,
    "E": 40,
}
_CIRCLED_DIGITS = {
    "⓪": 0,
    "①": 1,
    "②": 2,
    "③": 3,
    "④": 4,
    "⑤": 5,
    "⑥": 6,
    "⑦": 7,
    "⑧": 8,
    "⑨": 9,
}
_LINE_RE = re.compile(r"^\s*([eBGDAE])\s*\|(.*)\|?\s*$")


@dataclass(frozen=True)
class TabColumnEvent:
    """One vertical TAB position with the pitches found on any string."""

    column: int
    midis: tuple[int, ...]


def _digit_value(character: str) -> int | None:
    if character in _CIRCLED_DIGITS:
        return _CIRCLED_DIGITS[character]
    if character.isascii() and character.isdigit():
        return int(character)
    return None


def _parse_line(label: str, body: str) -> tuple[tuple[int, int], ...]:
    if label not in _STRING_MIDI:
        raise ValueError(f"unknown string label: {label}")
    base = _STRING_MIDI[label]
    tokens: list[tuple[int, int]] = []
    index = 0
    while index < len(body):
        value = _digit_value(body[index])
        if value is None:
            index += 1
            continue
        end = index + 1
        while end < len(body) and body[end].isdigit():
            end += 1
        fret = int("".join(str(_digit_value(ch)) for ch in body[index:end]))
        if 0 <= fret <= 36:
            tokens.append((index, base + fret))
        index = end
    return tuple(tokens)


def _parse_block(block: dict[str, str]) -> tuple[TabColumnEvent, ...]:
    tokens: list[tuple[int, int]] = []
    for label, body in block.items():
        for column, midi in _parse_line(label, body):
            tokens.append((column, midi))
    return tuple(
        TabColumnEvent(
            column=column,
            midis=tuple(sorted(midi for token_column, midi in tokens if token_column == column)),
        )
        for column in sorted({column for column, _ in tokens})
    )


def parse_ascii_tab(text: str) -> tuple[TabColumnEvent, ...]:
    """Return column-ordered MIDI groups, preserving repeated TAB blocks."""

    events: list[TabColumnEvent] = []
    current: dict[str, str] = {}
    for raw_line in text.splitlines():
        match = _LINE_RE.match(raw_line)
        if not match:
            continue
        label, body = match.groups()
        if label == "e" and current:
            events.extend(_parse_block(current))
            current = {}
        current[label] = body
    if current:
        events.extend(_parse_block(current))
    return tuple(events)


def flatten_midi_events(events: Iterable[TabColumnEvent]) -> tuple[int, ...]:
    """Flatten column events into one approximate melodic comparison stream."""

    flattened: list[int] = []
    for event in events:
        # Prefer the highest voice when a reference column contains a chord.
        flattened.append(max(event.midis))
    return tuple(flattened)


__all__ = ["TabColumnEvent", "flatten_midi_events", "parse_ascii_tab"]
