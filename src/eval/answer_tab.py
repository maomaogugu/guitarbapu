"""Parse hand-written answer TAB text (per-bar, possibly Unicode dashes).

The accepted format is the one used by our evaluation assets: each measure is
introduced by ``N小节`` and followed by string segments like ``e|--3-0--|``.
Segments may share a line. ``h``/``p`` between two frets mark hammer-on and
pull-off on the second note.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


_STRING_TO_NUMBER = {"e": 1, "B": 2, "G": 3, "D": 4, "A": 5, "E": 6}

_DASH_TRANSLATE = str.maketrans({
    "—": "-",
    "–": "-",
    "‑": "-",
    "−": "-",
})
_SEGMENT_RE = re.compile(r"([eBGDAE])\|([^|\n]*)\|")


@dataclass(frozen=True)
class AnswerEvent:
    measure: int
    order: int
    string: int
    fret: int
    technique: str | None = None
    span: int = 0


def parse_answer_tab(text: str) -> tuple[AnswerEvent, ...]:
    """Return ordered answer events grouped by measure and string column."""

    lines = text.splitlines()
    events: list[AnswerEvent] = []
    measure = 0
    pending: list[tuple[str, str]] = []
    for raw_line in lines:
        header = re.search(r"(\d+)\s*小节", raw_line)
        if header is not None:
            if pending:
                events.extend(_emit_measure(measure, pending))
                pending = []
            measure = int(header.group(1))
            raw_line = raw_line[header.end() :]
        normalized = raw_line.translate(_DASH_TRANSLATE).replace(" ", "")
        for match in _SEGMENT_RE.finditer(normalized):
            pending.append((match.group(1), match.group(2)))
    if pending:
        events.extend(_emit_measure(measure, pending))
    return tuple(events)


def _emit_measure(
    measure: int, segments: list[tuple[str, str]]
) -> list[AnswerEvent]:
    events: list[AnswerEvent] = []
    for label, body in segments:
        string_number = _STRING_TO_NUMBER[label]
        span = max(1, len(body) - 1)
        pending_technique: str | None = None
        for index, character in enumerate(body):
            if character.isdigit():
                events.append(
                    AnswerEvent(
                        measure=measure,
                        order=index,
                        string=string_number,
                        fret=int(character),
                        technique=pending_technique,
                        span=span,
                    )
                )
                pending_technique = None
            elif character in ("h", "p"):
                pending_technique = character
    return events


def expected_strings_by_measure(
    events: tuple[AnswerEvent, ...],
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Map (measure, string) to the ordered fret sequence."""

    grouped: dict[tuple[int, int], list[AnswerEvent]] = {}
    for event in events:
        grouped.setdefault((event.measure, event.string), []).append(event)
    return {
        key: tuple(item.fret for item in sorted(value, key=lambda e: e.order))
        for key, value in grouped.items()
    }


__all__ = ["AnswerEvent", "expected_strings_by_measure", "parse_answer_tab"]
