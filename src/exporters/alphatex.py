"""Convert our Tablature model into alphaTex markup for alphaTab rendering.

alphaTex note syntax: ``fret.string.duration`` (string 1 = high E), chords as
``(3.1 2.2).8``, rests ``r.4``, bars separated by ``|``.  Metadata commands
(``\\title``, ``\\tempo``, ``\\ts``, ``\\capo``) go first.
"""

from __future__ import annotations

from src.music.tab import Tablature

_EPS = 1e-6

# (beats, alphaTex token suffix); dotted variants included
_DURATIONS: tuple[tuple[float, str], ...] = (
    (6.0, "1 d"),
    (4.0, "1"),
    (3.0, "2 d"),
    (2.0, "2"),
    (1.5, "4 d"),
    (1.0, "4"),
    (0.75, "8 d"),
    (0.5, "8"),
    (0.375, "16 d"),
    (0.25, "16"),
    (0.125, "32"),
)


def _duration_token(beats: float) -> str:
    """Snap a beat length to the closest alphaTex duration."""
    if beats <= 0:
        return "16"
    best = min(_DURATIONS, key=lambda item: abs(item[0] - beats))
    return best[1]


def _beats_to_seconds(beats: float, tempo_bpm: float | None) -> float:
    if tempo_bpm is None or tempo_bpm <= 0:
        return beats
    return beats * 60.0 / tempo_bpm


def to_alphatex(tablature: Tablature, *, title: str = "Transcription") -> str:
    """Render ``tablature`` as alphaTex, one bar per line."""

    tempo = tablature.tempo_bpm
    numerator, denominator = tablature.time_signature
    lines = [
        f'\\title "{title}"',
        f"\\ts {numerator} {denominator}",
    ]
    if tempo is not None:
        lines.append(f"\\tempo {tempo:.0f}")
    if tablature.guitar.capo:
        lines.append(f"\\capo {tablature.guitar.capo}")
    lines.append("\\staff{tabs}")

    bar_length = 4.0 * numerator / denominator
    measure_count = max(tablature.measure_count, 1)
    events = sorted(tablature.events, key=lambda e: (e.start_beat or 0.0))

    for measure in range(1, measure_count + 1):
        start = (measure - 1) * bar_length
        end = measure * bar_length
        bar_events = [
            e
            for e in events
            if e.start_beat is not None and start - _EPS <= e.start_beat < end - _EPS
        ]
        bar_events.sort(key=lambda e: e.start_beat)
        tokens: list[str] = []
        cursor = start
        index = 0
        while index < len(bar_events):
            event = bar_events[index]
            # fill gap with a rest if the next event starts later
            gap = event.start_beat - cursor
            if gap > 0.01:
                tokens.append(f"r.{_duration_token(gap)}")
                cursor += gap
                continue
            # collect simultaneous events into a chord
            chord = [event]
            lookahead = index + 1
            while (
                lookahead < len(bar_events)
                and abs(bar_events[lookahead].start_beat - event.start_beat) < 0.01
            ):
                chord.append(bar_events[lookahead])
                lookahead += 1
            duration = max(
                (c.duration_beats or 0.0 for c in chord),
                default=0.25,
            ) or 0.25
            inner = " ".join(f"{c.fret}.{c.string}" for c in chord)
            if len(chord) > 1:
                tokens.append(f"({inner}).{_duration_token(duration)}")
            else:
                tokens.append(f"{inner}.{_duration_token(duration)}")
            cursor = max(cursor, event.start_beat) + duration
            index = lookahead
        # pad trailing silence
        remaining = end - cursor
        if remaining > 0.01:
            tokens.append(f"r.{_duration_token(remaining)}")
        lines.append(" ".join(tokens) + " |" if tokens else "|")
    return "\n".join(lines)


__all__ = ["to_alphatex"]
