"""Humanize dense neural-transcription output into playable note lists.

Neural models such as Basic Pitch happily emit 20+ notes per second; a real
guitarist tops out around 8–12.  ``simplify_notes`` applies four surgical
passes, in order:

1. merge duplicate onsets of the same pitch (< merge_gap_seconds apart)
2. drop micro-notes shorter than ``min_duration``
3. cap simultaneous notes at ``max_simultaneous`` (kept = highest confidence)
4. density cap: sliding window keeps at most ``max_notes_per_second``
"""

from __future__ import annotations

from collections.abc import Iterable

from .note import Note


def simplify_notes(
    notes: Iterable[Note],
    *,
    merge_gap_seconds: float = 0.03,
    min_duration: float = 0.07,
    max_simultaneous: int = 4,
    max_notes_per_second: float = 12.0,
) -> tuple[Note, ...]:
    """Return a humanized copy of ``notes`` (sorted by start, then midi)."""
    items = sorted(notes, key=lambda n: (n.start, n.midi))
    if not items:
        return ()

    # 1+2: merge rapid duplicate attacks of the same pitch and drop shorts
    merged: list[Note] = []
    for note in items:
        if (
            merged
            and note.midi == merged[-1].midi
            and note.start - merged[-1].start < merge_gap_seconds
        ):
            previous = merged[-1]
            end = max(
                previous.start + previous.duration, note.start + note.duration
            )
            merged[-1] = Note(
                midi=previous.midi,
                start=previous.start,
                duration=end - previous.start,
                velocity=max(previous.velocity, note.velocity),
                frequency_hz=previous.frequency_hz,
                confidence=max(
                    previous.confidence or 0.0, note.confidence or 0.0
                ) or None,
            )
            continue
        merged.append(note)
    merged = [
        note
        for note in merged
        if note.duration >= min_duration or note.duration <= 0
    ]

    # 3: cap simultaneous attacks (same start bucket at merge granularity)
    capped: list[Note] = []
    bucket: list[Note] = []

    def flush_bucket() -> None:
        if len(bucket) <= max_simultaneous:
            capped.extend(bucket)
        else:
            ranked = sorted(
                bucket,
                key=lambda n: (
                    -(n.confidence if n.confidence is not None else 0.5),
                    -(n.velocity),
                    n.midi,
                ),
            )
            capped.extend(ranked[:max_simultaneous])
        bucket.clear()

    bucket_start = None
    for note in merged:
        if bucket_start is None or note.start - bucket_start < merge_gap_seconds:
            bucket.append(note)
            if bucket_start is None:
                bucket_start = note.start
        else:
            flush_bucket()
            bucket.append(note)
            bucket_start = note.start
    flush_bucket()

    # 4: density cap over a sliding 1s window
    result: list[Note] = []
    window: list[float] = []
    for note in sorted(capped, key=lambda n: (n.start, n.midi)):
        window = [t for t in window if note.start - t < 1.0]
        if len(window) < max_notes_per_second:
            result.append(note)
            window.append(note.start)
    return tuple(result)


__all__ = ["simplify_notes"]
