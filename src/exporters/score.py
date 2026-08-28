"""Convert structured TAB into a music21 score shared by file exporters."""

from music21 import (
    articulations,
    expressions,
    instrument,
    meter,
    metadata,
    note,
    stream,
    tempo,
)

from ..music.tab import TabEvent, Tablature


def _event_midi(event: TabEvent, tablature: Tablature) -> int:
    if event.note is not None:
        return event.note.midi
    return tablature.guitar.midi_at(event.string, event.fret)


def tablature_to_score(tablature: Tablature) -> stream.Score:
    """Build a standard-notation score while retaining string/fret metadata."""

    score = stream.Score(id="GuitarBapuScore")
    score.metadata = metadata.Metadata()
    score.metadata.title = "GuitarBapu Transcription"

    part = stream.Part(id="Guitar")
    part.partName = "Guitar"
    part.insert(0, instrument.AcousticGuitar())
    numerator, denominator = tablature.time_signature
    part.insert(0, meter.TimeSignature(f"{numerator}/{denominator}"))
    if tablature.tempo_bpm is not None:
        part.insert(0, tempo.MetronomeMark(number=tablature.tempo_bpm))

    for event in sorted(
        tablature.events,
        key=lambda item: (item.start_beat or 0.0, item.string),
    ):
        start_beat = event.start_beat
        duration_beats = event.duration_beats
        if start_beat is None:
            start_beat = event.start * (tablature.tempo_bpm or 120.0) / 60.0
        if duration_beats is None:
            duration_beats = max(
                1.0 / tablature.subdivision,
                event.duration * (tablature.tempo_bpm or 120.0) / 60.0,
            )
        exported_note = note.Note(_event_midi(event, tablature))
        exported_note.duration.quarterLength = float(duration_beats)
        if event.note is not None:
            exported_note.volume.velocity = event.note.velocity
        exported_note.articulations.extend(
            (
                articulations.StringIndication(event.string),
                articulations.FretIndication(event.fret),
            )
        )
        if event.technique:
            # A text expression is intentionally used for this first
            # recognition baseline. It survives MusicXML export without
            # claiming exact start/end spanners that the detector does not
            # yet model reliably.
            part.insert(
                float(start_beat),
                expressions.TextExpression(event.technique),
            )
        part.insert(float(start_beat), exported_note)

    score.insert(0, part)
    return score


__all__ = ["tablature_to_score"]
