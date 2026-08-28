"""Widget-independent editing state for structured guitar tablature."""

from __future__ import annotations

from dataclasses import replace
import math

from src.music.fretboard import FretPosition, Fretboard
from src.music.note import Note
from src.music.tab import TabEvent, Tablature


class TabEditError(ValueError):
    """Raised when an edit would create an invalid or unplayable event."""


class TabEditController:
    """Edit immutable tablature snapshots with undo, redo, and dirty state."""

    def __init__(self, tablature: Tablature) -> None:
        self._history = [tablature]
        self._history_index = 0
        self._saved_snapshot = tablature

    @property
    def tablature(self) -> Tablature:
        return self._history[self._history_index]

    @property
    def dirty(self) -> bool:
        return self.tablature != self._saved_snapshot

    @property
    def can_undo(self) -> bool:
        return self._history_index > 0

    @property
    def can_redo(self) -> bool:
        return self._history_index + 1 < len(self._history)

    def mark_saved(self) -> None:
        self._saved_snapshot = self.tablature

    def reset(self, tablature: Tablature, *, saved: bool = True) -> None:
        self._history = [tablature]
        self._history_index = 0
        if saved:
            self._saved_snapshot = tablature

    def undo(self) -> Tablature:
        if not self.can_undo:
            return self.tablature
        self._history_index -= 1
        return self.tablature

    def redo(self) -> Tablature:
        if not self.can_redo:
            return self.tablature
        self._history_index += 1
        return self.tablature

    def _commit(self, tablature: Tablature) -> Tablature:
        if tablature == self.tablature:
            return tablature
        del self._history[self._history_index + 1 :]
        self._history.append(tablature)
        self._history_index += 1
        return tablature

    @staticmethod
    def _find_event_index(events: tuple[TabEvent, ...], target: TabEvent) -> int:
        """Find a normalized copy of ``target`` after measure fields change."""

        for index, event in enumerate(events):
            if (
                event.string == target.string
                and event.fret == target.fret
                and event.start_beat == target.start_beat
                and event.duration_beats == target.duration_beats
                and event.note == target.note
                and event.technique == target.technique
            ):
                return index
        raise TabEditError("编辑后的 TAB 事件无法定位")

    def _event(self, index: int) -> TabEvent:
        try:
            return self.tablature.events[index]
        except IndexError as exc:
            raise TabEditError("TAB 事件索引超出范围") from exc

    def positions_for_midi(self, midi: int) -> tuple[FretPosition, ...]:
        try:
            return Fretboard(self.tablature.guitar).find_positions(midi)
        except ValueError as exc:
            raise TabEditError(str(exc)) from exc

    def positions_for_event(self, index: int) -> tuple[FretPosition, ...]:
        event = self._event(index)
        return self.positions_for_midi(self._event_midi(event))

    def _event_midi(self, event: TabEvent) -> int:
        if event.note is not None:
            return event.note.midi
        return self.tablature.guitar.midi_at(event.string, event.fret)

    def _event_beats(self, event: TabEvent) -> tuple[float, float]:
        start = (
            event.start_beat
            if event.start_beat is not None
            else event.start / self._seconds_per_beat()
        )
        duration = (
            event.duration_beats
            if event.duration_beats is not None
            else max(
                1.0 / self.tablature.subdivision,
                event.duration / self._seconds_per_beat(),
            )
        )
        return start, duration

    def _position_conflicts(
        self,
        *,
        string: int,
        start_beat: float,
        duration_beats: float,
        exclude_index: int | None = None,
    ) -> bool:
        end_beat = start_beat + duration_beats
        for index, event in enumerate(self.tablature.events):
            if index == exclude_index or event.string != string:
                continue
            other_start, other_duration = self._event_beats(event)
            other_end = other_start + other_duration
            if max(start_beat, other_start) < min(end_beat, other_end) - 1e-9:
                return True
        return False

    def _seconds_per_beat(self) -> float:
        return 60.0 / (self.tablature.tempo_bpm or 120.0)

    def _note_for_event(
        self,
        event: TabEvent,
        *,
        midi: int,
        start_beat: float,
        duration_beats: float,
        pitch_changed: bool,
    ) -> Note:
        source = event.note or Note(self._event_midi(event))
        seconds_per_beat = self._seconds_per_beat()
        return Note(
            midi=midi,
            start=start_beat * seconds_per_beat,
            duration=duration_beats * seconds_per_beat,
            velocity=source.velocity,
            frequency_hz=None if pitch_changed else source.frequency_hz,
            confidence=None if pitch_changed else source.confidence,
        )

    def _normalized(self, events: tuple[TabEvent, ...]) -> Tablature:
        beats_per_measure = self.tablature.beats_per_measure
        normalized: list[TabEvent] = []
        maximum_beat = 0.0
        for event in sorted(
            events,
            key=lambda item: (
                item.start_beat if item.start_beat is not None else item.start,
                item.string,
                item.fret,
            ),
        ):
            start_beat = event.start_beat
            duration_beats = event.duration_beats
            if start_beat is None:
                start_beat = event.start / self._seconds_per_beat()
            if duration_beats is None:
                duration_beats = max(
                    1.0 / self.tablature.subdivision,
                    event.duration / self._seconds_per_beat(),
                )
            measure = int(start_beat // beats_per_measure) + 1
            end_beat = start_beat + duration_beats
            end_measure = int((end_beat - 1e-9) // beats_per_measure) + 1
            maximum_beat = max(maximum_beat, end_beat)
            normalized.append(
                replace(
                    event,
                    start_beat=start_beat,
                    duration_beats=duration_beats,
                    measure=measure,
                    tie_to_next=end_measure > measure,
                )
            )
        for rest in self.tablature.rests:
            if rest.start_beat is not None and rest.duration_beats is not None:
                maximum_beat = max(
                    maximum_beat, rest.start_beat + rest.duration_beats
                )
        measure_count = max(1, math.ceil(maximum_beat / beats_per_measure))
        return replace(
            self.tablature,
            events=tuple(normalized),
            measure_count=measure_count,
        )

    def update_event(
        self,
        index: int,
        *,
        midi: int,
        string: int,
        fret: int,
        start_beat: float,
        duration_beats: float,
        technique: str | None = None,
    ) -> int:
        """Atomically update an event while keeping pitch and position equal."""

        event = self._event(index)
        if start_beat < 0:
            raise TabEditError("开始拍不能为负数")
        if duration_beats <= 0:
            raise TabEditError("音符时值必须大于零")
        positions = self.positions_for_midi(midi)
        if not positions:
            raise TabEditError("该 MIDI 音高超出当前吉他的可演奏范围")
        position = FretPosition(string=string, fret=fret)
        if position not in positions:
            raise TabEditError("选择的琴弦/品位与 MIDI 音高不一致")
        if self._position_conflicts(
            string=string,
            start_beat=start_beat,
            duration_beats=duration_beats,
            exclude_index=index,
        ):
            raise TabEditError("该琴弦在所选时间范围内已有其他音符")

        pitch_changed = midi != self._event_midi(event)
        note = self._note_for_event(
            event,
            midi=midi,
            start_beat=start_beat,
            duration_beats=duration_beats,
            pitch_changed=pitch_changed,
        )
        updated = replace(
            event,
            string=string,
            fret=fret,
            start=note.start,
            duration=note.duration,
            note=note,
            start_beat=start_beat,
            duration_beats=duration_beats,
            technique=technique.strip() if technique and technique.strip() else None,
            confidence=None if pitch_changed else event.confidence,
        )
        events = list(self.tablature.events)
        events[index] = updated
        result = self._commit(self._normalized(tuple(events)))
        return self._find_event_index(result.events, updated)

    def change_pitch(self, index: int, midi: int) -> int:
        """Change pitch and choose the playable position nearest the old one."""

        event = self._event(index)
        previous = FretPosition(event.string, event.fret)
        start_beat, duration_beats = self._event_beats(event)
        fretboard = Fretboard(self.tablature.guitar)
        try:
            positions = sorted(
                fretboard.find_positions(midi),
                key=lambda item: fretboard.score_position(item, previous),
            )
        except ValueError as exc:
            raise TabEditError(str(exc)) from exc
        if not positions:
            raise TabEditError("该 MIDI 音高超出当前吉他的可演奏范围")
        for position in positions:
            if not self._position_conflicts(
                string=position.string,
                start_beat=start_beat,
                duration_beats=duration_beats,
                exclude_index=index,
            ):
                return self.update_event(
                    index,
                    midi=midi,
                    string=position.string,
                    fret=position.fret,
                    start_beat=start_beat,
                    duration_beats=duration_beats,
                    technique=event.technique,
                )
        raise TabEditError("该时间范围内没有不冲突的可用琴弦")

    def change_position(self, index: int, position: FretPosition) -> int:
        """Move an event to another legal position for the same pitch."""

        event = self._event(index)
        return self.update_event(
            index,
            midi=self._event_midi(event),
            string=position.string,
            fret=position.fret,
            start_beat=event.start_beat or 0.0,
            duration_beats=event.duration_beats
            or 1.0 / self.tablature.subdivision,
            technique=event.technique,
        )

    def insert_event(
        self,
        *,
        midi: int,
        start_beat: float,
        duration_beats: float,
        position: FretPosition | None = None,
        technique: str | None = None,
    ) -> int:
        if start_beat < 0:
            raise TabEditError("开始拍不能为负数")
        if duration_beats <= 0:
            raise TabEditError("音符时值必须大于零")
        previous_event = None
        for event in self.tablature.events:
            if (event.start_beat or 0.0) <= start_beat:
                previous_event = event
            else:
                break
        previous = (
            FretPosition(previous_event.string, previous_event.fret)
            if previous_event is not None
            else None
        )
        fretboard = Fretboard(self.tablature.guitar)
        try:
            available_positions = fretboard.find_positions(midi)
        except ValueError as exc:
            raise TabEditError(str(exc)) from exc
        if not available_positions:
            raise TabEditError("该 MIDI 音高超出当前吉他的可演奏范围")
        if position is not None and position not in available_positions:
            raise TabEditError("选择的琴弦/品位与 MIDI 音高不一致")
        if position is None:
            candidates = sorted(
                available_positions,
                key=lambda item: fretboard.score_position(item, previous),
            )
            position = next(
                (
                    item
                    for item in candidates
                    if not self._position_conflicts(
                        string=item.string,
                        start_beat=start_beat,
                        duration_beats=duration_beats,
                    )
                ),
                None,
            )
            if position is None:
                raise TabEditError("该时间范围内没有不冲突的可用琴弦")
        elif self._position_conflicts(
            string=position.string,
            start_beat=start_beat,
            duration_beats=duration_beats,
        ):
            raise TabEditError("该琴弦在所选时间范围内已有其他音符")
        seconds_per_beat = self._seconds_per_beat()
        inserted = TabEvent(
            string=position.string,
            fret=position.fret,
            start=start_beat * seconds_per_beat,
            duration=duration_beats * seconds_per_beat,
            note=Note(
                midi,
                start=start_beat * seconds_per_beat,
                duration=duration_beats * seconds_per_beat,
            ),
            start_beat=start_beat,
            duration_beats=duration_beats,
            technique=technique.strip() if technique and technique.strip() else None,
        )
        result = self._commit(
            self._normalized(self.tablature.events + (inserted,))
        )
        return self._find_event_index(result.events, inserted)

    def delete_event(self, index: int) -> Tablature:
        self._event(index)
        events = list(self.tablature.events)
        del events[index]
        return self._commit(self._normalized(tuple(events)))


__all__ = ["TabEditController", "TabEditError"]
