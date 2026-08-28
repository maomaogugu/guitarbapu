"""Generate structured guitar tablature from cleaned audio analysis."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import TYPE_CHECKING

from .fingering import FingeringOptimizer
from .guitar import Guitar
from .tab import TabEvent, TabRest, Tablature, UnmappedNote
from .timing import QuantizedNote, TimingInfo, quantize_notes

if TYPE_CHECKING:
    from ..audio.analyzer import AudioAnalysis


class TabGenerator:
    """Convert Phase 4 notes and timing into playable TAB events."""

    def __init__(
        self,
        guitar: Guitar | None = None,
        *,
        default_time_signature: tuple[int, int] = (4, 4),
        fallback_tempo_bpm: float = 120.0,
    ) -> None:
        self.guitar = guitar or Guitar.standard()
        numerator, denominator = default_time_signature
        if numerator < 1 or denominator < 1:
            raise ValueError("time signature values must be positive")
        if fallback_tempo_bpm <= 0:
            raise ValueError("fallback_tempo_bpm must be positive")
        self.default_time_signature = default_time_signature
        self.fallback_tempo_bpm = float(fallback_tempo_bpm)
        self.optimizer = FingeringOptimizer(self.guitar)

    def _quantized_notes(
        self, analysis: AudioAnalysis
    ) -> tuple[QuantizedNote, ...]:
        if analysis.rhythm is not None and analysis.rhythm.quantized_notes:
            return analysis.rhythm.quantized_notes
        timing = TimingInfo(
            tempo_bpm=self.fallback_tempo_bpm,
            time_signature=self.default_time_signature,
            subdivision=4,
        )
        return quantize_notes(analysis.notes, timing)

    @staticmethod
    def _group_quantized(
        notes: tuple[QuantizedNote, ...]
    ) -> tuple[tuple[QuantizedNote, ...], ...]:
        grouped: dict[float, list[QuantizedNote]] = defaultdict(list)
        for item in notes:
            key = round(
                item.start_beat if item.start_beat is not None else item.note.start,
                6,
            )
            grouped[key].append(item)
        return tuple(
            tuple(sorted(grouped[key], key=lambda item: item.note.midi))
            for key in sorted(grouped)
        )

    def generate(self, analysis: AudioAnalysis) -> Tablature:
        """Generate tablature while retaining rests and mapping failures."""

        quantized = self._quantized_notes(analysis)
        timing = analysis.rhythm.timing if analysis.rhythm is not None else TimingInfo()
        time_signature = timing.time_signature or self.default_time_signature
        numerator, denominator = time_signature
        beats_per_measure = numerator * 4.0 / denominator
        subdivision = timing.subdivision
        tempo_bpm = timing.tempo_bpm
        diagnostics: list[str] = []
        if tempo_bpm is None:
            diagnostics.append(
                f"未检测到稳定 BPM；TAB 排版临时按 {self.fallback_tempo_bpm:.0f} BPM。"
            )

        groups = self._group_quantized(quantized)
        technique_by_note = {
            detection.note_key: detection for detection in analysis.techniques
        }
        position_groups = self.optimizer.optimize_groups(
            tuple(tuple(item.note for item in group) for group in groups)
        )

        events: list[TabEvent] = []
        unmapped: list[UnmappedNote] = []
        maximum_beat = 0.0
        for group, positions in zip(groups, position_groups):
            for item, position in zip(group, positions):
                technique = technique_by_note.get(
                    (
                        item.source.midi,
                        round(item.source.start, 6),
                        round(item.source.duration, 6),
                    )
                )
                start_beat = item.start_beat
                duration_beats = item.duration_beats
                if start_beat is None:
                    start_beat = item.note.start * self.fallback_tempo_bpm / 60.0
                if duration_beats is None:
                    duration_beats = max(
                        1.0 / subdivision,
                        item.note.duration * self.fallback_tempo_bpm / 60.0,
                    )
                measure = int(start_beat // beats_per_measure) + 1
                maximum_beat = max(maximum_beat, start_beat + duration_beats)
                if position is None:
                    unmapped.append(
                        UnmappedNote(
                            note=item.source,
                            reason="音符超出当前吉他的可演奏音域或和弦琴弦冲突",
                            start_beat=start_beat,
                            measure=measure,
                        )
                    )
                    continue
                end_measure = int(
                    (start_beat + duration_beats - 1e-9) // beats_per_measure
                ) + 1
                events.append(
                    TabEvent(
                        string=position.string,
                        fret=position.fret,
                        start=item.note.start,
                        duration=item.note.duration,
                        note=item.source,
                        start_beat=start_beat,
                        duration_beats=duration_beats,
                        measure=measure,
                        tie_to_next=item.tie_to_next or end_measure > measure,
                        technique=(technique.technique.value if technique else None),
                        technique_confidence=(
                            technique.confidence if technique else None
                        ),
                        confidence=item.source.confidence,
                    )
                )

        tab_rests: list[TabRest] = []
        if analysis.rhythm is not None:
            for rest in analysis.rhythm.rests:
                start_beat = rest.start_beat
                duration_beats = rest.duration_beats
                if start_beat is None:
                    start_beat = rest.start * self.fallback_tempo_bpm / 60.0
                if duration_beats is None:
                    duration_beats = rest.duration * self.fallback_tempo_bpm / 60.0
                measure = int(start_beat // beats_per_measure) + 1
                maximum_beat = max(maximum_beat, start_beat + duration_beats)
                tab_rests.append(
                    TabRest(
                        start=rest.start,
                        duration=rest.duration,
                        start_beat=start_beat,
                        duration_beats=duration_beats,
                        measure=measure,
                    )
                )

        measure_count = max(1, math.ceil(maximum_beat / beats_per_measure))
        return Tablature(
            guitar=self.guitar,
            events=tuple(sorted(events, key=lambda event: (event.start_beat or 0, event.string))),
            rests=tuple(tab_rests),
            unmapped_notes=tuple(unmapped),
            tempo_bpm=tempo_bpm,
            time_signature=time_signature,
            subdivision=subdivision,
            measure_count=measure_count,
            diagnostics=tuple(diagnostics),
        )


__all__ = ["TabGenerator"]
