"""Render structured tablature as deterministic monospaced text."""

from .tab import Tablature


class TextTabRenderer:
    """Render four measures per block using a fixed rhythmic grid."""

    def __init__(self, *, measures_per_line: int = 4, cell_width: int = 3) -> None:
        if measures_per_line < 1:
            raise ValueError("measures_per_line must be positive")
        if cell_width < 2:
            raise ValueError("cell_width must be at least 2")
        self.measures_per_line = int(measures_per_line)
        self.cell_width = int(cell_width)

    @staticmethod
    def _string_labels(tablature: Tablature) -> dict[int, str]:
        labels: dict[int, str] = {}
        for guitar_string in tablature.guitar.strings:
            name = guitar_string.name
            if name == "high E":
                labels[guitar_string.number] = "e"
            elif name == "low E":
                labels[guitar_string.number] = "E"
            else:
                labels[guitar_string.number] = name[:1]
        return labels

    def _empty_measure(self, slots: int) -> list[str]:
        return ["-" * self.cell_width for _ in range(slots)]

    def render(self, tablature: Tablature) -> str:
        """Return a complete text TAB including headers and diagnostics."""

        numerator, denominator = tablature.time_signature
        slots_per_measure = round(
            tablature.beats_per_measure * tablature.subdivision
        )
        labels = self._string_labels(tablature)
        grids = {
            measure: {
                string.number: self._empty_measure(slots_per_measure)
                for string in tablature.guitar.strings
            }
            for measure in range(1, tablature.measure_count + 1)
        }

        for event in tablature.events:
            if event.start_beat is None or event.measure not in grids:
                continue
            measure_start = (event.measure - 1) * tablature.beats_per_measure
            slot = round(
                (event.start_beat - measure_start) * tablature.subdivision
            )
            slot = max(0, min(slots_per_measure - 1, slot))
            marker = str(event.fret)
            if event.tie_to_next:
                marker += "~"
            grids[event.measure][event.string][slot] = marker.ljust(
                self.cell_width, "-"
            )[: self.cell_width]

        for item in tablature.unmapped_notes:
            if item.start_beat is None or item.measure not in grids:
                continue
            measure_start = (item.measure - 1) * tablature.beats_per_measure
            slot = round(
                (item.start_beat - measure_start) * tablature.subdivision
            )
            slot = max(0, min(slots_per_measure - 1, slot))
            for string in grids[item.measure].values():
                string[slot] = "x".ljust(self.cell_width, "-")

        tempo = (
            f"{tablature.tempo_bpm:.1f} BPM"
            if tablature.tempo_bpm is not None
            else "未检测到稳定 BPM"
        )
        capo = tablature.guitar.capo
        tuning = "E A D G B E" + (f"   Capo {capo}" if capo else "")
        lines = [
            f"Tempo: {tempo}   Time: {numerator}/{denominator}   "
            f"Tuning: {tuning}",
            f"Mapped: {len(tablature.events)}   "
            f"Unmapped: {len(tablature.unmapped_notes)}",
            "",
        ]

        for block_start in range(
            1, tablature.measure_count + 1, self.measures_per_line
        ):
            measures = range(
                block_start,
                min(
                    tablature.measure_count + 1,
                    block_start + self.measures_per_line,
                ),
            )
            measure_tuple = tuple(measures)
            lines.append(
                "     "
                + "".join(
                    f"| M{measure:<{slots_per_measure * self.cell_width - 2}}"
                    for measure in measure_tuple
                )
                + "|"
            )
            beat_line = "".join(
                str(slot // tablature.subdivision + 1).ljust(
                    self.cell_width, " "
                )
                if slot % tablature.subdivision == 0
                else ".".ljust(self.cell_width, " ")
                for slot in range(slots_per_measure)
            )
            lines.append(
                "beat " + "".join(f"|{beat_line}" for _ in measure_tuple) + "|"
            )
            for string_number in range(1, len(tablature.guitar.strings) + 1):
                content = "".join(
                    "|" + "".join(grids[measure][string_number])
                    for measure in measure_tuple
                )
                lines.append(f"{labels[string_number]:>4} {content}|")
            lines.append("")

        if tablature.warnings:
            lines.append("警告：")
            lines.extend(f"- {warning}" for warning in tablature.warnings)
        techniques = [event for event in tablature.events if event.technique]
        if techniques:
            lines.append("")
            lines.append("技巧候选（请试听确认）：")
            for event in techniques:
                confidence = (
                    f"，可信度 {event.technique_confidence:.0%}"
                    if event.technique_confidence is not None
                    else ""
                )
                name = event.note.name if event.note is not None else "未知音符"
                lines.append(
                    f"- {event.start:.2f}s {name}: {event.technique}{confidence}"
                )
        return "\n".join(lines).rstrip()


__all__ = ["TextTabRenderer"]
