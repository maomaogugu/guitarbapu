"""Diagnostic: dump expected vs detected events per answer target."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.loader import load_audio
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer
from src.audio.track_classifier import TrackClassifier
from src.audio.transcription_service import TranscriptionService
from src.eval.answer_tab import parse_answer_tab
from src.music.guitar import Guitar

from scripts.match_answer import _answer_targets


def main() -> None:
    audio_path = Path('/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3')
    answer_path = Path('/Users/youzi/Downloads/晴天_1-8小节_TAB_含击勾弦.txt')
    audio = load_audio(audio_path)
    events = parse_answer_tab(answer_path.read_text(encoding='utf-8'))
    guitar = Guitar.standard()
    service = TranscriptionService(
        analyzer=PolyphonicAudioAnalyzer(
            attack_weight=0.35,
            relative_pitch_threshold=0.24,
            harmonic_ratio=0.58,
            log_compress=True,
        ),
        track_classifier=TrackClassifier(),
    )
    result = service.transcribe(audio_path, audio=audio)
    targets = _answer_targets(events, 72.0)
    bar_length = 4.0 * 60.0 / 72.0
    offset = 25.0

    tab_by_time = sorted(result.tablature.events, key=lambda e: e.start)
    note_by_time = sorted(result.analysis.notes, key=lambda n: n.start)

    def closest(events_by_time, t, tol=0.35):
        out = []
        for item in events_by_time:
            if abs(item.start - t) <= tol:
                out.append(item)
        return out

    for time, string, fret, measure, tech in targets:
        absolute = offset + time
        near_tab = closest(tab_by_time, absolute)
        near_notes = closest(note_by_time, absolute)
        target_midi = guitar.midi_at(string, fret)
        tab_matches = [
            (e.string, e.fret) for e in near_tab
            if e.string == string and e.fret == fret
        ]
        midi_matches = [
            n.midi for n in near_notes
            if n.midi in (target_midi, target_midi - 12, target_midi + 12)
        ]
        status = 'HIT' if tab_matches else ('MIDI' if midi_matches else 'miss')
        if status != 'HIT' or measure in (1, 2):
            print(
                f"m{measure} t={time:6.3f} abs={absolute:7.3f} "
                f"expect(s{string},f{fret})=midi{target_midi} -> {status}"
            )
            if near_tab:
                print("   tab:", [(round(e.start, 3), e.string, e.fret) for e in near_tab[:4]])
            if near_notes:
                print("   notes:", [(round(n.start, 3), n.midi, round(n.duration, 2)) for n in near_notes[:4]])


if __name__ == '__main__':
    main()
