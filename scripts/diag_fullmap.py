"""Compute the FULL strict score vs offset curve to find the true bar-1 location."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.audio.loader import load_audio
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer
from src.audio.track_classifier import TrackClassifier
from src.audio.transcription_service import TranscriptionService
from src.eval.answer_tab import parse_answer_tab
from src.music.guitar import Guitar
from scripts.match_answer import _answer_targets


def main():
    audio_path = Path('/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3')
    answer_path = Path('/Users/youzi/Downloads/晴天_1-8小节_TAB_含击勾弦.txt')
    audio = load_audio(audio_path)
    events = parse_answer_tab(answer_path.read_text(encoding='utf-8'))
    guitar = Guitar.standard()
    az = PolyphonicAudioAnalyzer(attack_weight=0.35, harmonic_ratio=0.58, log_compress=True)
    service = TranscriptionService(analyzer=az, track_classifier=TrackClassifier())
    result = service.transcribe(audio_path, audio=audio)
    tablature = result.tablature
    notes = result.analysis.notes
    targets = _answer_targets(events, 72.0)
    bar_len = 4.0 * 60.0 / 72.0
    span = targets[-1][0] + bar_len

    tab_evt = [(e.start, e.string, e.fret) for e in tablature.events]
    note_evt = [(n.start, n.midi) for n in notes]
    print(f"tab events: {len(tab_evt)}, notes: {len(note_evt)}")

    scores = []
    for offset in np.arange(0.0, audio.duration - span, 0.25):
        hits = 0
        midi_hits = 0
        for time, string, fret, _m, _t in targets:
            abs_t = offset + time
            target_midi = guitar.midi_at(string, fret)
            ok = any(abs(s - abs_t) <= 0.3 and ds == string and df == fret for s, ds, df in tab_evt)
            if ok:
                hits += 1
            else:
                if any(abs(s - abs_t) <= 0.3 and m in (target_midi, target_midi-12, target_midi+12) for s, m in note_evt):
                    midi_hits += 1
        scores.append((hits, hits+midi_hits, round(offset, 2)))
    scores.sort(reverse=True)
    print("\nTop 15 by strict hits:")
    for h, th, o in scores[:15]:
        print(f"  offset={o:6.2f} strict={h} +midi={th} ({h/len(targets):.3f})")


if __name__ == '__main__':
    main()
