"""Compute strict recall at forced offsets to verify alignment quality."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.loader import load_audio
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer
from src.audio.track_classifier import TrackClassifier
from src.audio.transcription_service import TranscriptionService
from src.eval.answer_tab import parse_answer_tab
from src.music.guitar import Guitar
from scripts.match_answer import _answer_targets, _strict_score, _sequence_score


def score_at(offset: float) -> None:
    audio_path = Path('/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3')
    answer_path = Path('/Users/youzi/Downloads/晴天_1-8小节_TAB_含击勾弦.txt')
    audio = load_audio(audio_path)
    events = parse_answer_tab(answer_path.read_text(encoding='utf-8'))
    guitar = Guitar.standard()
    service = TranscriptionService(
        analyzer=PolyphonicAudioAnalyzer(attack_weight=0.35, harmonic_ratio=0.58, log_compress=True),
        track_classifier=TrackClassifier(),
    )
    result = service.transcribe(audio_path, audio=audio)
    targets = _answer_targets(events, 72.0)
    rep = _strict_score(targets, result.tablature, offset, guitar.midi_at, result.analysis.notes)
    rep.update(_sequence_score(targets, result.tablature, offset))
    print(f"offset={offset:.2f}: strict={rep['recall']} midi={rep['midi_recall']} seq={rep['sequence_recall']} hits={rep['hits']}/{rep['total']}")


if __name__ == '__main__':
    for off in [25.0, 41.5, 55.15, 62.0, 63.0, 63.5, 70.3, 76.6]:
        score_at(off)
