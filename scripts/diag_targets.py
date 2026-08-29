"""Per-target diagnostic: why does each answer event miss?

For every answer target (time, string, fret) it reports the CQT strength of the
expected midi inside the covering segment, whether it passed the candidate
thresholds, and what the detector emitted instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.match_answer import _answer_targets  # noqa: E402
from src.audio.loader import load_audio  # noqa: E402
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer  # noqa: E402
from src.eval.answer_tab import parse_answer_tab  # noqa: E402
from src.music.guitar import Guitar  # noqa: E402

AUDIO = Path("/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3")
ANSWER = Path("/Users/youzi/Downloads/晴天_1-8小节_TAB_含击勾弦.txt")
OFFSET = 25.0
BARS = 8


def main() -> int:
    audio = load_audio(AUDIO)
    events = parse_answer_tab(ANSWER.read_text(encoding="utf-8"))
    events = [e for e in events if e.measure <= BARS]
    targets = _answer_targets(events, 72.0)
    guitar = Guitar.standard()

    analyzer = PolyphonicAudioAnalyzer(
        attack_weight=0.35,
        harmonic_ratio=0.58,
        log_compress=True,
    )
    analysis = analyzer.analyze(audio)
    waveform = analyzer._waveform(audio)
    strengths, rms, frame_times = analyzer._midi_strengths(
        waveform, audio.sample_rate
    )
    onset_times, _timing = analyzer.rhythm_analyzer.detect(audio)
    boundaries = analyzer._boundaries(audio.duration, tuple(onset_times))

    global_peak = float(np.max(rms)) if rms.size else 0.0

    rows = []
    for time, string, fret, measure, technique in targets:
        moment = time + OFFSET
        midi = guitar.midi_at(string, fret)
        seg_start = max(b for b in boundaries if b <= moment)
        seg_end = min((b for b in boundaries if b > moment), default=audio.duration)
        frame_mask = (frame_times >= seg_start) & (frame_times < seg_end)
        voiced = frame_mask & (rms >= global_peak * analyzer.energy_threshold)
        if not np.any(voiced):
            rows.append((measure, string, fret, midi, "NO_VOICED", 0.0, (), ()))
            continue
        scores = np.median(strengths[:, voiced], axis=1)
        attack_window = min(0.12, max(0.03, (seg_end - seg_start) * 0.35))
        attack_mask = voiced & (frame_times < seg_start + attack_window)
        if np.any(attack_mask):
            attack_scores = np.max(strengths[:, attack_mask], axis=1)
            scores = (1 - analyzer.attack_weight) * scores + analyzer.attack_weight * attack_scores
        scores = np.maximum(scores - float(np.median(scores)), 0.0)
        scores = np.log1p(scores * 40.0)
        maximum = float(scores.max())
        idx = midi - analyzer.min_midi
        target_score = float(scores[idx]) if 0 <= idx < scores.size else 0.0
        rank = int((scores > target_score).sum()) + 1
        seg_midis = tuple(
            n.midi for n in analysis.notes if seg_start - 0.05 <= n.start < seg_end
        )
        threshold = maximum * analyzer.relative_pitch_threshold
        if target_score == 0.0:
            cause = "ZERO_SCORE"
        elif target_score < threshold:
            cause = f"BELOW_THRESH r={target_score / maximum:.3f}"
        else:
            left = scores[idx - 1] if idx > 0 else -1.0
            right = scores[idx + 1] if idx + 1 < scores.size else -1.0
            if target_score < left or target_score < right:
                cause = "NOT_PEAK"
            else:
                cause = "SHOULD_PASS"
        rows.append((measure, string, fret, midi, cause, target_score, (rank,), seg_midis))

    from collections import Counter

    counts = Counter(r[4].split()[0] for r in rows)
    print("cause counts:", dict(counts))
    print("\nmisses (cause, measure, string, fret, midi, rank, detected):")
    hits = 0
    for measure, string, fret, midi, cause, _, rank, seg_midis in rows:
        matched = any(m == midi for m in seg_midis)
        if matched:
            hits += 1
            continue
        print(
            f"  bar{measure} s{string} f{fret} midi{midi} {cause} rank={rank[0]} "
            f"seg={sorted(seg_midis)}"
        )
    print(f"\nsegments containing target midi: {hits}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
