#!/usr/bin/env sh
# Autonomous tuning loop: repeatedly score generated TAB against the answer.
#
# The loop sweeps analyzer parameters and keeps the best-scoring report. It
# never edits source code; use the winning configuration to discuss or apply a
# change.
#
# Usage:
#   ./loop.sh                      # run forever until Ctrl-C (or score reaches 1.0)
#   GB_LOOP_MAX_CYCLES=2 ./loop.sh # run two cycles and stop
#
# Environment overrides:
#   GB_AUDIO   default: the local 晴天 mp3
#   GB_ANSWER  default: the local answer TAB txt
#   GB_BARS    default: 8

set -u
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PY="$script_dir/.venv/bin/python"
AUDIO=${GB_AUDIO:-"/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3"}
ANSWER=${GB_ANSWER:-"/Users/youzi/Downloads/晴天_1-8小节_TAB_含击勾弦.txt"}
BARS=${GB_BARS:-8}
OUT_DIR="$script_dir/build/answer-loop"
BEST_FILE="$OUT_DIR/best.json"
LOG="$OUT_DIR/loop.log"
mkdir -p "$OUT_DIR"

best_score=0
if [ -f "$BEST_FILE" ]; then
  best_score=$("$PY" -c "import json;print(json.load(open('$BEST_FILE'))['recall'])" 2>/dev/null || echo 0)
fi

cycle=0
while :; do
  cycle=$((cycle + 1))
  printf '%s cycle %s (best so far %s)\n' "$(date +%H:%M:%S)" "$cycle" "$best_score" >> "$LOG"
  for aw in 0.0 0.2 0.35; do
    for rel in 0.12 0.16 0.2 0.24; do
      for harm in 0.45 0.58 0.7; do
        report="$OUT_DIR/last.json"
        "$PY" "$script_dir/scripts/match_answer.py" \
          --audio "$AUDIO" --answer "$ANSWER" --bars "$BARS" \
          --attack-weight "$aw" --relative-threshold "$rel" --harmonic-ratio "$harm" \
          --output "$report" 2>>"$LOG"
        score=$("$PY" -c "import json;d=json.load(open('$report'));print(d['recall'])" 2>/dev/null || echo 0)
        better=$(awk -v a="$score" -v b="$best_score" 'BEGIN{print (a>b)?1:0}')
        if [ "$better" = "1" ]; then
          best_score="$score"
          cp "$report" "$BEST_FILE"
          printf '%s NEW BEST %s aw=%s rel=%s harm=%s\n' "$(date +%H:%M:%S)" "$score" "$aw" "$rel" "$harm" >> "$LOG"
          echo "new best: $score (aw=$aw rel=$rel harm=$harm)"
        fi
        if [ "$best_score" = "1.0" ]; then
          echo "perfect match reached" | tee -a "$LOG"
          exit 0
        fi
      done
    done
  done
  if [ "${GB_LOOP_MAX_CYCLES:-0}" -gt 0 ] && [ "$cycle" -ge "${GB_LOOP_MAX_CYCLES:-0}" ]; then
    echo "done: best=$best_score ($BEST_FILE)"
    exit 0
  fi
done
