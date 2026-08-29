#!/usr/bin/env sh
# Self-driving tuning loop for the 晴天 answer TAB.
#
# Iterates polyphonic-analyzer parameters, scores each output against the
# hand-checked answer (bars 1-8), and keeps the best candidate.  It never
# edits source code; use the winning configuration to discuss a change.
#
# Stop conditions: score reaches 1.0, GB_LOOP_MAX_CYCLES exceeded, or Ctrl-C.
# Reports land in build/ailoop/:
#   best.json     best report so far
#   best-tab.txt  rendered TAB of the best candidate
#   loop.log      timestamped history

set -u
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PY="$script_dir/.venv/bin/python"
AUDIO=${GB_AUDIO:-"/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3"}
ANSWER=${GB_ANSWER:-"/Users/youzi/Downloads/晴天_1-8小节_TAB_含击勾弦.txt"}
BARS=${GB_BARS:-8}
OUT="$script_dir/build/ailoop"
mkdir -p "$OUT"
LOG="$OUT/loop.log"
BEST="$OUT/best.json"
BEST_TAB="$OUT/best-tab.txt"
best=
if [ -f "$BEST" ]; then
  best=$("$PY" -c "import json;print(json.load(open('$BEST'))['recall'])" 2>/dev/null)
fi
best=${best:-0}

cycle=0
while :; do
  cycle=$((cycle + 1))
  printf '%s cycle %s best=%s\n' "$(date +%m-%d\ %H:%M:%S)" "$cycle" "$best" >> "$LOG"
  for aw in 0.0 0.2 0.35 0.5; do
    for rel in 0.12 0.16 0.2 0.24 0.3; do
      for harm in 0.45 0.58 0.7; do
        report="$OUT/last.json"
        "$PY" "$script_dir/scripts/match_answer.py" \
          --audio "$AUDIO" --answer "$ANSWER" --bars "$BARS" \
          --attack-weight "$aw" --relative-threshold "$rel" --harmonic-ratio "$harm" \
          --output "$report" 2>>"$LOG" || continue
        score=$("$PY" -c "import json;print(json.load(open('$report'))['recall'])" 2>/dev/null || echo 0)
        better=$(awk -v a="$score" -v b="$best" 'BEGIN{print (a>b)?1:0}')
        if [ "$better" = 1 ]; then
          best="$score"
          cp "$report" "$BEST"
          printf '%s BEST recall=%s aw=%s rel=%s harm=%s\n' "$(date +%H:%M:%S)" "$score" "$aw" "$rel" "$harm" >> "$LOG"
          echo "best=$best aw=$aw rel=$rel harm=$harm"
          "$PY" "$script_dir/scripts/match_answer.py" \
            --audio "$AUDIO" --answer "$ANSWER" --bars "$BARS" \
            --attack-weight "$aw" --relative-threshold "$rel" --harmonic-ratio "$harm" \
            --export-tab "$BEST_TAB" --output "$OUT/best.json" 2>>"$LOG"
        fi
        if awk -v a="$best" 'BEGIN{exit !(a>=1.0)}'; then
          echo "perfect score reached" | tee -a "$LOG"
          exit 0
        fi
      done
    done
  done
  if [ "${GB_LOOP_MAX_CYCLES:-0}" -gt 0 ] && [ "$cycle" -ge "${GB_LOOP_MAX_CYCLES:-0}" ]; then
    echo "stopped after $cycle cycles; best=$best ($BEST)" | tee -a "$LOG"
    exit 0
  fi
done
