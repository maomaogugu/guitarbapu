#!/usr/bin/env sh
# Continuously evaluate local reference-TAB cases and write timestamped reports.
# This script only runs evaluation and never edits source code by itself.

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
cases_json=${GB_EVAL_CASES:-"$project_root/eval_cases.local.json"}
output_dir=${GB_EVAL_OUTPUT_DIR:-"$project_root/build/eval-loop"}
sleep_seconds=${GB_EVAL_SLEEP_SECONDS:-300}
max_runs=${GB_EVAL_MAX_RUNS:-0}

if [ ! -f "$cases_json" ]; then
  echo "Cases file not found: $cases_json" >&2
  echo "Create it locally, for example:" >&2
  echo '[{"name":"qingtian-intro","audio":"/path/audio.mp3","tab":"/path/ref.txt"}]' >&2
  exit 2
fi

mkdir -p "$output_dir"
run_count=0
while :; do
  run_count=$((run_count + 1))
  stamp=$(date +%Y%m%dT%H%M%S)
  "$project_root/.venv/bin/python" "$project_root/scripts/evaluate_polyphonic_grid.py" \
    "$cases_json" \
    --output "$output_dir/report-$stamp.json"
  printf '%s\n' "$output_dir/report-$stamp.json" > "$output_dir/latest.txt"
  echo "wrote $output_dir/report-$stamp.json"
  if [ "$max_runs" -gt 0 ] && [ "$run_count" -ge "$max_runs" ]; then
    break
  fi
  sleep "$sleep_seconds"
done
