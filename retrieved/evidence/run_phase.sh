#!/usr/bin/env bash
set -uo pipefail
export TZ=UTC PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

phase=${1:?phase required}
case "$phase" in
  setup|data|preflight|baseline|bench|train|merge|quant|verify|probes|ship) ;;
  *)
    echo "Unknown phase: $phase" >&2
    exit 2
    ;;
esac

mkdir -p logs || exit 1
dir=$(mktemp -d "logs/$(date -u +%Y%m%dT%H%M%SZ)-${phase}-XXXXXX") || exit 1
export PHASE_LOG_DIR="$PWD/$dir"
printf 'started_utc=%s\nphase=%s\n' "$(date -u +%FT%TZ)" "$phase" > "$dir/status.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$dir/nvidia-smi-before.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,index,name,driver_version,memory.total,memory.used,utilization.gpu \
    --format=csv -lms 200 > "$dir/nvidia-smi.csv" 2> "$dir/nvidia-smi.stderr" &
  monitor=$!
  trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
else
  echo "nvidia-smi not found" > "$dir/nvidia-smi-before.txt"
fi

bash -euo pipefail "stages/$phase.sh" 2>&1 | tee "$dir/raw.log" |
  awk '{ print strftime("%Y-%m-%dT%H:%M:%SZ", systime()), $0; fflush() }' |
  tee "$dir/timestamped.log"
codes=("${PIPESTATUS[@]}")

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$dir/nvidia-smi-after.txt" 2>&1
fi

printf 'ended_utc=%s\nphase_exit=%s\npipeline_codes=%s\n' \
  "$(date -u +%FT%TZ)" "${codes[0]}" "${codes[*]}" >> "$dir/status.txt"

for code in "${codes[@]}"; do
  if (( code != 0 )); then
    exit "$code"
  fi
done
