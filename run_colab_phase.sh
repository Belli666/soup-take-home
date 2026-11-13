#!/usr/bin/env bash
# Асинхронный лаунчер фаз на Google Colab через google-colab-cli:
# запускает стадию на VM через nohup в фоне и опрашивает status.txt короткими вызовами с ретраями,
# чтобы обрыв WebSocket (Connection was lost) не обрывал долгое обучение на GPU.
set -uo pipefail

PHASE=${1:?phase required}
SESSION=${COLAB_SESSION:-soup-t4-01}
mkdir -p local-logs
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG_FILE="local-logs/${PHASE}-${STAMP}.log"

run_exec_retry() {
  local code_file=$1
  local attempt out rc
  for attempt in 1 2 3 4; do
    out=$(timeout 30 colab exec --timeout 25 -s "$SESSION" -f "$code_file" 2>&1)
    rc=$?
    if [[ $rc -eq 0 ]]; then
      printf '%s\n' "$out"
      return 0
    fi
    echo "[run_colab_phase] attempt $attempt failed (rc=$rc): ${out: -200}" >&2
    sleep 3
  done
  printf '%s\n' "$out" >&2
  return 1
}

LAUNCH_PY=$(mktemp)
POLL_PY=$(mktemp)
trap 'rm -f "$LAUNCH_PY" "$POLL_PY"' EXIT

cat > "$LAUNCH_PY" <<PY
import subprocess, time
from pathlib import Path

logs_dir = Path('/content/soup_takehome/logs')
before = {p.name for p in logs_dir.glob('*')} if logs_dir.exists() else set()

p = subprocess.Popen(
    ['bash', '-lc', 'nohup bash run_phase.sh $PHASE > /tmp/bg_${PHASE}.out 2>&1 < /dev/null & echo \$!'],
    cwd='/content/soup_takehome',
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)
pid_str, err = p.communicate()
pid = pid_str.strip()
log_dir_name = ''
for _ in range(20):
    after = {p.name for p in logs_dir.glob('*-$PHASE-*')} if logs_dir.exists() else set()
    new_dirs = sorted(after - before)
    if new_dirs:
        log_dir_name = new_dirs[-1]
        break
    time.sleep(0.5)
print(f'LAUNCHED_PID={pid}')
print(f'LOG_DIR={log_dir_name}')
PY

launch_out=$(run_exec_retry "$LAUNCH_PY") || exit 1
echo "$launch_out" | tee "$LOG_FILE"
PID=$(printf '%s\n' "$launch_out" | awk -F= '/^LAUNCHED_PID=/{print $2}' | tail -n1)
LOG_DIR=$(printf '%s\n' "$launch_out" | awk -F= '/^LOG_DIR=/{print $2}' | tail -n1)

if [[ -z "$LOG_DIR" ]]; then
  echo "Failed to detect LOG_DIR for phase $PHASE" >&2
  exit 1
fi

cat > "$POLL_PY" <<PY
import os, subprocess
from pathlib import Path

d = Path('/content/soup_takehome/logs') / '$LOG_DIR'
st_file = d / 'status.txt'
st = st_file.read_text() if st_file.exists() else ''
alive = Path('/proc/$PID').exists() if '$PID' else False
done = ('ended_utc=' in st) and not alive
print(f'DONE={int(done)} ALIVE={int(alive)}')
if done:
    print('=== STATUS ===')
    print(st.strip())
    raw = d / 'raw.log'
    if raw.exists():
        print('=== RAW_LOG ===')
        print(raw.read_text(errors='replace'))
else:
    raw = d / 'raw.log'
    if raw.exists():
        lines = raw.read_text(errors='replace').splitlines()
        print('TAIL:', lines[-1] if lines else '(empty)')
PY

while true; do
  sleep 10
  poll_out=$(run_exec_retry "$POLL_PY") || continue
  if printf '%s\n' "$poll_out" | grep -q '^DONE=1'; then
    printf '%s\n' "$poll_out" | tee -a "$LOG_FILE"
    phase_exit=$(printf '%s\n' "$poll_out" | awk -F= '/^phase_exit=/{print $2}' | tail -n1)
    exit "${phase_exit:-1}"
  fi
  tail_line=$(printf '%s\n' "$poll_out" | grep '^TAIL:' | head -n1 || true)
  echo "[$(date -u +%FT%TZ)] running ($LOG_DIR) $tail_line"
done
