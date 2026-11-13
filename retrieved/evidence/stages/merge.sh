#!/usr/bin/env bash
set -euo pipefail

if command -v /usr/bin/time >/dev/null 2>&1; then
  /usr/bin/time -v -o "$PHASE_LOG_DIR/cpu-time.txt" \
    soup merge --base ./base_snapshot --adapter ./artifacts/adapter \
    --output ./artifacts/merged --dtype float16
else
  python - "$PHASE_LOG_DIR/cpu-time.txt" <<'PY'
import resource, subprocess, sys, time
t0 = time.perf_counter()
res = subprocess.run([
    "soup", "merge",
    "--base", "./base_snapshot",
    "--adapter", "./artifacts/adapter",
    "--output", "./artifacts/merged",
    "--dtype", "float16",
])
dt = time.perf_counter() - t0
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
summary = (
    f"Elapsed (wall clock) seconds: {dt:.3f}\n"
    f"Maximum resident set size (kbytes): {ru.ru_maxrss}\n"
    f"Exit status: {res.returncode}\n"
)
open(sys.argv[1], "w").write(summary)
print(summary.strip())
sys.exit(res.returncode)
PY
fi

sha256sum artifacts/merged/* > "$PHASE_LOG_DIR/merged-sha256.txt" 2>/dev/null || true
cp "$PHASE_LOG_DIR/cpu-time.txt" results/merge-cpu-time.txt 2>/dev/null || true
