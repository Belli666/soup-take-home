#!/usr/bin/env bash
set -euo pipefail

rc=0
soup ship --base ./base_snapshot --tuned ./artifacts/merged \
  --task-eval ./data/ship-dev.jsonl --task-mode metric \
  --output "$PHASE_LOG_DIR/ship-merged.json" || rc=$?

echo "SOUP_SHIP_EXIT=$rc"
cp "$PHASE_LOG_DIR/ship-merged.json" results/ship-merged.json 2>/dev/null || true
# Note: exit code 0 = SHIP, 2 = DON'T SHIP; both are valid diagnostic verdicts from soup ship.
if (( rc != 0 && rc != 2 )); then
  exit "$rc"
fi
