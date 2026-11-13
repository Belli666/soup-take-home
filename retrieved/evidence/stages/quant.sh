#!/usr/bin/env bash
set -euo pipefail

# Step 1: Quantize saved FP16 merge to standalone NF4
python -u scripts/quantize_nf4.py --mode quantize \
  --merged-dir ./artifacts/merged --nf4-dir ./artifacts/nf4 --base-dir ./base_snapshot

# Step 2: Reload in a completely fresh Python process to prove standalone NF4 integrity
python -u scripts/quantize_nf4.py --mode verify-reload \
  --nf4-dir ./artifacts/nf4 --base-dir ./base_snapshot \
  > "$PHASE_LOG_DIR/nf4-reload-check.json"
