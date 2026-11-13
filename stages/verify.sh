#!/usr/bin/env bash
set -euo pipefail

python -u scripts/verify.py --mode full \
  --base-dir ./base_snapshot \
  --adapter-dir ./artifacts/adapter \
  --merged-dir ./artifacts/merged \
  --nf4-dir ./artifacts/nf4 \
  --manifest ./data/manifest.json \
  --dev ./data/dev.jsonl \
  --test ./data/test.jsonl \
  --out-dir results

cp results/verify-summary.json "$PHASE_LOG_DIR/verify-summary.json"
cp results/verify-per-example.jsonl "$PHASE_LOG_DIR/verify-per-example.jsonl"
