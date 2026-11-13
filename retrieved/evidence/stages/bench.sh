#!/usr/bin/env bash
set -euo pipefail

python -u scripts/measure_soup.py bench train --config soup.yaml --steps 20 --warmup 3 \
  --output "$PHASE_LOG_DIR/bench-train.json"
cp "$PHASE_LOG_DIR/bench-train.json" results/bench-train.json
