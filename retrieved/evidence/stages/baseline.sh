#!/usr/bin/env bash
set -euo pipefail

python -u scripts/verify.py --mode base-dev --out-dir results
cp results/baseline-dev.json "$PHASE_LOG_DIR/baseline-dev.json"
