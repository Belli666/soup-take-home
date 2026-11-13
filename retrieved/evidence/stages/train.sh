#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts
python -u scripts/measure_soup.py train --config soup.yaml --yes

if [[ -d artifacts/adapter ]]; then
  sha256sum artifacts/adapter/* > "$PHASE_LOG_DIR/adapter-sha256.txt" 2>/dev/null || true
fi
cp "$PHASE_LOG_DIR/cuda-memory.json" results/train-cuda-memory.json 2>/dev/null || true
