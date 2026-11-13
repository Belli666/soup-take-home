#!/usr/bin/env bash
set -euo pipefail

python -u scripts/eval_3topics_4models.py \
  --base-dir base_snapshot \
  --adapter-dir artifacts/adapter \
  --merged-dir artifacts/merged \
  --nf4-dir artifacts/nf4 \
  --ru-jsonl data/raw/ru-RU.jsonl \
  --manifest data/manifest.json \
  --train-jsonl data/train.jsonl \
  --dev-jsonl data/dev.jsonl \
  --dataset-out data/blind-300.jsonl \
  --paired-jsonl results/blind-300-paired.jsonl \
  --paired-csv results/blind-300-paired.csv \
  --summary-out results/blind-300-summary.json

cp data/blind-300.jsonl "$PHASE_LOG_DIR/"
cp results/blind-300-paired.jsonl results/blind-300-paired.csv results/blind-300-summary.json "$PHASE_LOG_DIR/"
