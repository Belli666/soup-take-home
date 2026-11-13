#!/usr/bin/env bash
set -euo pipefail

soup data validate ./data/train.jsonl --format chatml --min-valid-fraction 1.0
soup data doctor ./data/train.jsonl --model ./base_snapshot --max-length 512 \
  --sample 2000 --show-mask 5 --train-on-responses-only --train-on-eot \
  --output "$PHASE_LOG_DIR/data-doctor.json"
soup doctor --config soup.yaml
soup train --config soup.yaml --dry-run --yes
