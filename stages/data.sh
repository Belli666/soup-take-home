#!/usr/bin/env bash
set -euo pipefail

python -u scripts/prepare_data.py --seed 42 --num-classes 12 --train-per-class 100 --dev-size 100 --test-size 264 --max-length 512
