#!/usr/bin/env bash
set -euo pipefail

PROBE_DIR="$PHASE_LOG_DIR/probes"
mkdir -p "$PROBE_DIR"

echo "=== PROBE 1: Label swap in train copy (data doctor) ==="
python - "$PROBE_DIR/train-label-swap.jsonl" <<'PY'
import json, sys
from pathlib import Path

rows = Path("data/train.jsonl").read_text(encoding="utf-8").splitlines()
first = json.loads(rows[0])
orig = first["messages"][-1]["content"]
# Swap to another valid label from manifest
manifest = json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))
other = [x for x in manifest["selected_intents"] if x != orig][0]
first["messages"][-1]["content"] = other
rows[0] = json.dumps(first, ensure_ascii=False)
Path(sys.argv[1]).write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"Swapped label row 0: {orig} -> {other}")
PY
sha256sum "$PROBE_DIR/train-label-swap.jsonl"
rc=0
soup data doctor "$PROBE_DIR/train-label-swap.jsonl" --model ./base_snapshot --max-length 512 \
  --sample 200 --train-on-responses-only --train-on-eot \
  --output "$PROBE_DIR/probe1-doctor.json" || rc=$?
echo "PROBE_1_EXIT=$rc (data doctor checks structure/tokens, not semantic label validity)"

echo "=== PROBE 2: Overlong user input truncating assistant response ==="
python - "$PROBE_DIR/train-overlong.jsonl" <<'PY'
import json, sys
from pathlib import Path

rows = Path("data/train.jsonl").read_text(encoding="utf-8").splitlines()[:20]
first = json.loads(rows[0])
first["messages"][1]["content"] = ("длинное сообщение пользователя " * 180).strip()
rows[0] = json.dumps(first, ensure_ascii=False)
Path(sys.argv[1]).write_text("\n".join(rows) + "\n", encoding="utf-8")
PY
sha256sum "$PROBE_DIR/train-overlong.jsonl"
rc=0
soup data doctor "$PROBE_DIR/train-overlong.jsonl" --model ./base_snapshot --max-length 512 \
  --sample 20 --show-mask 2 --train-on-responses-only --train-on-eot \
  --output "$PROBE_DIR/probe2-doctor.json" || rc=$?
echo "PROBE_2_EXIT=$rc"

echo "=== PROBE 3: Fresh zero-B LoRA adapter (verify.py + soup ship) ==="
python - "$PROBE_DIR/zero_b_adapter" <<'PY'
import sys
from pathlib import Path
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

out_dir = Path(sys.argv[1])
base = AutoModelForCausalLM.from_pretrained("./base_snapshot", local_files_only=True, torch_dtype=torch.float16)
tok = AutoTokenizer.from_pretrained("./base_snapshot", local_files_only=True)
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
peft = get_peft_model(base, cfg)
peft.save_pretrained(str(out_dir))
tok.save_pretrained(str(out_dir))
print(f"Saved fresh zero-B adapter to {out_dir}")
PY
rc_verify=0
python -u scripts/verify.py --mode adapter-activity --adapter-dir "$PROBE_DIR/zero_b_adapter" || rc_verify=$?
echo "PROBE_3_VERIFY_EXIT=$rc_verify (expected non-zero)"

rc_ship=0
soup ship --base ./base_snapshot --adapter "$PROBE_DIR/zero_b_adapter" \
  --task-eval ./data/ship-dev.jsonl --task-mode metric \
  --output "$PROBE_DIR/probe3-ship-zero-b.json" || rc_ship=$?
echo "PROBE_3_SHIP_EXIT=$rc_ship (expected 2 for adapter_noop DON'T SHIP)"

echo "=== PROBE 4: Base model passed as merge (merge-parity) ==="
rc=0
python -u scripts/verify.py --mode merge-parity --merged-dir ./base_snapshot || rc=$?
echo "PROBE_4_EXIT=$rc (expected non-zero)"

echo "=== PROBE 5: Dense FP16 merge passed as NF4 (quantize_nf4 --verify-reload) ==="
rc=0
python -u scripts/quantize_nf4.py --mode verify-reload --nf4-dir ./artifacts/merged --base-dir ./base_snapshot || rc=$?
echo "PROBE_5_EXIT=$rc (expected non-zero)"
