#!/usr/bin/env python3
"""
Вспомогательный скрипт для инспекции реальных токенов и предполагаемой маски loss (labels)
по данным сохранённого train.jsonl и rewind.jsonl (для live discussion).
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect batch row from rewind.jsonl and train.jsonl.")
    parser.add_argument("--row", type=int, default=688, help="Row index in train.jsonl (default: 688 = step 1 micro 0)")
    args = parser.parse_args()

    train_rows = [json.loads(l) for l in Path("retrieved/evidence/data/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rewind_rows = [json.loads(l) for l in Path("retrieved/evidence/artifacts/adapter/rewind.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    batch_entry = next((r for r in rewind_rows if r.get("kind") == "batch" and args.row in r.get("rows", [])), None)
    sample = train_rows[args.row]
    user_msg = next(m["content"] for m in sample["messages"] if m["role"] == "user")
    asst_msg = next(m["content"] for m in sample["messages"] if m["role"] == "assistant")

    print(f"=== Train row {args.row} ===")
    print(f"User:      {user_msg}")
    print(f"Assistant: {asst_msg}")
    if batch_entry:
        print(f"Rewind:    step={batch_entry['step']}, micro={batch_entry['micro']}, row_tokens={batch_entry['row_tokens']}, row_loss={batch_entry['row_loss']}")

    total_rewind = sum(sum(r["row_tokens"]) for r in rewind_rows if r.get("kind") == "batch")
    audit = json.loads(Path("retrieved/evidence/results/data-token-audit.json").read_text(encoding="utf-8"))
    total_audit = audit["total_supervised_tokens_in_train"]
    diff = total_rewind - total_audit
    print(f"\nTotal rewind tokens: {total_rewind} vs data-token-audit answer-only: {total_audit}")
    print(f"Difference: +{diff} tokens across {len(train_rows)} rows (+{diff / len(train_rows):.1f} role-prefix tokens per example: <|im_start|>, assistant, \\n)")


if __name__ == "__main__":
    main()
