#!/usr/bin/env python3
"""
Quantize saved FP16 merged model to standalone 4-bit NF4 (bitsandbytes) and verify reload.
Follows Section 8 of soup-take-home-colab-cli-plan.md.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def sha256_dir(directory: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            hashes[str(p.relative_to(directory))] = h.hexdigest()
    return hashes


def dir_size_bytes(directory: Path) -> int:
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def quantize(merged_dir: Path, nf4_dir: Path) -> None:
    if not merged_dir.exists():
        raise FileNotFoundError(f"Merged directory not found: {merged_dir}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        local_files_only=True,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir), local_files_only=True)

    nf4_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(nf4_dir))
    tokenizer.save_pretrained(str(nf4_dir))
    elapsed = time.perf_counter() - t0

    meta: Dict[str, Any] = {
        "merged_dir": str(merged_dir),
        "nf4_dir": str(nf4_dir),
        "elapsed_sec": round(elapsed, 3),
        "merged_disk_bytes": dir_size_bytes(merged_dir),
        "nf4_disk_bytes": dir_size_bytes(nf4_dir),
        "merged_sha256": sha256_dir(merged_dir),
        "nf4_sha256": sha256_dir(nf4_dir),
        "bnb_config": bnb_config.to_dict(),
        "package_versions": {
            pkg: importlib.metadata.version(pkg)
            for pkg in ["torch", "transformers", "bitsandbytes", "accelerate"]
        },
        "note": "Supported Linear layers are quantized to 4-bit NF4; embeddings/lm_head/norms remain FP16.",
    }
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated()
        res = torch.cuda.max_memory_reserved()
        meta["cuda_peak"] = {
            "max_memory_allocated_bytes": alloc,
            "max_memory_allocated_mib": round(alloc / (1024**2), 2),
            "max_memory_reserved_bytes": res,
            "max_memory_reserved_mib": round(res / (1024**2), 2),
        }

    for out_dir in [Path(os.environ.get("PHASE_LOG_DIR", "results")), Path("results")]:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "quant-meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(f"[OK] Saved NF4 model to {nf4_dir} ({meta['nf4_disk_bytes'] / (1024**2):.2f} MiB)")


def verify_reload(nf4_dir: Path, base_dir: Path) -> None:
    if not nf4_dir.exists():
        print(f"[FAIL] NF4 directory not found: {nf4_dir}", file=sys.stderr)
        sys.exit(2)

    config_file = nf4_dir / "config.json"
    if not config_file.exists():
        print(f"[FAIL] Missing config.json in {nf4_dir}", file=sys.stderr)
        sys.exit(2)

    cfg_json = json.loads(config_file.read_text(encoding="utf-8"))
    if "quantization_config" not in cfg_json:
        print(
            f"[FAIL] {nf4_dir} has no quantization_config in config.json (dense model passed as NF4?).",
            file=sys.stderr,
        )
        sys.exit(3)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        str(nf4_dir),
        local_files_only=True,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
    )

    is_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
    linear4bit_count = 0
    nf4_types = set()
    for _, mod in model.named_modules():
        if isinstance(mod, bnb.nn.Linear4bit):
            linear4bit_count += 1
            qs = getattr(mod, "quant_state", None)
            if qs is not None and hasattr(qs, "quant_type"):
                nf4_types.add(qs.quant_type)

    has_peft_wrapper = hasattr(model, "peft_config") or any(
        "lora_" in n for n, _ in model.named_modules()
    )

    tok_nf4 = AutoTokenizer.from_pretrained(str(nf4_dir), local_files_only=True)
    tok_base = AutoTokenizer.from_pretrained(str(base_dir), local_files_only=True)
    tok_match = (
        len(tok_nf4) == len(tok_base)
        and tok_nf4.eos_token_id == tok_base.eos_token_id
        and tok_nf4.chat_template == tok_base.chat_template
    )

    report = {
        "nf4_dir": str(nf4_dir),
        "is_loaded_in_4bit": is_4bit,
        "linear4bit_module_count": linear4bit_count,
        "quant_types_found": sorted(nf4_types),
        "has_peft_wrapper": has_peft_wrapper,
        "tokenizer_matches_base": tok_match,
        "disk_bytes": dir_size_bytes(nf4_dir),
    }
    if torch.cuda.is_available():
        report["cuda_allocated_mib"] = round(torch.cuda.memory_allocated() / (1024**2), 2)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not is_4bit or linear4bit_count == 0 or "nf4" not in nf4_types or has_peft_wrapper or not tok_match:
        print("[FAIL] NF4 reload verification failed!", file=sys.stderr)
        sys.exit(4)

    print("[OK] NF4 reload verification PASSED.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize merged FP16 to NF4 and verify reload.")
    parser.add_argument("--mode", choices=["quantize", "verify-reload"], required=True)
    parser.add_argument("--merged-dir", type=Path, default=Path("artifacts/merged"))
    parser.add_argument("--nf4-dir", type=Path, default=Path("artifacts/nf4"))
    parser.add_argument("--base-dir", type=Path, default=Path("base_snapshot"))
    args = parser.parse_args()

    if args.mode == "quantize":
        quantize(args.merged_dir, args.nf4_dir)
    else:
        verify_reload(args.nf4_dir, args.base_dir)


if __name__ == "__main__":
    main()
