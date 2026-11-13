#!/usr/bin/env python3
"""
Verification script for Soup take-home task (Section 9 of soup-take-home-colab-cli-plan.md).
Checks:
1. Input SHA256 integrity against data/manifest.json
2. Adapter activity: (alpha/r) * B @ A norms + adapter-on vs adapter-off log-prob delta vs repeat/zero-B floor
3. Merge parity: FP16 base + adapter-on vs reloaded FP16 merge on teacher-forced target tokens + EOT
4. Artifact evaluation: base, merged, nf4 (and optional nf4-base) greedy exact match, valid-format rate, macro-F1,
   per-example JSONL, and paired bootstrap 95% CI (5000 resamples).
"""

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_inputs(manifest_path: Path, dev_path: Path, test_path: Path) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split_name, p in [("dev", dev_path), ("test", test_path)]:
        if p.exists() and split_name in manifest.get("splits", {}):
            expected_sha = manifest["splits"][split_name]["sha256"]
            actual_sha = sha256_file(p)
            if actual_sha != expected_sha:
                raise RuntimeError(f"SHA256 mismatch for {p}: expected {expected_sha}, got {actual_sha}")
    return manifest


def extract_ids(res: Any) -> List[int]:
    if isinstance(res, list):
        return res
    if hasattr(res, "get") or isinstance(res, dict):
        ids = res["input_ids"]
        return list(ids[0]) if ids and isinstance(ids[0], list) else list(ids)
    return list(res)


def get_target_logprobs(
    model,
    tokenizer,
    examples: List[Dict[str, Any]],
    max_length: int = 512,
) -> List[Dict[str, Any]]:
    device = next(model.parameters()).device
    results = []
    model.eval()

    with torch.inference_mode():
        for ex in examples:
            prompt_msgs = ex["messages"]
            full_msgs = prompt_msgs + [{"role": "assistant", "content": ex["expected"]}]

            prompt_ids = extract_ids(
                tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=True, add_generation_prompt=True, return_dict=False
                )
            )
            full_ids = extract_ids(
                tokenizer.apply_chat_template(
                    full_msgs, tokenize=True, add_generation_prompt=False, return_dict=False
                )
            )

            if len(full_ids) > max_length:
                raise RuntimeError(f"Example {ex['id']} exceeds max_length={max_length} ({len(full_ids)} tokens)")
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise RuntimeError(f"Chat template boundary mismatch on example {ex['id']}")

            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
            logits = model(input_ids=input_tensor).logits[0]  # [seq_len, vocab]
            log_probs = F.log_softmax(logits.float(), dim=-1)

            target_start = len(prompt_ids)
            target_tokens = full_ids[target_start:]
            # Position i-1 predicts token i
            pred_positions = list(range(target_start - 1, len(full_ids) - 1))
            token_lps = [
                float(log_probs[pos, tok_id].item())
                for pos, tok_id in zip(pred_positions, target_tokens)
            ]
            greedy_tokens = [
                int(torch.argmax(log_probs[pos]).item()) for pos in pred_positions
            ]
            results.append(
                {
                    "id": ex["id"],
                    "seq_len": len(full_ids),
                    "target_len": len(target_tokens),
                    "target_tokens": target_tokens,
                    "token_logprobs": token_lps,
                    "greedy_tokens": greedy_tokens,
                }
            )
    return results


def summarize_diffs(diffs: List[float]) -> Dict[str, float]:
    arr = np.array(diffs, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "median": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def check_adapter_activity(
    base_dir: Path,
    adapter_dir: Path,
    dev_examples: List[Dict[str, Any]],
    sample_n: int = 32,
) -> Dict[str, Any]:
    """
    Доказывает, что LoRA-адаптер реально обучился (не нулевой no-op):
    1) Считает норму дельт весов ΔW = (alpha / r) * (B @ A) по каждому целевому слою.
    2) Сравнивает сдвиг log-вероятностей при включённом и выключенном адаптере (on vs off)
       против двух уровней шума: повторного прогона (repeat_noise) и свежего нулевого адаптера (zero-B).
    """
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    scale = alpha / r

    sf_path = adapter_dir / "adapter_model.safetensors"
    if sf_path.exists():
        sd = load_file(str(sf_path))
    else:
        sd = torch.load(str(adapter_dir / "adapter_model.bin"), map_location="cpu")

    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        if ".lora_A." in k:
            prefix = k.split(".lora_A.")[0]
            pairs.setdefault(prefix, {})["A"] = v.float()
        elif ".lora_B." in k:
            prefix = k.split(".lora_B.")[0]
            pairs.setdefault(prefix, {})["B"] = v.float()

    delta_norms = []
    nonzero_layers = 0
    for prefix, ab in pairs.items():
        if "A" in ab and "B" in ab:
            delta = scale * (ab["B"] @ ab["A"])
            if not torch.isfinite(delta).all():
                raise RuntimeError(f"Non-finite LoRA delta in {prefix}")
            nrm = float(torch.norm(delta).item())
            delta_norms.append(nrm)
            if nrm > 1e-8:
                nonzero_layers += 1

    subset = dev_examples[:sample_n]
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), local_files_only=True)
    device_map = {"": 0} if torch.cuda.is_available() else None

    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_dir), local_files_only=True, torch_dtype=torch.float16, device_map=device_map
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)

    on_1 = get_target_logprobs(peft_model, tokenizer, subset)
    on_2 = get_target_logprobs(peft_model, tokenizer, subset)
    with peft_model.disable_adapter():
        off_1 = get_target_logprobs(peft_model, tokenizer, subset)

    on_off_diffs = [
        abs(a - b)
        for r1, r2 in zip(on_1, off_1)
        for a, b in zip(r1["token_logprobs"], r2["token_logprobs"])
    ]
    repeat_diffs = [
        abs(a - b)
        for r1, r2 in zip(on_1, on_2)
        for a, b in zip(r1["token_logprobs"], r2["token_logprobs"])
    ]

    del peft_model, base_model
    cleanup_cuda()

    # Fresh zero-B LoRA control
    base_for_noop = AutoModelForCausalLM.from_pretrained(
        str(base_dir), local_files_only=True, torch_dtype=torch.float16, device_map=device_map
    )
    noop_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    noop_model = get_peft_model(base_for_noop, noop_cfg)
    noop_on = get_target_logprobs(noop_model, tokenizer, subset)
    with noop_model.disable_adapter():
        noop_off = get_target_logprobs(noop_model, tokenizer, subset)
    noop_diffs = [
        abs(a - b)
        for r1, r2 in zip(noop_on, noop_off)
        for a, b in zip(r1["token_logprobs"], r2["token_logprobs"])
    ]
    del noop_model, base_for_noop
    cleanup_cuda()

    on_off_stats = summarize_diffs(on_off_diffs)
    repeat_stats = summarize_diffs(repeat_diffs)
    noop_stats = summarize_diffs(noop_diffs)
    floor = max(repeat_stats["max"], noop_stats["max"], 1e-6)
    passed = nonzero_layers == len(pairs) and len(pairs) > 0 and on_off_stats["median"] > floor * 10.0

    return {
        "status": "PASSED" if passed else "FAILED",
        "lora_pair_count": len(pairs),
        "nonzero_delta_layers": nonzero_layers,
        "delta_norm_stats": summarize_diffs(delta_norms),
        "on_off_logprob_diff": on_off_stats,
        "repeat_noise_diff": repeat_stats,
        "zero_b_noop_diff": noop_stats,
        "noise_floor": floor,
    }


def check_merge_parity(
    base_dir: Path,
    adapter_dir: Path,
    merged_dir: Path,
    dev_examples: List[Dict[str, Any]],
    sample_n: int = 32,
    median_tol: float = 0.01,
    p95_tol: float = 0.05,
) -> Dict[str, Any]:
    """
    Проверяет паритет склейки (merge parity):
    перезагруженная с диска плотная модель `artifacts/merged` должна давать практически те же
    log-вероятности и >= 99% совпадения greedy-токенов, что и связка `base + adapter`.
    """
    subset = dev_examples[:sample_n]
    device_map = {"": 0} if torch.cuda.is_available() else None

    tok_base = AutoTokenizer.from_pretrained(str(base_dir), local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_dir), local_files_only=True, torch_dtype=torch.float16, device_map=device_map
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
    adapter_on = get_target_logprobs(peft_model, tok_base, subset)
    del peft_model, base_model
    cleanup_cuda()

    tok_merged = AutoTokenizer.from_pretrained(str(merged_dir), local_files_only=True)
    merged_model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir), local_files_only=True, torch_dtype=torch.float16, device_map=device_map
    )
    merged_res = get_target_logprobs(merged_model, tok_merged, subset)
    del merged_model
    cleanup_cuda()

    diffs = []
    agree = 0
    total_toks = 0
    mismatches = []
    for r_on, r_m in zip(adapter_on, merged_res):
        for idx, (lp1, lp2, g1, g2) in enumerate(
            zip(r_on["token_logprobs"], r_m["token_logprobs"], r_on["greedy_tokens"], r_m["greedy_tokens"])
        ):
            d = abs(lp1 - lp2)
            diffs.append(d)
            total_toks += 1
            if g1 == g2:
                agree += 1
            elif len(mismatches) < 5:
                mismatches.append({"id": r_on["id"], "pos": idx, "on_tok": g1, "merged_tok": g2, "diff": d})

    stats = summarize_diffs(diffs)
    greedy_agreement = agree / max(total_toks, 1)
    passed = stats["median"] <= median_tol and stats["p95"] <= p95_tol and greedy_agreement >= 0.99

    return {
        "status": "PASSED" if passed else "FAILED",
        "median_tol": median_tol,
        "p95_tol": p95_tol,
        "logprob_abs_diff": stats,
        "greedy_token_agreement": round(greedy_agreement, 6),
        "first_mismatches": mismatches,
    }


def compute_macro_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    f1s = []
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(sum(f1s) / max(len(f1s), 1))


def paired_bootstrap_ci(
    a_correct: List[int],
    b_correct: List[int],
    n_resamples: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    a = np.array(a_correct, dtype=np.float64)
    b = np.array(b_correct, dtype=np.float64)
    n = len(a)
    diff = b - a
    obs = float(np.mean(diff))
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean_diff": obs,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "n": int(n),
        "n_resamples": n_resamples,
    }


def evaluate_artifact(
    name: str,
    model_dir: Path,
    examples: List[Dict[str, Any]],
    valid_intents: List[str],
    is_nf4: bool = False,
    max_new_tokens: int = 16,
    base_vocab_size: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    device_map = {"": 0} if torch.cuda.is_available() else None
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)

    load_kwargs: Dict[str, Any] = {"local_files_only": True, "device_map": device_map}
    if not is_nf4:
        load_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
    model.eval()

    if base_vocab_size is not None and len(tokenizer) != base_vocab_size:
        print(f"[WARN] Tokenizer length mismatch for {name}: {len(tokenizer)} != {base_vocab_size}", file=sys.stderr)

    device = next(model.parameters()).device
    valid_set = set(valid_intents)
    records: List[Dict[str, Any]] = []
    y_true, y_pred = [], []
    class_errors: Counter = Counter()

    with torch.inference_mode():
        for ex in examples:
            prompt_ids = extract_ids(
                tokenizer.apply_chat_template(
                    ex["messages"], tokenize=True, add_generation_prompt=True, return_dict=False
                )
            )
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = out_ids[0, len(prompt_ids) :].tolist()
            raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            pred = raw_text.strip()
            hit_eos = tokenizer.eos_token_id in new_tokens
            stop_reason = "eos" if hit_eos else "max_new_tokens"

            is_valid = pred in valid_set
            is_exact = pred == ex["expected"]
            y_true.append(ex["expected"])
            y_pred.append(pred if is_valid else "__INVALID__")
            if not is_exact:
                class_errors[ex["expected"]] += 1

            records.append(
                {
                    "artifact": name,
                    "id": ex["id"],
                    "expected": ex["expected"],
                    "raw_output": raw_text,
                    "parsed_pred": pred,
                    "valid_format": is_valid,
                    "exact_match": is_exact,
                    "stop_reason": stop_reason,
                    "new_token_ids": new_tokens,
                }
            )

    del model
    cleanup_cuda()

    n = len(records)
    em = sum(1 for r in records if r["exact_match"]) / max(n, 1)
    vf = sum(1 for r in records if r["valid_format"]) / max(n, 1)
    mf1 = compute_macro_f1(y_true, y_pred, valid_intents)

    summary = {
        "artifact": name,
        "model_dir": str(model_dir),
        "n": n,
        "exact_match": round(em, 6),
        "valid_format_rate": round(vf, 6),
        "macro_f1": round(mf1, 6),
        "errors_by_expected_class": dict(class_errors),
    }
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Soup training, merge parity, and NF4 export.")
    parser.add_argument(
        "--mode",
        choices=["base-dev", "adapter-activity", "merge-parity", "eval", "full"],
        required=True,
    )
    parser.add_argument("--base-dir", type=Path, default=Path("base_snapshot"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("artifacts/adapter"))
    parser.add_argument("--merged-dir", type=Path, default=Path("artifacts/merged"))
    parser.add_argument("--nf4-dir", type=Path, default=Path("artifacts/nf4"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--dev", type=Path, default=Path("data/dev.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = verify_inputs(args.manifest, args.dev, args.test)
    valid_intents = manifest["selected_intents"]
    dev_examples = load_jsonl(args.dev)
    tok_base = AutoTokenizer.from_pretrained(str(args.base_dir), local_files_only=True)
    base_vocab_size = len(tok_base)

    if args.mode == "base-dev":
        summary, recs = evaluate_artifact(
            "base_dev", args.base_dir, dev_examples, valid_intents, is_nf4=False,
            max_new_tokens=args.max_new_tokens, base_vocab_size=base_vocab_size
        )
        (args.out_dir / "baseline-dev.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.mode == "adapter-activity":
        res = check_adapter_activity(args.base_dir, args.adapter_dir, dev_examples)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        sys.exit(0 if res["status"] == "PASSED" else 2)

    if args.mode == "merge-parity":
        res = check_merge_parity(args.base_dir, args.adapter_dir, args.merged_dir, dev_examples)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        sys.exit(0 if res["status"] == "PASSED" else 3)

    # eval or full
    test_examples = load_jsonl(args.test)
    report: Dict[str, Any] = {"mode": args.mode}
    all_records: List[Dict[str, Any]] = []

    if args.mode == "full":
        act = check_adapter_activity(args.base_dir, args.adapter_dir, dev_examples)
        mp = check_merge_parity(args.base_dir, args.adapter_dir, args.merged_dir, dev_examples)
        report["adapter_activity"] = act
        report["merge_parity"] = mp

    base_sum, base_recs = evaluate_artifact(
        "base", args.base_dir, test_examples, valid_intents, is_nf4=False,
        max_new_tokens=args.max_new_tokens, base_vocab_size=base_vocab_size
    )
    merged_sum, merged_recs = evaluate_artifact(
        "merged", args.merged_dir, test_examples, valid_intents, is_nf4=False,
        max_new_tokens=args.max_new_tokens, base_vocab_size=base_vocab_size
    )
    nf4_sum, nf4_recs = evaluate_artifact(
        "nf4", args.nf4_dir, test_examples, valid_intents, is_nf4=True,
        max_new_tokens=args.max_new_tokens, base_vocab_size=base_vocab_size
    )
    all_records.extend(base_recs + merged_recs + nf4_recs)

    base_em = [int(r["exact_match"]) for r in base_recs]
    merged_em = [int(r["exact_match"]) for r in merged_recs]
    nf4_em = [int(r["exact_match"]) for r in nf4_recs]

    report["artifacts"] = {"base": base_sum, "merged": merged_sum, "nf4": nf4_sum}
    report["paired_bootstrap_95ci"] = {
        "merged_minus_base": paired_bootstrap_ci(base_em, merged_em),
        "nf4_minus_merged": paired_bootstrap_ci(merged_em, nf4_em),
    }

    (args.out_dir / "verify-summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.out_dir / "verify-per-example.jsonl").open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== VERIFY SUMMARY ===")
    for k in ["base", "merged", "nf4"]:
        s = report["artifacts"][k]
        print(f"{k:<10} n={s['n']}  EM={s['exact_match']:.4f}  valid={s['valid_format_rate']:.4f}  macro_F1={s['macro_f1']:.4f}")
    if args.mode == "full":
        act_status = report["adapter_activity"]["status"]
        mp_status = report["merge_parity"]["status"]
        print(f"adapter_activity: {act_status}")
        print(f"merge_parity:     {mp_status}")
        # Не даём --mode full завершиться с exit 0, если провалена активность адаптера или паритет склейки
        if act_status != "PASSED":
            sys.exit(2)
        if mp_status != "PASSED":
            sys.exit(3)


if __name__ == "__main__":
    main()
