#!/usr/bin/env python3
"""
Verification Script for Soup Internship Take-Home Task (Part 2).
Author: Candidate (Belli)

Answers:
1. "Why doesn't a falling loss prove learning or an active adapter?"
2. Proves the model actually trained vs finished without meaningfully changing.
3. Compares three artifacts: Base, Merged, and 4-bit Quantized.
4. Explains what the check would NOT detect.

Components:
1. Weight Difference (Base vs Merged: L2 norm, max abs, % modified params)
2. Adapter Integrity (LoRA B matrix non-zero test, target modules audit)
3. Logits Comparison (KL Divergence, Cosine Similarity, Top-1 Token Agreement)
4. Holdout Evaluation (Masked-prompt Perplexity, Deterministic Greedy outputs)
5. Quantization Fidelity (Merged vs 4-bit degradation analysis)
"""

import os
import sys
import json
import argparse
import math
from typing import Dict, List, Any, Tuple

# Try importing torch/transformers, provide friendly error if running outside ML environment
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftConfig
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =====================================================================
# THEORETICAL FOUNDATION: What the check would NOT detect
# =====================================================================
UNDETECTED_LIMITATIONS = [
    {
        "limitation": "Data Contamination / Train-Test Leakage",
        "reason": "If holdout samples or evaluation benchmarks were accidentally present in the training set, the model will score high perplexity/accuracy via memorization rather than generalized reasoning."
    },
    {
        "limitation": "Catastrophic Forgetting on General Capabilities",
        "reason": "Evaluating only on in-domain instruction tasks does not detect degradation in general knowledge, coding, multi-turn dialogue, or multilingual capabilities that were not part of the holdout set."
    },
    {
        "limitation": "Hallucinations & Factuality Drift",
        "reason": "Top-1 token agreement and low perplexity measure distribution fit, not factual truth. The model can output fluent, confident, but factually false completions."
    },
    {
        "limitation": "Calibration & Adversarial Robustness",
        "reason": "Greedy decoding on clean prompts does not detect overconfidence, susceptibility to prompt injections, or degeneration on long-horizon generation loops."
    },
    {
        "limitation": "Silent Masking Artifacts (False Positive Learning)",
        "reason": "If prompt masking in training was inverted (trained on user prompts instead of assistant responses), weights and logits will change significantly, passing weight-diff tests while learning the wrong distribution."
    }
]

WHY_FALLING_LOSS_PROVES_NOTHING = """
WHY A FALLING LOSS DOES NOT PROVE MEANINGFUL LEARNING OR AN ACTIVE ADAPTER:
1. Unmasked Prompt Fitting: If the loss mask fails to mask user prompts with -100, the model
   minimizes loss simply by memorizing common template tokens and user prompts, without learning
   to generate good responses.
2. Mode Collapse / Repetition: A model predicting uniform punctuation, whitespace, or frequent
   filler tokens ('The', 'I') can show a rapidly falling cross-entropy loss while degrading generation.
3. LoRA B Initialization & Dead Adapters: LoRA initializes matrix B to 0 (so delta_W = B * A = 0).
   If learning rate is too low, gradients vanish, or optimizer state fails to apply, loss can drift
   downward slightly due to learning-rate scheduler decay on base model norms or bias layers,
   while B remains ~0 and the adapter never activates.
4. Overfitting to Spurious Correlations: Loss on the training set drops to near-zero when memorizing
   1k examples, while validation loss diverges (catastrophic overfitting).
"""


def check_adapter_integrity(adapter_path: str) -> Dict[str, Any]:
    """
    Audits the LoRA adapter before merging.
    Crucial check: In LoRA, delta_W = B * A, where B is initialized to 0.
    If training did not update B, B remains zero, meaning a 'phantom run'.
    """
    if not os.path.exists(adapter_path):
        return {"error": f"Adapter path not found: {adapter_path}"}

    report = {"path": adapter_path, "status": "PASSED", "modules_checked": {}}

    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        report["target_modules"] = cfg.get("target_modules", [])
        report["r"] = cfg.get("r", None)
        report["lora_alpha"] = cfg.get("lora_alpha", None)

    # Inspect weights (safetensors or bin)
    weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    is_safetensors = os.path.exists(weights_path)
    if not is_safetensors:
        weights_path = os.path.join(adapter_path, "adapter_model.bin")

    if not os.path.exists(weights_path):
        report["status"] = "WARNING"
        report["message"] = "No weight file found in adapter directory."
        return report

    if HAS_TORCH:
        if is_safetensors:
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location="cpu")

        zero_b_matrices = []
        b_norms = []
        a_norms = []

        for name, param in state_dict.items():
            norm_val = torch.norm(param.float()).item()
            if "lora_B" in name:
                b_norms.append(norm_val)
                if norm_val < 1e-7:
                    zero_b_matrices.append(name)
            elif "lora_A" in name:
                a_norms.append(norm_val)

        report["total_adapter_tensors"] = len(state_dict)
        report["avg_lora_B_norm"] = float(sum(b_norms) / max(len(b_norms), 1))
        report["avg_lora_A_norm"] = float(sum(a_norms) / max(len(a_norms), 1))

        if zero_b_matrices:
            report["status"] = "FAILED"
            report["error"] = f"Dead adapter detected! {len(zero_b_matrices)} LoRA B matrices have zero norm. Training did not modify weights."
            report["zero_b_matrices"] = zero_b_matrices
        else:
            report["status"] = "PASSED"
            report["notes"] = "All LoRA B matrices have non-zero norms; weights were updated."

    return report


def compare_weights(base_model, merged_model) -> Dict[str, Any]:
    """
    Compares Base vs Merged weights tensor by tensor.
    Computes Frobenius/L2 norm, relative delta, max absolute difference,
    and percentage of modified parameters.
    """
    base_dict = base_model.state_dict()
    merged_dict = merged_model.state_dict()

    total_elements = 0
    changed_elements = 0
    layer_summaries = []
    total_l2_diff_sq = 0.0
    total_base_l2_sq = 0.0
    max_global_abs_diff = 0.0

    for name in base_dict.keys():
        if name not in merged_dict:
            continue

        w_base = base_dict[name].float()
        w_merged = merged_dict[name].float()

        num_el = w_base.numel()
        total_elements += num_el

        diff = w_merged - w_base
        abs_diff = torch.abs(diff)
        max_abs = torch.max(abs_diff).item()
        if max_abs > max_global_abs_diff:
            max_global_abs_diff = max_abs

        changed = torch.count_nonzero(abs_diff > 1e-6).item()
        changed_elements += changed

        l2_diff = torch.norm(diff).item()
        base_l2 = torch.norm(w_base).item()

        total_l2_diff_sq += l2_diff ** 2
        total_base_l2_sq += base_l2 ** 2

        # Record if linear projection layers changed
        if any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
            layer_summaries.append({
                "layer": name,
                "l2_diff": round(l2_diff, 5),
                "rel_l2_diff": round(l2_diff / (base_l2 + 1e-12), 6),
                "max_abs_diff": round(max_abs, 6),
                "changed_pct": round((changed / num_el) * 100, 2)
            })

    overall_l2_diff = math.sqrt(total_l2_diff_sq)
    overall_base_l2 = math.sqrt(total_base_l2_sq)
    overall_rel_diff = overall_l2_diff / (overall_base_l2 + 1e-12)

    return {
        "total_parameters_checked": total_elements,
        "changed_parameters_count": changed_elements,
        "changed_parameters_pct": round((changed_elements / max(total_elements, 1)) * 100, 4),
        "overall_l2_diff": round(overall_l2_diff, 4),
        "overall_relative_diff": round(overall_rel_diff, 6),
        "max_absolute_diff": round(max_global_abs_diff, 6),
        "sample_target_layers": layer_summaries[:7]  # First block sample
    }


def compare_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> Dict[str, float]:
    """
    Computes statistical similarity between logits tensors of identical inputs:
    - KL Divergence: D_KL(P_a || P_b)
    - Cosine Similarity across vocabulary dimension
    - Top-1 Token Agreement rate (% of positions where argmax matches)
    - Mean absolute logit delta
    """
    # Shapes: [batch_size, seq_len, vocab_size]
    flat_a = logits_a.view(-1, logits_a.size(-1)).float()
    flat_b = logits_b.view(-1, logits_b.size(-1)).float()

    # Probabilities
    log_p_a = F.log_softmax(flat_a, dim=-1)
    p_b = F.softmax(flat_b, dim=-1)
    kl_div = F.kl_div(log_p_a, p_b, reduction="batchmean").item()

    # Cosine similarity
    cos_sim = F.cosine_similarity(flat_a, flat_b, dim=-1).mean().item()

    # Top-1 Agreement
    top1_a = torch.argmax(flat_a, dim=-1)
    top1_b = torch.argmax(flat_b, dim=-1)
    agreement = (top1_a == top1_b).float().mean().item() * 100.0

    # Mean absolute difference
    mean_abs_diff = torch.abs(flat_a - flat_b).mean().item()

    return {
        "kl_divergence": round(kl_div, 5),
        "cosine_similarity": round(cos_sim, 5),
        "top1_agreement_pct": round(agreement, 2),
        "mean_abs_logit_diff": round(mean_abs_diff, 5)
    }


def compute_masked_perplexity(model, tokenizer, test_prompts: List[Dict[str, str]], device: str = "cuda") -> Tuple[float, List[Dict[str, str]]]:
    """
    Computes perplexity strictly on response tokens (prompt tokens masked with -100).
    Also generates deterministic greedy output (do_sample=False, temperature=0.0).
    """
    model.eval()
    total_loss = 0.0
    total_target_tokens = 0
    generations = []

    for item in test_prompts:
        prompt = item["prompt"]
        expected = item.get("expected", "")

        # Format full prompt with assistant answer for perplexity
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": expected}
        ] if expected else [{"role": "user", "content": prompt}]

        if hasattr(tokenizer, "apply_chat_template"):
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_only = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        else:
            full_text = f"User: {prompt}\nAssistant: {expected}"
            prompt_only = f"User: {prompt}\nAssistant:"

        input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
        prompt_ids = tokenizer(prompt_only, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]

        # Labels: mask prompt tokens with -100
        labels = input_ids.clone()
        if prompt_len < labels.shape[1]:
            labels[:, :prompt_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            target_tokens = (labels != -100).sum().item()
            if target_tokens > 0:
                total_loss += outputs.loss.item() * target_tokens
                total_target_tokens += target_tokens

            # Deterministic greedy generation
            gen_ids = model.generate(
                prompt_ids,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            completion = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True).strip()
            generations.append({
                "prompt": prompt,
                "completion": completion
            })

    avg_loss = total_loss / max(total_target_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return round(ppl, 3), generations


def run_verification(
    base_path: str,
    merged_path: str,
    quant_path: str,
    adapter_path: str = None,
    holdout_path: str = None,
    device: str = None,
    output_json: str = "verification_report.json"
) -> Dict[str, Any]:
    """
    Main verification pipeline comparing Base, Merged, and 4-bit models.
    """
    if device is None:
        device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"

    print("=" * 75)
    print(" STARTING VERIFICATION: PROVE THE MODEL ACTUALLY TRAINED (Part 2)")
    print("=" * 75)

    report = {
        "status": "RUNNING",
        "why_falling_loss_proves_nothing": WHY_FALLING_LOSS_PROVES_NOTHING.strip(),
        "undetected_limitations": UNDETECTED_LIMITATIONS
    }

    # 1. Adapter Integrity Check
    if adapter_path and os.path.exists(adapter_path):
        print("\n[1/5] Checking Adapter Integrity (LoRA matrices & non-zero B check)...")
        report["adapter_integrity"] = check_adapter_integrity(adapter_path)
    else:
        report["adapter_integrity"] = {"note": "No adapter path provided or path does not exist."}

    # Standard Holdout Test Prompts
    test_prompts = [
        {"prompt": "Explain the difference between a process and a thread in one sentence.", "expected": "A process has its own memory space, while threads share memory within the same process."},
        {"prompt": "Write a Python function to check if a number is prime.", "expected": "def is_prime(n):\n    if n <= 1: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True"},
        {"prompt": "What is the capital of France?", "expected": "The capital of France is Paris."}
    ]
    if holdout_path and os.path.exists(holdout_path):
        with open(holdout_path, "r", encoding="utf-8") as f:
            test_prompts = json.load(f)

    # 2. Load Tokenizer
    print("\n[2/5] Loading Tokenizer (Strict parity across all models)...")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)

    # 3. Load Models
    print("\n[3/5] Loading Base Model & Merged Model for Weight & Logits Diff...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    # Compare Weights
    print(" -> Executing Weight Difference Analysis...")
    report["weight_difference_base_vs_merged"] = compare_weights(base_model, merged_model)

    # Compare Logits (Base vs Merged)
    print(" -> Extracting logits for alignment & KL Divergence...")
    sample_text = test_prompts[0]["prompt"]
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)

    with torch.no_grad():
        base_logits = base_model(**inputs).logits
        merged_logits = merged_model(**inputs).logits

    report["logits_base_vs_merged"] = compare_logits(base_logits, merged_logits)

    # Holdout Evaluation for Base & Merged
    print(" -> Computing Holdout Perplexity & Deterministic Generations...")
    base_ppl, base_gens = compute_masked_perplexity(base_model, tokenizer, test_prompts, device=device)
    merged_ppl, merged_gens = compute_masked_perplexity(merged_model, tokenizer, test_prompts, device=device)

    report["holdout_eval"] = {
        "base_perplexity": base_ppl,
        "merged_perplexity": merged_ppl,
        "sample_generations": [
            {"prompt": test_prompts[0]["prompt"], "base": base_gens[0]["completion"], "merged": merged_gens[0]["completion"]}
        ]
    }

    # Free memory before loading 4-bit model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Quantization Fidelity (Merged vs 4-bit)
    print("\n[4/5] Loading 4-bit Quantized Model & Comparing with Merged...")
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        quant_model = AutoModelForCausalLM.from_pretrained(
            quant_path,
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=True
        )

        with torch.no_grad():
            quant_logits = quant_model(**inputs).logits

        report["quantization_fidelity_merged_vs_4bit"] = compare_logits(merged_logits, quant_logits)
        quant_ppl, quant_gens = compute_masked_perplexity(quant_model, tokenizer, test_prompts, device=device)
        report["holdout_eval"]["quant_4bit_perplexity"] = quant_ppl
        report["holdout_eval"]["sample_generations"][0]["quant_4bit"] = quant_gens[0]["completion"]

        del quant_model
    except Exception as e:
        report["quantization_fidelity_merged_vs_4bit"] = {"error": str(e)}

    del merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Final Summary Verdict
    print("\n[5/5] Generating Final Verification Verdict...")
    weight_diff = report["weight_difference_base_vs_merged"]
    logits_diff = report["logits_base_vs_merged"]

    is_trained = (
        weight_diff["changed_parameters_pct"] > 0.05 and
        logits_diff["top1_agreement_pct"] < 99.5 and
        logits_diff["kl_divergence"] > 0.001
    )

    report["verdict"] = {
        "model_actually_trained": is_trained,
        "summary": "Training successfully modified parameters and altered token output distribution." if is_trained else "WARNING: Model outputs and weights show negligible change; possible failed run or dead adapter."
    }
    report["status"] = "COMPLETED"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Verification finished! Full report saved to: {output_json}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 2 Verification Script")
    parser.add_argument("--base", type=str, default="./base_model", help="Path to base model")
    parser.add_argument("--merged", type=str, default="./merged_model", help="Path to merged model")
    parser.add_argument("--quant", type=str, default="./quant_model", help="Path to 4-bit model")
    parser.add_argument("--adapter", type=str, default="./adapter", help="Path to LoRA adapter")
    parser.add_argument("--holdout", type=str, default=None, help="Path to holdout test prompts JSON")
    parser.add_argument("--out", type=str, default="verification_report.json", help="Output JSON path")
    parser.add_argument("--explain_only", action="store_true", help="Print limitations and exit")

    args = parser.parse_args()

    if args.explain_only or not HAS_TORCH:
        print("=" * 75)
        print(" VERIFICATION SCRIPT ARCHITECTURE & THEORETICAL FOUNDATION")
        print("=" * 75)
        print(WHY_FALLING_LOSS_PROVES_NOTHING)
        print("\n" + "=" * 75)
        print("WHAT THIS VERIFICATION SCRIPT WOULD NOT DETECT:")
        print("=" * 75)
        for idx, item in enumerate(UNDETECTED_LIMITATIONS, 1):
            print(f"{idx}. {item['limitation']}:\n   -> {item['reason']}\n")
        sys.exit(0)

    run_verification(
        base_path=args.base,
        merged_path=args.merged,
        quant_path=args.quant,
        adapter_path=args.adapter,
        holdout_path=args.holdout,
        output_json=args.out
    )
