#!/usr/bin/env python3
"""
VRAM Budget Calculator for LLM Training (LoRA/QLoRA), Merge, and Quantization.
Designed for Soup Internship Take-Home Task (Part 1).

Covers:
- Base weights (FP16 / BF16 vs NF4)
- LoRA adapter parameters (exact breakdown for q, k, v, o, gate, up, down)
- LoRA gradients (FP32 vs FP16)
- Optimizer states (Standard AdamW FP32 vs 8-bit / Paged AdamW)
- Activations (Full vs Gradient Checkpointing)
- Logits & Cross-Entropy peak (Vocab size impact)
- Merge phase (GPU in-place vs CPU offloaded)
- Quantization phase
- Unit conversions: Bytes, MB (10^6), MiB (2^20)
"""

import json
import os
import argparse
from typing import Dict, Any

# Default config for Qwen/Qwen2.5-1.5B-Instruct if no local config.json is passed
DEFAULT_QWEN_1_5B_CONFIG = {
    "model_type": "qwen2",
    "hidden_size": 1536,
    "intermediate_size": 8960,
    "num_hidden_layers": 28,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
    "tie_word_embeddings": True
}

def bytes_to_human(b: int) -> Dict[str, Any]:
    return {
        "bytes": int(b),
        "MB": round(b / 1_000_000, 2),
        "MiB": round(b / (1024 * 1024), 2),
        "GB": round(b / 1_000_000_000, 3),
        "GiB": round(b / (1024 ** 3), 3)
    }

def calculate_lora_params(config: Dict[str, Any], r: int = 16, target_modules=None) -> Dict[str, Any]:
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    h = config["hidden_size"]
    inter = config["intermediate_size"]
    layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = h // num_heads
    kv_dim = num_kv_heads * head_dim  # For Grouped-Query Attention (GQA)

    breakdown = {}
    per_layer_params = 0

    # Attention projections
    # q_proj: (h, h) -> LoRA A (h, r) + B (r, h) = r * (h + h)
    if "q_proj" in target_modules:
        p = r * (h + h)
        breakdown["q_proj"] = p
        per_layer_params += p

    # k_proj: (h, kv_dim) -> LoRA A (h, r) + B (r, kv_dim) = r * (h + kv_dim)
    if "k_proj" in target_modules:
        p = r * (h + kv_dim)
        breakdown["k_proj"] = p
        per_layer_params += p

    # v_proj: (h, kv_dim) -> LoRA A (h, r) + B (r, kv_dim) = r * (h + kv_dim)
    if "v_proj" in target_modules:
        p = r * (h + kv_dim)
        breakdown["v_proj"] = p
        per_layer_params += p

    # o_proj: (h, h) -> LoRA A (h, r) + B (r, h) = r * (h + h)
    if "o_proj" in target_modules:
        p = r * (h + h)
        breakdown["o_proj"] = p
        per_layer_params += p

    # MLP projections
    # gate_proj: (h, inter) -> LoRA A (h, r) + B (r, inter) = r * (h + inter)
    if "gate_proj" in target_modules:
        p = r * (h + inter)
        breakdown["gate_proj"] = p
        per_layer_params += p

    # up_proj: (h, inter) -> LoRA A (h, r) + B (r, inter) = r * (h + inter)
    if "up_proj" in target_modules:
        p = r * (h + inter)
        breakdown["up_proj"] = p
        per_layer_params += p

    # down_proj: (inter, h) -> LoRA A (inter, r) + B (r, h) = r * (inter + h)
    if "down_proj" in target_modules:
        p = r * (inter + h)
        breakdown["down_proj"] = p
        per_layer_params += p

    total_lora_params = per_layer_params * layers

    return {
        "per_layer_breakdown": breakdown,
        "per_layer_params": per_layer_params,
        "total_lora_params": total_lora_params,
        "layers": layers,
        "rank": r
    }

def calculate_vram_budget(
    config: Dict[str, Any],
    r: int = 16,
    batch_size: int = 1,
    seq_len: int = 2048,
    base_dtype: str = "fp16",         # "fp16", "bf16", "int4" (QLoRA)
    lora_dtype: str = "fp32",         # PEFT default trainable param dtype: fp32 or fp16
    optimizer_type: str = "adamw_fp32", # "adamw_fp32", "adamw_8bit"
    gradient_checkpointing: bool = True
) -> Dict[str, Any]:

    # 1. Total Base Parameters calculation
    h = config["hidden_size"]
    inter = config["intermediate_size"]
    layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = h // num_heads
    kv_dim = num_kv_heads * head_dim
    v = config["vocab_size"]

    # Embedding weights
    embed_params = v * h
    # Layer weights: attn (q + k + v + o) + mlp (gate + up + down) + 2 layer norms (input + post_attn)
    attn_params = (h * h) + (h * kv_dim) + (h * kv_dim) + (h * h)
    mlp_params = (h * inter) + (h * inter) + (inter * h)
    ln_params = 2 * h
    layer_params = attn_params + mlp_params + ln_params
    # Final norm + lm_head (if not tied)
    final_ln_params = h
    lm_head_params = 0 if config.get("tie_word_embeddings", False) else (v * h)

    total_base_params = embed_params + (layer_params * layers) + final_ln_params + lm_head_params

    # 2. Base Model Memory
    bytes_per_param_base = 0.5 if base_dtype == "int4" else 2.0  # FP16/BF16 = 2 bytes, INT4/NF4 = 0.5 bytes
    # For INT4 (bitsandbytes NF4), add ~15-20% overhead for quantization metadata, scales, double quant
    base_weight_bytes = int(total_base_params * bytes_per_param_base * (1.18 if base_dtype == "int4" else 1.0))

    # 3. LoRA Memory
    lora_info = calculate_lora_params(config, r=r)
    total_lora_params = lora_info["total_lora_params"]
    bytes_per_param_lora = 4.0 if lora_dtype == "fp32" else 2.0

    lora_weight_bytes = int(total_lora_params * bytes_per_param_lora)
    # Gradients match lora parameter dtype (typically FP32 or FP16)
    lora_grad_bytes = int(total_lora_params * bytes_per_param_lora)

    # 4. Optimizer Memory
    # Standard AdamW stores 2 states (exp_avg, exp_avg_sq) in FP32 (8 bytes/param) + copy of master weights if low-prec
    if optimizer_type == "adamw_fp32":
        opt_bytes_per_param = 8.0
    elif optimizer_type == "adamw_8bit":
        opt_bytes_per_param = 2.0
    else:
        opt_bytes_per_param = 8.0
    optimizer_bytes = int(total_lora_params * opt_bytes_per_param)

    # 5. Activations Memory
    # Formula for activations with and without gradient checkpointing:
    # Per layer activation memory (without checkpointing):
    # Attn (Q, K, V, Softmax scores, Attn output) + MLP (Gate, Up, SwiGLU, Down)
    # With gradient checkpointing: only input activations per layer are stored,
    # recomputing activations layer-by-layer during backward pass.
    # Checkpointed activations ≈ b * s * h * 2 bytes * layers (layer inputs) + 1 block active forward/backward peak
    if gradient_checkpointing:
        # Layer inputs checkpointed: b * s * h * 2 bytes * layers
        checkpointed_inputs = batch_size * seq_len * h * 2 * layers
        # Single layer recomputation peak (MLP SwiGLU intermediate: b * s * inter * 2 * 2 + attn)
        peak_layer_activation = batch_size * seq_len * (h + inter * 3) * 2 * 2
        activation_bytes = int(checkpointed_inputs + peak_layer_activation)
    else:
        # Full activations: ~ 34 * b * s * h * layers bytes
        activation_bytes = int(34 * batch_size * seq_len * h * layers * 2)

    # 6. Logits & Cross-Entropy Peak
    # Logits tensor: [batch_size, seq_len, vocab_size] in FP32 (cross-entropy requires float32 for numerical stability)
    logits_bytes = int(batch_size * seq_len * v * 4)  # 4 bytes for float32

    # 7. Total Training VRAM (Allocated Tensors)
    # Peak during forward/backward happens when activations + logits + weights + grads + opt are all in memory
    training_allocated_peak_bytes = (
        base_weight_bytes +
        lora_weight_bytes +
        lora_grad_bytes +
        optimizer_bytes +
        activation_bytes +
        logits_bytes
    )

    # 8. Merge Phase
    # Case A: GPU in-place merge: Base model loaded in FP16 (2 bytes) + LoRA in FP32/16 (delta added to base in-place)
    # Base in FP16 for export/merge:
    base_fp16_bytes = int(total_base_params * 2.0)
    gpu_merge_inplace_bytes = base_fp16_bytes + lora_weight_bytes
    # Case B: If temporary unmerged + merged tensors exist simultaneously:
    gpu_merge_peak_with_copy_bytes = base_fp16_bytes * 2 + lora_weight_bytes

    # 9. Quantization Phase (e.g. 4-bit AWQ or bitsandbytes export)
    # Base FP16 model (3.1 GB) + quantized target buffer (0.8 GB) + calibration batch tensors
    quantization_peak_bytes = int(base_fp16_bytes + (total_base_params * 0.5) + (batch_size * seq_len * h * 4))

    return {
        "metadata": {
            "model_type": config.get("model_type", "qwen2"),
            "total_base_parameters": total_base_params,
            "total_lora_parameters": total_lora_params,
            "lora_rank": r,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "base_dtype": base_dtype,
            "lora_dtype": lora_dtype,
            "optimizer_type": optimizer_type,
            "gradient_checkpointing": gradient_checkpointing,
            "vocab_size": v
        },
        "training_breakdown": {
            "base_model_weights": bytes_to_human(base_weight_bytes),
            "lora_adapter_weights": bytes_to_human(lora_weight_bytes),
            "lora_gradients": bytes_to_human(lora_grad_bytes),
            "optimizer_states": bytes_to_human(optimizer_bytes),
            "activations": bytes_to_human(activation_bytes),
            "logits_cross_entropy_peak": bytes_to_human(logits_bytes),
            "total_training_allocated_peak": bytes_to_human(training_allocated_peak_bytes)
        },
        "merge_breakdown": {
            "base_fp16_weights": bytes_to_human(base_fp16_bytes),
            "gpu_merge_inplace_peak": bytes_to_human(gpu_merge_inplace_bytes),
            "gpu_merge_peak_with_copy": bytes_to_human(gpu_merge_peak_with_copy_bytes)
        },
        "quantization_breakdown": {
            "quantization_peak": bytes_to_human(quantization_peak_bytes)
        },
        "notes": {
            "cuda_allocator_overhead": "PyTorch Caching Allocator reserves blocks in advance (memory_reserved > memory_allocated). Usually adds 500MB - 1.5GB to nvidia-smi.",
            "cuda_context_overhead": "CUDA runtime context and cuBLAS workspace take ~400MB - 700MB VRAM before any tensor is allocated.",
            "qwen_logits_trap": f"Qwen vocab size is {v}. At seq_len={seq_len}, logits alone take {bytes_to_human(logits_bytes)['MiB']} MiB in FP32."
        }
    }

def print_report(res: Dict[str, Any]):
    meta = res["metadata"]
    tb = res["training_breakdown"]
    mb = res["merge_breakdown"]
    qb = res["quantization_breakdown"]

    print("=" * 75)
    print(" VRAM BUDGET REPORT (Soup Take-Home Task - Part 1)")
    print("=" * 75)
    print(f"Model: {meta['model_type']} (~{meta['total_base_parameters'] / 1e9:.2f}B params)")
    print(f"LoRA: rank={meta['lora_rank']}, params={meta['total_lora_parameters']:,} ({meta['total_lora_parameters'] / meta['total_base_parameters'] * 100:.2f}%)")
    print(f"Config: Batch={meta['batch_size']}, SeqLen={meta['seq_len']}, BaseDtype={meta['base_dtype']}, LoRADtype={meta['lora_dtype']}, Opt={meta['optimizer_type']}")
    print("-" * 75)
    print(f"{'Component':<32} | {'Bytes':<15} | {'MiB':<10} | {'GB (10^9)':<10}")
    print("-" * 75)
    for k, v in tb.items():
        print(f"{k:<32} | {v['bytes']:<15} | {v['MiB']:<10} | {v['GB']:<10}")
    print("-" * 75)
    print("MERGE & QUANTIZATION PEAKS:")
    print(f"{'Base FP16 Weights':<32} | {mb['base_fp16_weights']['bytes']:<15} | {mb['base_fp16_weights']['MiB']:<10} | {mb['base_fp16_weights']['GB']:<10}")
    print(f"{'Merge In-Place (GPU)':<32} | {mb['gpu_merge_inplace_peak']['bytes']:<15} | {mb['gpu_merge_inplace_peak']['MiB']:<10} | {mb['gpu_merge_inplace_peak']['GB']:<10}")
    print(f"{'Merge With Duplicate (GPU)':<32} | {mb['gpu_merge_peak_with_copy']['bytes']:<15} | {mb['gpu_merge_peak_with_copy']['MiB']:<10} | {mb['gpu_merge_peak_with_copy']['GB']:<10}")
    print(f"{'Quantization Peak':<32} | {qb['quantization_peak']['bytes']:<15} | {qb['quantization_peak']['MiB']:<10} | {qb['quantization_peak']['GB']:<10}")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VRAM Budget Calculator")
    parser.add_argument("--config", type=str, help="Path to model config.json")
    parser.add_argument("--r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--base_dtype", type=str, default="int4", choices=["fp16", "bf16", "int4"], help="Base model dtype")
    parser.add_argument("--lora_dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="LoRA parameter dtype")
    parser.add_argument("--optimizer", type=str, default="adamw_fp32", choices=["adamw_fp32", "adamw_8bit"], help="Optimizer type")
    parser.add_argument("--no_grad_checkpoint", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--out", type=str, default="vram_budget.json", help="Output JSON path")

    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = DEFAULT_QWEN_1_5B_CONFIG

    result = calculate_vram_budget(
        config=cfg,
        r=args.r,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        base_dtype=args.base_dtype,
        lora_dtype=args.lora_dtype,
        optimizer_type=args.optimizer,
        gradient_checkpointing=not args.no_grad_checkpoint
    )

    print_report(result)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Calculation saved to: {args.out}")
