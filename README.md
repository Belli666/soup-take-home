# Soup Take-Home Task — AI Engineer Internship Journal
**Author:** Candidate (Belli)  
**Target Hardware:** Google Colab Free Tier (NVIDIA T4 16 GB VRAM)  
**Target Stack:** `soup-cli[train]` / PyTorch / PEFT / Transformers  
**Target Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Dataset:** Open Instruction Dataset (1–2k examples)  

---

## Task Progress & Checklist

- [x] **Setup & Workspace Isolation**: Created isolated directory `soupTest/` (keeping production bot `MK-666` untouched).
- [x] **AI Collaboration & Tool Usage Log**: Logged Antigravity (Gemini) collaboration notes, critiques accepted/rejected, and independent verifications (required by ТЗ).
- [x] **Part 1: Memory Budget (Pre-training)**:
  - [x] Built reproducible `vram_calculator.py` parsing exact architecture specs.
  - [x] Exact LoRA parameter derivation ($r=16$, $18,464,768$ params across all 7 linear projections).
  - [x] Calculated memory for weights, LoRA gradients, AdamW optimizer states, activations, and logits peak.
  - [x] Calculated Merge and Quantization peaks (GPU in-place vs duplicate vs CPU).
  - [x] Saved budget artifact to `vram_budget.json`.
- [ ] **Part 1 (Post-run)**: Capture actual `nvidia-smi` and PyTorch memory peaks on Colab T4, calculate gap (`nvidia-smi` vs `memory_reserved` vs `memory_allocated`).
- [x] **Part 2: Prove the Model Actually Trained**:
  - [x] Built modular verification script `verification.py` across 5 pillars (weights, adapter, logits, holdout, quantization).
  - [x] Documented the 4 fundamental reasons why falling loss is deceptive.
  - [x] Formulated and documented what the verification script CANNOT detect (leakage, catastrophic forgetting, hallucinations, adversarial shifts).
  - [ ] Run verification on Colab T4 against actual saved artifacts (Base vs Merged vs 4-bit) and capture `verification_report.json`.
- [ ] **Part 3: Silent Failures Investigation**:
  - [ ] Tokenizer & Chat template silent issues.
  - [ ] Loss masking failure modes (masking prompt vs response).
  - [ ] LoRA merge numerical drift and 4-bit quantization collapse.
  - [ ] Test Soup built-in checks (`data doctor`, `pre-flight`, `soup ship`).
- [ ] **Part 4: Verdict (SHIP / DON'T SHIP)** & 2-page final report.

---

## AI Tools Usage Log (Required by ТЗ)

> *T3 Requirement: "Note where you used them, which suggestions you accepted, and what you verified, changed, or redid yourself."*

* **Session 1 (Initial Setup & VRAM Budget Calculation)**:
  * **Initial Calculation & Review (Antigravity / Gemini)**: Proposed baseline formulas for Qwen2.5-1.5B on T4 and refined edge cases:
    1. *Accepted*: LoRA parameters ($r=16$, all-linear) confirmed at exactly $18,464,768$.
    2. *Accepted & Implemented*: Dtype verification — do not assume FP16 for LoRA adapters; in PEFT, trainable adapter weights and gradients default to FP32 unless explicitly cast.
    3. *Accepted & Implemented*: Logits spike is not eliminated by Gradient Checkpointing; vocab size of 151,936 produces a ~1.19 GiB FP32 logits tensor during cross-entropy loss computation.
    4. *Accepted & Implemented*: Merge memory model distinguishes between GPU in-place addition (~3.01 GiB peak) and temporary duplicated buffers (~5.96 GiB peak).
    5. *Accepted & Implemented*: Explicit separation of `torch.cuda.memory_allocated()`, `torch.cuda.memory_reserved()`, and `nvidia-smi`.
    6. *Built*: Created reproducible Python calculator (`vram_calculator.py`) without hardcoded magic numbers.

---

## Part 1: VRAM Budget Summary (Qwen2.5-1.5B-Instruct)

### 1. Model & LoRA Parameters
* **Base Architecture**: 28 layers, hidden size $h=1536$, intermediate size $8960$, 12 query heads, 2 KV heads (GQA), vocab size $151,936$.
* **Total Base Parameters**: $\approx 1,543,656,960$ ($1.54\text{B}$).
* **LoRA Parameters ($r=16$)**:
  * $W_q, W_o$: $2 \times [16 \times (1536 + 1536)] = 98,304$ per layer.
  * $W_k, W_v$: $2 \times [16 \times (1536 + 256)] = 57,344$ per layer.
  * $W_{gate}, W_{up}, W_{down}$: $3 \times [16 \times (1536 + 8960)] = 503,808$ per layer.
  * Per layer total: $659,456$ params.
  * **28 layers total = $18,464,768$ params** ($1.20\%$ of base model).

### 2. Theoretical Memory Breakdown (Batch=1, SeqLen=2048)

| Component | Precision / Dtype | Memory (Bytes) | Memory (MiB) | Memory (GB, $10^9$) |
|---|---|---|---|---|
| **Base Model Weights** | INT4 (NF4 QLoRA + scales) | 910,757,606 | 868.57 MiB | 0.911 GB |
| **LoRA Trainable Weights** | FP32 (PEFT default) | 73,859,072 | 70.44 MiB | 0.074 GB |
| **LoRA Gradients** | FP32 | 73,859,072 | 70.44 MiB | 0.074 GB |
| **Optimizer States** | FP32 AdamW ($2 \times 4$ B/param) | 147,718,144 | 140.88 MiB | 0.148 GB |
| **Activations (with Grad Checkpoint)** | BF16/FP16 inputs + 1 block peak | 408,944,640 | 390.00 MiB | 0.409 GB |
| **Logits Tensor (Loss calculation)** | FP32 ($1 \times 2048 \times 151936 \times 4$) | 1,244,659,712 | 1,187.00 MiB | 1.245 GB |
| **TOTAL TRAINING ALLOCATED PEAK** | — | **2,859,798,246** | **2,727.32 MiB** | **2.860 GB** |

### 3. Merge & Quantization Peaks

* **Base Weights in FP16/BF16** (loaded for merging): $3,087,313,920$ B (~$2,944$ MiB / $3.09$ GB).
* **Merge In-Place on GPU**: Base FP16 + LoRA FP32 delta = **$3,014.73$ MiB (~$3.16$ GB)**.
* **Merge with Duplication on GPU** (unmerged + merged held simultaneously): **$5,959.02$ MiB (~$6.25$ GB)**.
* **4-bit Quantization Peak** (FP16 base + INT4 target + calibration buffer): **$3,692.36$ MiB (~$3.87$ GB)**.

### 4. Memory Gap Explanation (Theory vs Reality)
* **`torch.cuda.memory_allocated()`**: Tracks active PyTorch tensor memory ($2.73$ GiB during training).
* **`torch.cuda.memory_reserved()`**: Tracks blocks allocated by PyTorch's Caching Allocator. Does not release immediately to OS to avoid CUDA allocation overhead; creates fragmentation.
* **`nvidia-smi` (Physical VRAM usage)**: Includes CUDA runtime context (~$450$–$600$ MiB), cuBLAS/cuDNN workspace kernels, PyTorch reserved pool, and any OS desktop/driver allocations.
* **Hardware Headroom**: Colab T4 has $15,360$ MiB. The training peak ($~2.73$ GiB allocated, estimated $~4.5$ GiB `nvidia-smi`) leaves over **$10$ GiB of safety headroom**, proving this configuration safely avoids CUDA OOM.
