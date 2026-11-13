# SOUP INTERNSHIP · TAKE-HOME TASK — PROJECT STATE JOURNAL
**File:** `/home/belli/soupTest/PROJECT_STATE.md`  
**Last Updated:** 2026-11-12 22:25:00  
**Target Hardware:** Google Colab Free Tier (NVIDIA T4, 16 GB VRAM)  
**Target Stack:** `soup-cli[train]`, PyTorch, Hugging Face PEFT / Transformers  
**Primary Target Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Primary Target Dataset:** Open instruction dataset (1k–2k examples)  

---

## 📌 Status Legend & Rigor Rules
* **`[VERIFIED]`**: Experimentally confirmed by code execution, hardware telemetry, or raw logs.
* **`[CALCULATED]`**: Mathematically derived from model architecture and specifications.
* **`[UNVERIFIED]`**: Working hypothesis, assumption, or pending runtime confirmation.

---

## 1. Objective & Core Requirements (From ТЗ)
* **Goal:** Fine-tune open LLM ($\le 3\text{B}$) with LoRA, merge adapter into base, quantize to 4-bit, and make an engineering call: **SHIP or DON'T SHIP**.
* **Deliverables:**
  1. Report ($\le 2$ pages).
  2. Raw timestamped logs.
  3. Raw `nvidia-smi` output.
  4. Verification script (Base vs Merged vs 4-bit).
  5. Finished notebook or code.
  6. Reflection paragraph (surprises & ongoing concerns).
  7. AI tools log (accepted/rejected suggestions, verifications).

---

## 2. Model & Dataset Selection

### Target Model
* **Model ID:** `Qwen/Qwen2.5-1.5B-Instruct` `[CALCULATED]`
  * Parameters: $1,543,656,960$ ($1.54\text{B}$).
  * Vocab Size: $151,936$ (Gives large logits tensor: $1.19\text{ GiB}$ in FP32 at seq 2048).
  * Architecture: 28 layers, $h=1536$, intermediate $8960$, 12 Q-heads, 2 KV-heads (GQA).
  * **Justification:** Fits cleanly in Colab T4 (16 GB), representative modern architecture (GQA, RoPE, SwiGLU, large vocabulary), accessible instruction-following baseline.

### Target Dataset
* **Status:** `[UNVERIFIED]` (Candidate selection in progress).
* **Requirements:** 1k–2k examples, open instruction dataset, language directly evaluable by candidate (Russian / English).
* **Candidates under review:**
  1. `tatsu-lab/alpaca` (clean 1.5k English slice).
  2. `IlyaGusev/ru_turbo_alpaca` (clean 1.5k Russian slice).
  3. Custom domain-specific instruction holdout for evaluation.

---

## 3. Project File Inventory & Architecture

```text
/home/belli/soupTest/
├── PROJECT_STATE.md       # [VERIFIED] Master source of truth across all sessions/models
├── README.md              # [VERIFIED] Public research log & AI collaboration notes
├── requirements.txt       # [VERIFIED] Pinned dependencies (soup-cli[train], torch, peft, etc.)
├── vram_calculator.py     # [VERIFIED] Reproducible VRAM calculation script
├── vram_budget.json       # [VERIFIED] Exact memory breakdown in bytes, MiB, GB
└── verification.py        # [VERIFIED] 5-pillar verification script (Base vs Merged vs 4-bit)
```

---

## 4. Part 1: VRAM Budget & Memory Analysis

### 4.1. LoRA Parameters Breakdown ($r=16$) `[CALCULATED]`
* Modules targeted: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
* Attention per layer: $2 \times [16 \times (1536 + 1536)] + 2 \times [16 \times (1536 + 256)] = 155,648$.
* MLP per layer: $3 \times [16 \times (1536 + 8960)] = 503,808$.
* Per layer total: $659,456$ params.
* **Total LoRA Parameters (28 layers):** **$18,464,768$** ($1.20\%$ of base model). `[CALCULATED]`

### 4.2. Memory Budget Table (Batch=1, SeqLen=2048) `[CALCULATED]`

| Component | Dtype / Format | Bytes | MiB | GB ($10^9$) | Status |
|---|---|---|---|---|---|
| **Base Model Weights** | INT4 (NF4 QLoRA + scales) | 910,757,606 | 868.57 | 0.911 | `[CALCULATED]` |
| **LoRA Trainable Weights** | FP32 (PEFT default) | 73,859,072 | 70.44 | 0.074 | `[CALCULATED]` |
| **LoRA Gradients** | FP32 | 73,859,072 | 70.44 | 0.074 | `[CALCULATED]` |
| **Optimizer States** | FP32 AdamW ($2 \times 4\text{ B}$) | 147,718,144 | 140.88 | 0.148 | `[CALCULATED]` |
| **Activations (Grad Checkpointing)** | BF16/FP16 inputs + 1 block peak | 408,944,640 | 390.00 | 0.409 | `[CALCULATED]` |
| **Logits / Cross-Entropy Peak** | FP32 ($1 \times 2048 \times 151936 \times 4$) | 1,244,659,712 | 1,187.00 | 1.245 | `[CALCULATED]` |
| **TOTAL TRAINING PEAK (Allocated)** | — | **2,859,798,246** | **2,727.32** | **2.860** | `[CALCULATED]` |
| **Base FP16 Weights (for Merge)** | FP16 ($1.54\text{B} \times 2$) | 3,087,313,920 | 2,944.29 | 3.087 | `[CALCULATED]` |
| **Merge In-Place Peak (GPU)** | Base FP16 + LoRA delta | 3,161,172,992 | 3,014.73 | 3.161 | `[CALCULATED]` |
| **Merge Duplication Peak (GPU)** | Base FP16 $\times 2$ + LoRA | 6,248,486,912 | 5,959.02 | 6.248 | `[CALCULATED]` |
| **Quantization Peak (GPU)** | FP16 + INT4 target + buffers | 3,871,725,312 | 3,692.36 | 3.872 | `[CALCULATED]` |

### 4.3. The Memory Gap (Theory vs Measurement)
* `torch.cuda.memory_allocated()`: Tracks active tensor buffers ($~2.73\text{ GiB}$). `[CALCULATED]`
* `torch.cuda.memory_reserved()`: Tracks PyTorch caching allocator pool (expected $~3.5\text{–}4.5\text{ GiB}$). `[UNVERIFIED]` (Awaiting T4 run).
* `nvidia-smi`: Includes CUDA context (~$500\text{ MiB}$), cuBLAS runtime, driver allocations. Expected actual peak: $~4.2\text{–}5.0\text{ GiB}$. `[UNVERIFIED]` (Awaiting T4 run).

---

## 5. Part 2: Verification Framework ("Prove the Model Actually Trained")

### 5.1. Why Falling Loss Proves Nothing `[CALCULATED]`
1. **Unmasked Prompt Memorization:** Fitting input template tokens drops loss without teaching output generation.
2. **Mode Collapse / Degeneration:** Repetitive token outputs (periods, whitespace) minimize cross-entropy loss while destroying utility.
3. **Dead LoRA Matrix ($B \approx 0$):** Since $B=0$ at initialization, vanishing gradients leave weights untouched while scheduler decay creates superficial loss drop.
4. **Overfitting:** 1k samples can be memorized to zero loss while out-of-domain generalization diverges.

### 5.2. Verification Script (`verification.py`) Architecture `[VERIFIED]`
1. **Weight Difference:** Layer-by-layer $L_2$ norm, relative norm $\frac{\Vert\Delta W\Vert_2}{\Vert W\Vert_2}$, max abs diff, and % changed parameters.
2. **Adapter Integrity:** Direct norm inspection of LoRA $B$ matrices ($\Vert B\Vert_2 > 0$ assertion).
3. **Logits Comparison:** Token-aligned KL Divergence ($D_{KL}$), Cosine Similarity, and Top-1 Agreement Rate on identical prompts.
4. **Holdout Evaluation:** Masked-prompt Perplexity ($PPL = \exp(\text{loss})$ with `labels=-100` on prompts) + deterministic greedy generation (`do_sample=False`, `temp=0.0`).
5. **Quantization Fidelity:** Merged FP16 vs 4-bit degradation analysis on logits and holdout perplexity.

### 5.3. Explicit Limitations (What the Check CANNOT Detect) `[CALCULATED]`
1. **Train-Test Data Contamination:** Memorized holdout sets yield false high accuracy.
2. **Catastrophic Forgetting:** Does not measure regression on tasks outside the holdout distribution (e.g. coding, reasoning, multilingual).
3. **Hallucinations & Factuality Drift:** Statistical logit alignment does not guarantee factual correctness.
4. **Adversarial / Long-Context Degeneration:** Greedy single-turn checks miss multi-turn looping or injection vulnerabilities.
5. **Inverted Masking Artifacts:** Inverted loss masks alter weights significantly but learn the wrong conditional distribution.

---

## 6. Part 3: Silent Failures Registry (Pending Empirical Testing)
*Status: `[UNVERIFIED]` — Mapping failure modes to Soup CLI built-ins:*

| Stage | Potential Silent Failure | Detection Method | Caught by `soup doctor`? | Caught by `soup preflight`? | Caught by `soup ship`? |
|---|---|---|---|---|---|
| **Data** | Broken JSONL encoding / null values | Data schema validator | `[UNVERIFIED]` | `[UNVERIFIED]` | `[UNVERIFIED]` |
| **Tokenizer & Template** | Chat template missing generation prompt / stripped EOS token | Tokenizer inspection | `[UNVERIFIED]` | `[UNVERIFIED]` | `[UNVERIFIED]` |
| **Training** | Inverted loss mask (training on prompt, not response) | Token loss inspector | `[UNVERIFIED]` | `[UNVERIFIED]` | `[UNVERIFIED]` |
| **Merge** | Silent scaling factor omission ($W + \frac{\alpha}{r} \Delta W$) | Weight diff auditor | `[UNVERIFIED]` | `[UNVERIFIED]` | `[UNVERIFIED]` |
| **Quantization** | Outlier activation clipping destroying attention | KL logit divergence | `[UNVERIFIED]` | `[UNVERIFIED]` | `[UNVERIFIED]` |

---

## 7. Current Project Stage & Next Actions
* **Current Stage:** Part 1 & Part 2 Architectures Approved & Documented.
* **Immediate Next Actions:**
  1. Select and inspect the 1k–2k instruction dataset.
  2. Map out Part 3 (Silent Failures) test cases.
  3. Prepare Colab execution environment (install `soup-cli[train]` and capture baseline `nvidia-smi`).
