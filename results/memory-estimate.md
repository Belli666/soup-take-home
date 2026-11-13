# Предварительный расчёт памяти (Pre-run Memory Estimate)

**Дата фиксации:** 2026-11-13 (до первого GPU-прогона)  
**Целевое устройство:** Google Colab Free Tier — NVIDIA T4 (16 GB / 15,360 MiB)  
**Модель:** `Qwen/Qwen2.5-1.5B-Instruct` (~1,543,656,960 параметров, 28 слоёв, `hidden_size=1536`, `num_attention_heads=12`, `num_key_value_heads=2`, `kv_dim=256`, `vocab_size=151936`, `tie_word_embeddings=true`)  
**Стартовая конфигурация (`soup.yaml`):** FP16 LoRA (`training.quantization: none`), `stream_layers: false`, `gradient_checkpointing: true`, `batch_size: 1`, `gradient_accumulation_steps: 8`, `max_length: 512`, `lora: {r: 16, alpha: 32, target_modules: [q_proj, v_proj]}`

| Компонент / Фаза | Расчёт и порядок величины |
|---|---|
| **Замороженные веса базы (FP16)** | `1.54e9 × 2` байта ≈ **2.87 GiB**. Градиенты и состояния AdamW для базы не создаются. |
| **LoRA адаптер, градиенты и AdamW** | `28 × 16 × [(1536 + 1536) + (1536 + 256)] = 2,179,072` параметров (~0.14% базы). При FP32 весах, градиентах и двух моментах AdamW: `N × 16` байт ≈ **33.25 MiB**. |
| **Активации (Grad Checkpointing, `B=1, S=512`)** | Входы checkpoint-блоков: `28 × 1 × 512 × 1536 × 2 B` ≈ **42.0 MiB**. С учётом временных тензоров текущего блока и recompute: ориентир **0.08–0.20 GiB**. |
| **Логиты и loss (`B=1, S=512, V=151936`)** | `1 × 512 × 151936` элементов ≈ **148.4 MiB** (FP16) или **296.8 MiB** (FP32). FP16 логиты + FP32 копия + рабочий буфер loss: до **~0.72 GiB (742 MiB)**. |
| **Пик Train (прогноз)** | **~3.4–3.9 GiB** учтённых тензоров (`allocated`). Плановый бюджет устройства (`nvidia-smi`) с учётом CUDA context (~0.5 GiB), workspace kernels и PyTorch allocator cache: **5.0–6.5 GiB** (из 15.0 GiB T4). |
| **Merge (CPU)** | Одна плотная FP16 копия ≈ **2.87 GiB**, две копии при конвертации ≈ **5.74 GiB** + runtime/буферы. На GPU ожидается только фоновое потребление. Замеряем max RSS через `/usr/bin/time -v`. |
| **NF4 export (GPU)** | Связанные embeddings/head `E = 151936 × 1536` (~233.4M params). Пост-загрузочный резидентный ориентир: `E × 2 B + (1.54e9 - E) × 0.5 B` ≈ **1.04 GiB** (+ metadata/scales ≈ **1.1 GiB**). При одновременной резидентности полной FP16 и NF4 копий до освобождения буферов — **~4.0 GiB** (+ workspace). |

### Запасной вариант (если smoke/bench покажет OOM)
При OOM в FP16 переключиться на QLoRA (`training.quantization: 4bit`), где замороженная база занимает ~0.85–1.05 GiB вместо 2.87 GiB, и пересчитать бюджет перед повторным запуском.

---

### Таблица «Прогноз vs Факт» (заполняется после прогона на Colab T4)

| Фаза | Прогноз (Allocated / Device) | Факт `max_memory_allocated` | Факт `max_memory_reserved` | Факт `nvidia-smi max` | CPU Max RSS | Примечание / Gap |
|---|---|---|---|---|---|---|
| `train` | ~3.4–3.9 GiB / 5.0–6.5 GiB | `3537.15 MiB (3.454 GiB)` | `5992.0 MiB (5.852 GiB)` | `6135 MiB (5.991 GiB)` (`1960` сэмплов) | — | Аллоцировано строго в прогнозе; `reserved` +`2.40 GiB` (кэш аллокатора), `nvidia-smi` +`0.14 GiB` (CUDA context) |
| `merge` (CPU) | ~0 GiB GPU / ~3–6 GiB RAM | `0` (CPU only) | `0` | `115 MiB (0.112 GiB)` (`341` сэмплов) | `6914384 KB (~6.594 GiB)` | GPU почти не занят (фоновый контекст); CPU RSS `6.59 GiB` (база `~2.9 GiB` + merge буферы/сейфтензоры) |
| `quant` (NF4) | ~1.1–4.0 GiB GPU | `1125.09 MiB (1.099 GiB)` | `1146.0 MiB (1.119 GiB)` | `1267 MiB (1.237 GiB)` (`250` сэмплов) | — | Факт на нижней границе (~1.1 GiB): `bitsandbytes` квантует слои по мере загрузки, не держа одновременно полные FP16 и NF4 веса на GPU |
