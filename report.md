# Soup Take-Home Report — Qwen2.5-1.5B-Instruct (FP16 LoRA → Merge → NF4)

**Вердикт:** **DON'T SHIP** для целевого 4-битного `NF4`-артефакта. Плотная модель `merged` (`FP16`) улучшает точность на отложенном тесте (`+26.52 pp`: `0.5795 -> 0.8447`) и получила `SHIP` от `soup ship` в `bfloat16`, но целевой `NF4`-артефакт провалил два предрегистрированных порога из [acceptance.json](results/acceptance.json): `NF4 Exact Match = 80.30%` (< `90.0%`), а просадка от квантования `nf4 − merged = −4.17 pp` (`95% CI [−7.95, −1.14] pp`) превышает лимит `≤ 3.0 pp`. **Следующий шаг:** расширить LoRA на все линейные слои (`q/k/v/o/gate/up/down_proj`), обучить 2–3 эпохи (или QAT / калиброванный AWQ) и перепроверить `NF4`.

**Среда и конфиг (`soup-t4-02`):** Google Colab `Tesla T4` (`15360 MiB`, CUDA `13.0`), `soup-cli==0.77.2`, `torch==2.14.1+cu130`, `transformers==5.22.1`, `peft==0.21.2`, `trl==0.29.1`, `bitsandbytes==0.50.2`. Модель: `Qwen/Qwen2.5-1.5B-Instruct @ 989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Датасет: Amazon MASSIVE 1.1 `ru-RU`, 12 классов (`1200 train / 100 dev / 264 test`). Конфиг ([soup.yaml](soup.yaml)): FP16 LoRA (`r=16, alpha=32, q_proj/v_proj`, `max_length=512`, `batch=1`, `grad_accum=8`, `epochs=1`, `lr=2e-4`).

---

## 1. Бюджет VRAM: Прогноз vs Факт

| Фаза | Прогноз (Allocated / Device) | `max_memory_allocated` | `max_memory_reserved` | `nvidia-smi max` | CPU Max RSS | Комментарий |
|---|---|---|---|---|---|---|
| `train` (`FP16 LoRA`) | ~3.4–3.9 / 5.0–6.5 GiB | `3537.15 MiB` (`3.454 GiB`) | `5992.0 MiB` (`5.852 GiB`) | `6135 MiB` (`5.991 GiB`) | — | `allocated` в прогнозе; разница с `nvidia-smi` отражает два независимых метода замера (кэш аллокатора vs опрос драйвера 200 мс + CUDA-контекст/буферы) |
| `merge` (CPU) | ~0 GPU / ~3–6 GiB RAM | `0` | `0` | `115 MiB` (`0.112 GiB`) | `6914384 KB` (`~6.59 GiB`) | Время `56.15s`; база + промежуточные тензоры merge в RAM |
| `quant` (`NF4`) | ~1.1–4.0 GiB GPU | `1125.09 MiB` (`1.099 GiB`) | `1146.0 MiB` (`1.119 GiB`) | `1267 MiB` (`1.237 GiB`) | — | Нижняя граница: `bitsandbytes` квантует линейные слои потоково при загрузке |

---

## 2. Сравнение артефактов (`test`, `n=264`)

| Артефакт | Exact Match | Valid Format | Macro-F1 | Размер | Примечание |
|---|---:|---:|---:|---:|---|
| `base` (`FP16`) | `0.5795` (`153/264`) | `0.8674` | `0.5952` | `base_snapshot` | `qa_factoid`: 21, `general_quirky`: 18, `calendar_set`: 14 ошибок |
| `merged` (`FP16`) | `0.8447` (`223/264`) | `0.9811` | `0.8562` | `2.89 GiB` | `+26.52 pp` (`95% CI [+21.21, +31.82] pp`); `soup ship` (`bfloat16`): `SHIP` (`0.67 -> 0.81`, регрессий нет) |
| `nf4` (4-bit) | `0.8030` (`212/264`) | `0.9962` | `0.8165` | `1.08 GiB` | `−4.17 pp` (`95% CI [−7.95, −1.14] pp`): 16 верных ответов сломались, 5 исправились; `qa_factoid` `9 -> 15` |

**Дополнительный 4-модельный тест по 3 темам** ([blind-300-summary.json](retrieved/blind-300-summary.json)): после ревью очищен от 24 пересечений (`3 train + 21 test`) с жёстким сравнением сырого ответа (`n=276`):
- **Тема 1 (обученная: дела/почта, `n=91`):** `base 0.7363`, `adapter = merged 0.9121`, `nf4 0.9011` (`100%` совпадение `adapter == merged`).
- **Тема 2 (обученная: медиа/факты, `n=85`):** `base 0.4824`, `merged 0.7882`, `nf4 0.6353` (**`−15.29 pp`** на тонких классах).
- **Тема 3 (вне обучения: 6 новых классов, `n=100`):** `base 0.8500`, `merged 0.8000`, `nf4 0.7700` (лёгкая переспециализация синтаксиса меток: `cookining_recipe`).

---

## 3. Доказательство обучения, паритета и тихих сбоев

- **Активность адаптера ([verify-summary.json](retrieved/evidence/results/verify-summary.json)):** `PASSED` — `56/56` ненулевых матриц `ΔW = (α/r)·B·A` (`median=0.2674, max=0.5268`), сдвиг логпробов `on/off` `median=0.001146, max=13.005` при нулевом контроле `repeat_noise = 0.0` и `zero_b = 0.0`.
- **Паритет `merge`:** `PASSED` — `logprob_abs_diff` (`n=134`): `median=4.59e-6` (`<0.01`), `p95=0.003625` (`<0.05`), `greedy_agreement=1.000`.
- **Независимый reload `NF4`:** `PASSED` — `196` модулей `Linear4bit` (`nf4`), `has_peft_wrapper=false`, GPU allocated `1090.43 MiB`.

| Негативный контроль (`probes`, `soup-t4-02`) | Что поймало / пропустило |
|---|---|
| Подмена метки (`play_radio -> calendar_query`) | `data doctor` (`exit=0`, `MINOR`): проверяет формат, семантику меток не видит |
| Длинный prompt (`max=1106` vs `512`) | `data doctor` (`exit=0`, `MINOR`): диагностировал `truncation_risk: 1/20 (5.0%)` |
| Свежий `zero-B` LoRA | `soup ship` (`exit=2`, `adapter_noop`) и `verify.py` (`exit=2`, `0/56` ненулевых слоёв) отклонили |
| Подстановка `base` вместо `merged` | `verify.py --mode merge-parity` отклонил (`exit=3`, `p95=1.262`, `agreement=0.9254`) |
| Подстановка dense `FP16` вместо `nf4` | `quantize_nf4.py --mode verify-reload` отклонил (`exit=3`, нет `quantization_config`) |

*(Колонка `pre-flight` прогонялась на чистом датасете; для негативных контролей целевые проверки — `data doctor`, `soup ship`, `verify.py`, `quantize_nf4.py`.)*

---

## 4. Ограничения, важные находки и заметка об ИИ

1. **Маска loss и скорость токенов:** в [rewind.jsonl](retrieved/evidence/artifacts/adapter/rewind.jsonl) в loss вошли `8 800` токенов против `5 200` целевых в [data-token-audit.json](retrieved/evidence/results/data-token-audit.json) (`+3` токена заголовка `<|im_start|>assistant\n` на каждую из 1200 строк из-за отсутствия `{% generation %}` в шаблоне Qwen). Три тривиальных токена префикса занижают средний `loss` и завышают `supervised tok/s` (`~21.5` по логу vs `~12.7` полезных токенов/с).
2. **Расхождение `soup ship` и целевого `NF4`:** `soup ship` проверяет только плотный `merged` (`SHIP`), не видя последующую 4-битную деградацию.
3. **Исправления после код-ревью (локально проверены [test_verifier_and_splits.py](tests/test_verifier_and_splits.py)):**
   - В [verify.py](scripts/verify.py) режим `--mode full` теперь возвращает `exit 2/3` при `FAILED` в `adapter_activity` / `merge_parity` (ранее возвращал `0`).
   - В [eval_3topics_4models.py](scripts/eval_3topics_4models.py) исправлен фильтр утечек (`role == "user"` вместо `messages[-1]` + исключение `test.jsonl`).
4. **Использование ИИ:** работа выполнена совместно с **Antigravity (Gemini 4 Hydrogen)** (пайплайн стадий, скрипты верификации и замера памяти); ручной аудит выявил и исправил баги с `transformers==5.22.1`, `torchao==0.10.0`, `/usr/bin/time`, маской `loss` и кодами возврата `verify.py`. Подробный пошаговый журнал: [EXECUTION_LOG.md](EXECUTION_LOG.md).
