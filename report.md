# Soup Take-Home Report — Qwen2.5-1.5B-Instruct (FP16 LoRA → Merge → NF4)

**Вердикт:** **DON'T SHIP** для целевого 4-битного `NF4`-артефакта. Плотная модель `merged` (`FP16`) статистически значимо улучшает задачу (`+26.52 pp`, `0.5795 -> 0.8447`) и получила `SHIP` от `soup ship`, но целевой `NF4`-артефакт не прошёл два предрегистрированных порога из [acceptance.json](file:///home/belli/soupTest/results/acceptance.json): `NF4 Exact Match = 80.30%` (< `90.0%`), а потеря при квантовании `nf4 − merged = −4.17 pp` (`95% CI [−7.95, −1.14] pp`) превышает лимит `≤ 3.0 pp`. **Следующий шаг:** расширить LoRA на все линейные слои (`q/k/v/o/gate/up/down_proj`), обучить 2–3 эпохи или с QAT (либо калиброванный AWQ/GPTQ), чтобы преодолеть порог `≥ 90%` и удержать потерю квантования `≤ 3.0 pp`.  
**Целевой сценарий:** маршрутизация коротких русскоязычных запросов по 12 классам намерений (Amazon MASSIVE 1.1 `ru-RU`, 1 200 train / 100 dev / 264 test).  
**Целевой runtime:** `transformers` + `bitsandbytes` NF4 на CUDA (Google Colab NVIDIA T4 16 GB).  

---

## 1. Стек и фиксация среды

- **Инструмент обучения:** `soup-cli[train]==0.77.2`, `backend: transformers`
- **Модель:** `Qwen/Qwen2.5-1.5B-Instruct` (`resolved SHA`: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`)
- **Конфигурация (`soup.yaml`):** `quantization: none` (FP16 LoRA), `stream_layers: false`, `gradient_checkpointing: true`, `batch_size: 1`, `gradient_accumulation_steps: 8`, `epochs: 1`, `lr: 2e-4`, `max_length: 512`, `train_on_responses_only: true`, `train_on_eot: true`, `lora: {r: 16, alpha: 32, dropout: 0.05, target_modules: [q_proj, v_proj]}`.
- **Оборудование и ПО (`setup`):**
  - VM: Google Colab (`Linux 6.6.159+ x86_64`), Python `3.13.16`, RAM `12 GiB` (swap `0B`), диск `113G` свободно.
  - GPU: `Tesla T4` (`15360 MiB`, Compute Capability `7.5`), Driver `580.82.07`, CUDA `13.0`.
  - Пакеты: `soup-cli==0.77.2`, `torch==2.14.1+cu130`, `transformers==5.22.1`, `peft==0.21.2`, `trl==0.29.1`, `accelerate==1.15.0`, `bitsandbytes==0.50.2`, `datasets==4.8.5`, `scipy==1.16.3`, `numpy==2.4.6`.
  - `pip check` вернул exit code `1` только из-за предъустановленного `ipython 7.34.0 requires jedi, which is not installed` (стек обучения без конфликтов).

---

## 2. Бюджет VRAM: Прогноз vs Факт

| Фаза | Прогноз (Allocated / Device) | Факт `max_memory_allocated` | Факт `max_memory_reserved` | Факт `nvidia-smi max` | CPU Max RSS | Разница и объяснение |
|---|---|---|---|---|---|---|
| `train` (`FP16 LoRA`, `S=512`) | ~3.4–3.9 GiB / 5.0–6.5 GiB | `3537.15 MiB (3.454 GiB)` | `5992.0 MiB (5.852 GiB)` | `6135 MiB (5.991 GiB)` (`1968` сэмплов) | — | Аллоцированная память точно в прогнозе; разница `reserved - allocated` (`2.40 GiB`) — кэш аллокатора; `+0.14 GiB` в `nvidia-smi` — CUDA-контекст |
| `merge` (CPU) | ~0 GiB GPU / ~3.0–6.0 GiB RAM | `0` (CPU only) | `0` | `115 MiB (0.112 GiB)` (`341` сэмпл) | `6914384 KB (~6.594 GiB)` | Время `56.15s`; GPU почти свободен, CPU RSS `6.59 GiB` (~база `2.9 GiB` + merge-буферы) |
| `quant` (NF4 export) | ~1.1–4.0 GiB GPU | `1125.09 MiB (1.099 GiB)` | `1146.0 MiB (1.119 GiB)` | `1267 MiB (1.237 GiB)` (`250` сэмплов) | — | Нижняя граница (~1.1 GiB): `bitsandbytes` квантует слои потоково при загрузке, без одновременного хранения FP16+NF4 на GPU |

---

## 3. Сравнение артефактов (Base vs Merged vs NF4)

| Артефакт | Split / N | Exact Match | Valid Format | Macro-F1 | Размер на диске | Примечание |
|---|---|---|---|---|---|---|
| `base` (dev baseline) | dev (`100`) | `0.6200` | `0.9300` | `0.6197` | — | `qa_factoid`: 8, `general_quirky`: 6 ошибок |
| `base` | test (`264`) | `0.5795` | `0.8674` | `0.5952` | `base_snapshot` | `qa_factoid`: 21, `general_quirky`: 18, `calendar_set`: 14 |
| `merged` (FP16) | test (`264`) | `0.8447` | `0.9811` | `0.8562` | `2.89 GiB` (`3.10 GB`) | `+26.52 pp` к `base`; `qa_factoid`: 9, `general_quirky`: 7 |
| `nf4` (4-bit reload) | test (`264`) | `0.8030` | `0.9962` | `0.8165` | `1.08 GiB` (`1.15 GB`) | `−4.17 pp` к `merged`; ошибки `qa_factoid`: `9 -> 15`, `weather_query`: `2 -> 5` |

- **Paired bootstrap 95% CI (5 000 ресемплов):**
  - `merged − base`: `+0.2652` (`+26.52 pp`), 95% CI `[+0.2121, +0.3182]` (уверенное улучшение, нижняя граница `> 0`)
  - `nf4 − merged`: `−0.0417` (`−4.17 pp`), 95% CI `[−0.0795, −0.0114]` (порог потерь `≤ 3.0 pp` провален)
- **Вердикт `soup ship` (проверяет только плотный `merged`, `numerics: bfloat16`):** `SHIP` (`Leg 1 task win: 0.6700 -> 0.8100`; все 8 мини-бенчмарков без регрессии `> 5%`: `mini_format_json: −0.0500`, `mini_instruction: −0.0417`, `mini_mmlu: +0.0250`, `mini_over_refusal: +0.0250`, остальные `0.0`).

### 3.1. Слепой 4-модельный прогон на 300 вопросах (`3 × 100`, 0 пересечений с train/dev)

| Срез (N) | `base` | `base + adapter` | `merged` (FP16) | `nf4` (4-bit) | Совп. `adapter==merged` | Разница `nf4 − merged` |
|---|---:|---:|---:|---:|---:|---:|
| Тема 1: дела/почта/календарь (`100`, обученная) | `0.77` | `0.90` | `0.90` | `0.89` | `100%` | `−1.0 pp` (`[−5, +2]`) |
| Тема 2: медиа/погода/факты (`100`, обученная) | `0.53` | `0.80` | `0.80` | `0.65` | `100%` | `−15.0 pp` (`[−23, −7]`) |
| Тема 3: новые классы (`100`, **вне обучения**) | `0.87` | `0.80` | `0.80` | `0.77` | `100%` | `−3.0 pp` (`[−7, 0]`) |
| **Итого по 2 обученным темам (`200`)** | `0.65` | `0.85` | `0.85` | `0.77` | `100%` | `−8.0 pp` (`[−13, −3.5]`) |
| **Все 300 вопросов подряд (`300`)** | `0.723` | `0.833` | `0.833` | `0.770` | `100%` | `−6.3 pp` (`[−9.7, −3.3]`) |

*(Сырые ответы всех 4 моделей по каждому вопросу сохранены без редактуры в [blind-300-paired.csv](file:///home/belli/soupTest/retrieved/blind-300-paired.csv) и [blind-300-summary.json](file:///home/belli/soupTest/retrieved/blind-300-summary.json).)*

---

## 4. Доказательство обучения и паритета merge

- **Активность адаптера (`verify.py`):** `PASSED` — `56/56` ненулевых слоёв (`delta_norm`: median `0.2674`, p95 `0.4633`, max `0.5268`); сдвиг логпробов `adapter on/off` (n=134): median `0.001146`, p95 `1.2623`, max `13.0050`, тогда как контрольный повтор (`repeat_noise`) и `zero-B` дают `0.0` (порог `1e-6`).
- **Паритет merge (`adapter-on` vs reloaded `merged`):** `PASSED` — `logprob_abs_diff` (n=134): median `4.59e-6` (< `0.01`), p95 `0.003625` (< `0.05`), max `0.03055`, `greedy_token_agreement = 1.000` (`0` расхождений).
- **Независимая перезагрузка NF4 (`quantize_nf4.py --mode verify-reload`):** `PASSED` — `is_loaded_in_4bit=true`, `linear4bit_module_count=196`, `quant_types_found=["nf4"]`, `has_peft_wrapper=false`, `tokenizer_matches_base=true`, GPU allocated `1090.43 MiB`.

---

## 5. Исследование тихих сбоев (Silent Failures)

| Сбой / Контрольный опыт | `data doctor` | `pre-flight` | `soup ship` | `verify.py` / `quantize_nf4.py` | Лог и детали (`soup-t4-02`, `probes`) |
|---|---|---|---|---|---|
| Подмена метки класса в копии train | `exit=0` (`MINOR`) | — | — | — | `play_radio -> calendar_query`: формат валиден, но семантическую ошибку инструмент не ловит |
| Длинный user-ввод (обрезка ответа) | `exit=0` (`MINOR`) | — | — | — | Зафиксировал `truncation_risk: 1/20 rows (5.0%) would be truncated` (`max=1106 vs 512`), но не заблокировал |
| Свежий LoRA с нулевой матрицей `B` | — | — | `exit=2` (`DON'T SHIP`) | `exit=2` (`FAILED`) | `soup ship`: `adapter_noop`; `verify.py`: `nonzero_delta_layers=0/56`, `on_off=0.0` |
| Подстановка `base` вместо `merged` | — | — | — | `exit=3` (`FAILED`) | `verify.py --mode merge-parity`: `p95=1.262`, `max=13.005`, `greedy_agreement=0.9254` |
| Подстановка dense FP16 вместо `nf4` | — | — | — | `exit=3` (`FAILED`) | `quantize_nf4.py --mode verify-reload`: нет `quantization_config` в `config.json` |

---

## 6. Ограничения проверки, неожиданности и заметка об ИИ

- **Ограничения:** проверка измеряет узкий сценарий маршрутизации (12 классов, `ru-RU`, greedy decoding) и не покрывает утечку из pretrain, общую деградацию на других языках/задачах, многопоточную нагрузку и длинный контекст.
- **Что удивило и что беспокоит:**
  1. Расхождение между `soup ship` и целевым деплоем: `soup ship` проверял плотный `merged` и выдал `SHIP` (`0.67 -> 0.81`), но последующая 4-битная квантизация NF4 съела `−4.17 pp` на `test` (особенно в тонких классах `qa_factoid`: `9 -> 15` ошибок и `weather_query`: `2 -> 5`), провалив оба порога приёмки (`>= 90%` EM и падение `<= 3 pp`).
  2. Пик GPU на `quant` составил `1125.09 MiB` (`nvidia-smi max 1267 MiB`) вместо верхней оценки `~4.0 GiB`, потому что `transformers==5.22.1` + `bitsandbytes==0.50.2` квантуют линейные слои потоково при загрузке.
  3. Три «тихие» особенности стека потребовали явных фиксов: `transformers==5.22.1` возвращает `BatchEncoding` из `apply_chat_template`, Colab `torchao==0.10.0` ломает PEFT, а шаблон Qwen без `{% generation %}` оставляет `soup data doctor` на уровне `MINOR` с эвристическим маскированием.
- **Использование ИИ:**
  - Работа выполнена совместно с **Antigravity (Gemini 4 Hydrogen)**.
  - **Принято от ИИ:** автоматизация фазового пайплайна (`run_phase.sh`, `stages/*.sh`, `scripts/*.py`), изоляция замеров `max_memory_allocated` / `max_memory_reserved` в том же процессе (`measure_soup.py`), схема 5-компонентной верификации и парного bootstrap.
  - **Проверено, исправлено и переделано:** обновлена конфигурация с чернового QLoRA (`seq_len=2048`, 7 проекций) на экспериментально проверяемую **FP16 LoRA** (`q_proj, v_proj`, `max_length=512`, `Amazon MASSIVE 1.1 ru-RU`); добавлен `keyrings.alt` для сохранения OAuth-сессии `google-colab-cli` в WSL/Linux без D-Bus; исправлено экранирование в [run_colab_phase.sh](file:///home/belli/soupTest/run_colab_phase.sh) для корректного стриминга stdout через `colab exec`.

---

## Журнал выполнения шагов (Live Log)

- **2026-11-13 09:23–09:26:** Подготовлены локальные конфиги и скрипты (`soup.yaml`, `run_phase.sh`, `stages/*`, `scripts/*`, `results/memory-estimate.md`, `results/acceptance.json`). Проверен синтаксис (`bash -n`, `py_compile`).
- **2026-11-13 09:30–09:33:** Установлен `google-colab-cli==0.8.0` (+ `keyrings.alt==5.0.2`), пройдена OAuth2-авторизация, выделена сессия Colab `soup-t4-01` (`--gpu T4`, Python 3.13.16, `nvidia-smi` ОК).
- **2026-11-13 09:35–10:10:** Выполнена фаза `data` (`phase_exit=0`): скачан `amazon-massive-dataset-1.1.tar.gz` (`sha256: 4cba5faa...`), подготовлены 12 классов (`train=1200`, `dev=100`, `test=264`, исключено 9 конфликтующих нормализованных текстов и 119 дублей/пересечений), скачана модель `Qwen/Qwen2.5-1.5B-Instruct` (`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`). При проверке `results/data-token-audit.json` обнаружен баг совместимости с `transformers==5.22.1`: `apply_chat_template(..., tokenize=True)` по умолчанию возвращает `BatchEncoding` (`dict` из 2 ключей), а не `list[int]`. Исправлено в [prepare_data.py](file:///home/belli/soupTest/scripts/prepare_data.py) и [verify.py](file:///home/belli/soupTest/scripts/verify.py) (`return_dict=False` + `extract_ids`). Также зафиксирован баг `google-colab-cli==0.8.0`: при обрыве WebSocket (`Connection was lost.`) не завершается поток `jupyter_kernel_client`. После перезапуска фазы `data` получен реальный аудит токенов: `train_full_tokens` (`min=206, p50=217, p95=229, max=257`, `truncated=0` при `max_length=512`), `train_response_tokens` (`min=4, p50=4, max=6`, всего `5200` supervised-токенов).
- **2026-11-13 10:10:** Завершена фаза `preflight` (`phase_exit=0`):
  - `soup data validate`: `1200/1200 rows valid for chatml format`.
  - `soup data doctor`: общий статус `MINOR` (`1200/1200`); `eos_in_labels: OK` (`eos_token_id=[151645]`), `truncation_risk: OK` (`p50=217, p95=229, max=257 vs 512`), но `generation_markers: MINOR` (шаблон Qwen без `{% generation %}`, поэтому `assistant_only` использует эвристику).
  - `soup doctor --config soup.yaml`: все `[train]`-зависимости OK, unread-настроек `0`, GPU `Tesla T4 (sm_75, BF16=no)`.
  - `soup train --dry-run --yes`: `Data OK: 1200 train samples`, `Config valid. Ready to train!`.
  - Завершена фаза `baseline` (`phase_exit=0`): на `dev` (`n=100`) базовая модель `Qwen2.5-1.5B-Instruct` показала `exact_match=0.62`, `valid_format_rate=0.93`, `macro_f1=0.6197`.
- **2026-11-13 10:12–10:23:** При первом запуске `bench` `colab exec` упал с `Connection was lost.` и завис до `timeout 600`. Переделан [run_colab_phase.sh](file:///home/belli/soupTest/run_colab_phase.sh): теперь фазы запускаются на VM через `nohup` в фоне, а локальный раннер опрашивает `status.txt` короткими вызовами с `timeout 30` и ретраями.
- **2026-11-13 10:22–10:23:** Первый запуск `bench` (`logs/20261113T042217Z-bench-CLgr5d`, `phase_exit=1`) упал при инициализации LoRA: предъустановленный в Colab `torchao==0.10.0` несовместим с `peft==0.21.2` (`soup doctor` ранее отмечал `torchao outdated (need >=0.18.0)`). При этом зафиксирована память загрузки FP16 базы на T4: `max_memory_allocated = 2.8754 GiB (2944.4 MiB)`, `max_memory_reserved = 2.9297 GiB (3000.0 MiB)`. Удалён `torchao` на VM и в [setup.sh](file:///home/belli/soupTest/stages/setup.sh).
- **2026-11-13 10:23–10:25:** Повторная фаза `bench` (`logs/20261113T042302Z-bench-MHXjMh`, `phase_exit=0`):
  - `LoRA applied: 2,179,072 trainable / 1,545,893,376 total (0.14%)`, `dtype=torch.float16`, `trainable_parameters=112` тензоров, `grad_norm` конечны и ненулевые, `parameters_changed=true`, `valid=true`.
  - Скорость (17 шагов после 3 warmup): median `2.509 s/step`, p95 `3.505 s/step`, `21.73 supervised tok/s` (`1006 / 29844` токенов, `3.37%`), `644.71 total tok/s`.
  - Память на T4: `max_memory_allocated = 3522.17 MiB (3.440 GiB)`, `max_memory_reserved = 5110.0 MiB (4.990 GiB)`, пик `nvidia-smi` (413 сэмплов) = `5253 MiB (5.130 GiB)`. Полное совпадение с предрасчётными диапазонами (`~3.4–3.9 GiB` / `5.0–6.5 GiB`).
- **2026-11-13 10:25–10:32:** Завершено полное обучение `train` (`logs/20261113T042520Z-train-rndnsy`, `phase_exit=0`):
  - `150/150` шагов оптимизатора (`train_runtime=367s` / `6m07s`, `elapsed_sec=392.12s`, `0.409 steps/s`, `3.27 samples/s`).
  - `Loss: 1.1946 -> 0.0545` (`train_loss=0.1595`, финальный `mean_token_accuracy=0.9828`, `grad_norm=0.2011`).
  - Память на T4: `max_memory_allocated = 3537.15 MiB (3.454 GiB)`, `max_memory_reserved = 5992.0 MiB (5.852 GiB)`, пик `nvidia-smi` (1 960 сэмплов) = `6135 MiB (5.991 GiB)`.
  - Адаптер сохранён в `artifacts/adapter` (`adapter_model.safetensors`, `sha256: 8875473b...`).
- **2026-11-13 10:37–10:41:** Пробный прогон последующих фаз:
  - `merge` (`logs/20261113T043721Z-merge-ducOWr`, `exit=127`): в контейнере Colab отсутствует `/usr/bin/time`. В [merge.sh](file:///home/belli/soupTest/stages/merge.sh) добавлен кросс-платформенный fallback на Python `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`.
  - Из-за отсутствия `artifacts/merged` ожидаемо упали `quant` (`exit=1`), `verify` (`exit=1` при загрузке `artifacts/merged`), `ship` (`exit=1`) и Probe 5.
  - При этом в `probes` (`logs/20261113T043936Z-probes-twtwCD`, `exit=0`) уже получены рабочие результаты по Пробам 1–4:
    - **Probe 1 (подмена метки):** `data doctor` вернул `exit=0` (`MINOR`) — семантическую ошибку в метках не видит.
    - **Probe 2 (длинный user-ввод, 1106 токенов):** `data doctor` зафиксировал `truncation_risk: 1/20 rows (5.0%) would be truncated` (`max=1106 vs 512`), общий вердикт остался `MINOR` (`exit=0`).
    - **Probe 3 (свежий zero-B адаптер):** `verify.py` отклонил (`status=FAILED`, `nonzero_delta_layers=0/56`, `exit=2`); `soup ship` отклонил по правилу `adapter_noop` (`DON'T SHIP`, `exit=2`).
    - **Probe 4 (`base` вместо `merged`):** `verify.py --mode merge-parity` отклонил (`status=FAILED`, `p95=1.261`, `max=13.01`, `greedy_agreement=0.9254`, `exit=3`).
- **2026-11-13 11:24–12:06 (`soup-t4-02`, полный чистый прогон):** На свежей T4 VM последовательно и без ошибок (`phase_exit=0` на всех 11 стадиях) прогнаны `setup -> data -> preflight -> baseline -> bench -> train -> merge -> quant -> verify -> probes -> ship`. Ключевые замеры: `train` (`150/150`, `Loss 1.1943 -> 0.0545`, `max_memory_allocated=3537.15 MiB`, `max_memory_reserved=5992.0 MiB`, `nvidia-smi max=6135 MiB`), `merge` (`56.148s`, CPU `Max RSS=6914384 KB`, `nvidia-smi max=115 MiB`), `quant` (`10.123s`, `max_memory_allocated=1125.09 MiB`, `nvidia-smi max=1267 MiB`), `verify` (`adapter_activity` и `merge_parity` PASSED; `test` `n=264`: `base=0.5795`, `merged=0.8447`, `nf4=0.8030`), `probes` (Probes 1–5 по плану), `ship` (`SHIP` для dense `merged`). Архив доказательств [evidence-20261113T060438Z.tar.gz](file:///home/belli/soupTest/retrieved/evidence-20261113T060438Z.tar.gz) (`sha256: 96ba882c3f2794c38ae51b0a34c8899a7ff4127af3d2a10db544ebc6505d4a26`) и ноутбук сессии [evidence-20261113T060438Z-session.ipynb](file:///home/belli/soupTest/retrieved/evidence-20261113T060438Z-session.ipynb) скачаны локально и проверены (`sha256sum -c: OK`), сессия `soup-t4-02` остановлена (`colab stop`).
- **2026-11-13 13:19–13:29 (`soup-t4-03`, слепой 4-модельный тест на 300 вопросах):** Из сохранённого адаптера заново собраны `merged` (`53.28s`) и `nf4` (`1101.26 MiB`) на свежей T4 VM и запущен [eval_3topics_4models.py](file:///home/belli/soupTest/scripts/eval_3topics_4models.py) на 300 вопросах (`0` пересечений с `train`/`dev`: 2 обученные темы × 100 + 1 новая тема × 100). Подтверждён 100% паритет `base+adapter == merged` (`300/300` идентичных ответов), прирост `+20.0 pp` на обученных темах (`0.65 -> 0.85`), провал `nf4` на второй теме (`0.80 -> 0.65`, `−15 pp`) и лёгкое переобучение на новой теме (`base 0.87` vs `merged 0.80`, `nf4 0.77`). Результаты скачаны в [blind-300-summary.json](file:///home/belli/soupTest/retrieved/blind-300-summary.json) и [blind-300-paired.csv](file:///home/belli/soupTest/retrieved/blind-300-paired.csv), сессия `soup-t4-03` остановлена.
