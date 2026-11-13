# План Soup take-home на Colab T4

Рабочий план для выполнения задания. Команды сверены с документацией, результаты эксперимента предстоит получить на Colab. В сдаваемый отчёт войдут только фактические измерения и выводы. Лимит отчёта — две страницы. На выполнение отведено 4–6 часов, на отправку — 48 часов.

## 1. Зафиксировать эксперимент

**Qwen2.5-1.5B-Instruct → FP16 LoRA в Soup → FP16 merge → отдельный NF4-артефакт → проверка на T4.** Целевой сценарий — классификация коротких русскоязычных запросов по ограниченному списку намерений. Целевой runtime — Transformers с bitsandbytes на CUDA. Qwen поддерживает русский, весит около 1,54 млрд параметров и распространяется под Apache 2.0. Определённый список допустимых ответов позволяет проверять точность без LLM-судьи. [Карточка Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) · [Документация bitsandbytes](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes)

Оставь `data.streaming: false`, потому что 1 200 строк помещаются в RAM. Оставь `training.stream_layers: false`, чтобы держать базу на T4 и избежать переносов между GPU, RAM и диском. Явно укажи `training.quantization: none` для обычной LoRA. Если короткий прогон подтвердит нехватку VRAM при microbatch 1 и checkpointing, сохрани ошибку и повтори smoke с `training.quantization: 4bit` в отдельном каталоге. Это QLoRA. В обеих ветках финальный NF4 создаётся отдельно из сохранённого FP16 merge. [Схема Soup 0.77.2](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.2/docs/config-reference.md) · [Стриминг слоёв](https://trysoup.dev/docs/layer-streaming) · [QLoRA](https://arxiv.org/abs/2305.14314)

| Время работы | Что должно быть готово |
|---|---|
| 0:00–0:50 | Рабочие скрипты подготовки данных, фиксированные сплиты, предварительный расчёт памяти, доступ к T4. |
| 0:50–1:30 | Версии и диагностика, аудит масок, dev baseline, smoke train. |
| 1:30–3:10 | Основной train, резервная копия адаптера, merge, NF4 и загрузка с диска. |
| 3:10–4:40 | Сравнение артефактов, отрицательные контроли, `soup ship`, замеры по этапам. |
| 4:40–6:00 | Вердикт, двухстраничный отчёт, проверка архива и отправка. |

Сначала подготовь локальные файлы и сохрани расчёт из раздела 6, затем выделяй GPU. Если времени мало, выполни один полный прогон и отрицательные контроли на копиях артефактов. Качество измеряй на одном фиксированном test.

## 2. Подготовить рабочий каталог

Создай отдельный каталог эксперимента. Все файлы ниже, кроме этого плана, предстоит написать или получить при запуске. Один комплект кода должен запускаться как через CLI, так и из notebook.

```text
soup-takehome/
  soup.yaml
  run_phase.sh
  stages/                 # setup.sh, data.sh, preflight.sh, baseline.sh, bench.sh, train.sh,
                          # merge.sh, quant.sh, verify.sh, probes.sh, ship.sh
  scripts/
    prepare_data.py       # raw JSONL → train/dev/test и manifest
    measure_soup.py       # запуск Soup и CUDA-метрики в том же процессе
    quantize_nf4.py       # сохранённый FP16 merge → самостоятельный NF4
    verify.py             # доказательство обучения и сравнение артефактов
  data/                   # входные сплиты, prompt, label map, атрибуция, хэши
  artifacts/              # adapter/, merged/, nf4/
  logs/                   # отдельные каталоги фаз и попыток
  results/                # per-example JSONL, метрики, сверка памяти и пороги
  report.md               # черновик сдаваемого отчёта
```

Зафиксируй SHA модели и исходного архива данных, `seed`, версию CLI и пакетов, конфиг и хэши входов. Используй один снимок базовой модели для обучения, merge и baseline. Отдельный каталог попытки позволяет сохранить неудачные результаты, повторить этап и установить, каким адаптером был получен конкретный NF4.

## 3. Открыть T4 через Google Colab CLI

### Локальный запуск из Windows

В Windows запускай команды Colab CLI из **WSL 2 с Ubuntu**. Официальный CLI документирует Linux и macOS. Выбранный релиз CLI — `0.8.0`, ему нужен локальный Python 3.12 или новее. Если WSL уже установлен, открой Ubuntu. В противном случае установка выполняется в PowerShell по инструкции Microsoft. Команды далее относятся к Bash в WSL. [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) · [Релиз 0.8.0](https://pypi.org/project/google-colab-cli/0.8.0/) · [Установка WSL](https://learn.microsoft.com/en-us/windows/wsl/install)

```powershell
# PowerShell. Установка нужна только при отсутствии WSL и Ubuntu.
wsl --install -d Ubuntu
wsl --list --verbose
```

```bash
# В WSL после установки uv и Google Cloud CLI.
uv tool install --python 3.12 'google-colab-cli==0.8.0'
colab version
colab new --help
colab exec --help

gcloud auth application-default login \
  --scopes='openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory'

colab usage
colab sessions
```

Войди под своим Colab-аккаунтом. Этот вход создаёт локальные ADC для CLI. Токены, файлы `gcloud` и файлы состояния Colab CLI оставь в локальном профиле. Запиши версию и результат `colab usage` без данных учётной записи. [Установка uv](https://docs.astral.sh/uv/getting-started/installation/) · [Установка Google Cloud CLI](https://cloud.google.com/sdk/docs/install) · [Авторизация Google Colab CLI](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md#required-scopes)

### Выделить сессию и подтвердить устройство

У Google есть пример запуска T4 через официальный CLI. Доступность бесплатной T4 и время жизни сессии зависят от аккаунта и текущих лимитов. `colab usage` показывает баланс и расход на уровне аккаунта. Если CLI не выдаёт T4 на бесплатном режиме или запрашивает платные compute units, открой обычный Colab notebook, выбери T4 в настройках runtime и запускай те же файлы через ячейки `%%bash`. Сохрани отказ CLI и фактический способ запуска. В обоих случаях проверь GPU по `nvidia-smi`. [Пример Google с T4](https://developers.googleblog.com/introducing-the-google-colab-cli/) · [Colab FAQ и ограничения бесплатных ресурсов](https://research.google.com/colaboratory/faq.html)

```bash
# Локально в WSL из каталога эксперимента. Используй уникальное имя сессии.
SESSION=soup-t4-01
colab new -s "$SESSION" --gpu T4
colab status -s "$SESSION"
colab usage

colab exec -s "$SESSION" <<'PY'
from pathlib import Path
import subprocess

Path('/content/soup_takehome').mkdir(parents=True, exist_ok=True)
subprocess.run(['nvidia-smi'], check=True)
PY
```

Для нескольких этапов используй именованную сессию. `colab run` по умолчанию удаляет VM после команды, поэтому для рабочего процесса используй `new`, `exec`, `download` и `stop`. Официальный CLI передаёт Python в Jupyter kernel. На бесплатном режиме выполняй команды через kernel, учитывая ограничения Colab на SSH. [Команды Colab CLI 0.8.0](https://pypi.org/project/google-colab-cli/0.8.0/) · [Жизненный цикл `colab run`](https://github.com/googlecolab/google-colab-cli/blob/main/docs/05_run_command.md) · [Ограничения Colab](https://research.google.com/colaboratory/faq.html#limitations)

### Загрузить код и запускать по этапам

Собери архив из подготовленных скриптов и конфига. Приведённый вызов `tar` предполагает, что эти файлы уже созданы. При повторе используй тот же хэш входного архива или запиши новую ревизию.

```bash
# Локально, в WSL.
tar -czf soup-code.tar.gz soup.yaml run_phase.sh stages scripts
sha256sum soup-code.tar.gz
colab ls -s "$SESSION" content/soup_takehome
colab upload -s "$SESSION" soup-code.tar.gz content/soup_takehome/soup-code.tar.gz

colab exec -s "$SESSION" <<'PY'
import subprocess

subprocess.run(['tar', '-xzf', 'soup-code.tar.gz'],
               cwd='/content/soup_takehome', check=True)
PY

mkdir -p local-logs
run_colab_phase() {
  case "$1" in
    setup|data|preflight|baseline|bench|train|merge|quant|verify|probes|ship) ;;
    *) return 2 ;;
  esac
  local stamp
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  (
    set -o pipefail
    printf 'import subprocess\nsubprocess.run(["bash", "run_phase.sh", "%s"], cwd="/content/soup_takehome", check=True)\n' "$1" |
      colab exec -s "$SESSION" 2>&1 | tee "local-logs/$1-$stamp.log"
  )
}

run_colab_phase setup
```

Результат этапа сверяй с сохранённым на VM `status.txt`, логом и выходными файлами. При ошибке `colab exec` сначала проверь состояние сессии и наличие результата на VM. Это помогает отличить падение кода от разрыва подключения. В browser fallback распакуй тот же архив в `/content/soup_takehome` и из ячейки `%%bash` вызывай `cd /content/soup_takehome` и `bash run_phase.sh setup` или нужную фазу. [Загрузка и скачивание файлов в CLI](https://github.com/googlecolab/google-colab-cli/blob/main/docs/03_file_management.md) · [Выполнение через kernel](https://github.com/googlecolab/google-colab-cli/blob/main/docs/02_execution_and_interactive.md)

## 4. Настроить логи и среду до обучения

Минимальный `run_phase.sh`. Он сохраняет объединённый stdout/stderr, копию с UTC-метками, исходный полный `nvidia-smi` до и после фазы и отдельный непрерывный CSV. Метка строки отражает время её получения логгером. Поля CSV формирует сам `nvidia-smi`. Пустой или оборванный `status.txt` указывает на прерванную фазу.

```bash
#!/usr/bin/env bash
set -uo pipefail
export TZ=UTC PYTHONUNBUFFERED=1
cd /content/soup_takehome || exit 1
phase=${1:?phase required}
case "$phase" in
  setup|data|preflight|baseline|bench|train|merge|quant|verify|probes|ship) ;;
  *) exit 2 ;;
esac
mkdir -p logs || exit 1
dir=$(mktemp -d "logs/$(date -u +%Y%m%dT%H%M%SZ)-${phase}-XXXXXX") || exit 1
export PHASE_LOG_DIR="$PWD/$dir"
printf 'started_utc=%s\n' "$(date -u +%FT%TZ)" > "$dir/status.txt"
nvidia-smi > "$dir/nvidia-smi-before.txt" 2>&1
nvidia-smi --query-gpu=timestamp,index,name,driver_version,memory.total,memory.used,utilization.gpu \
  --format=csv -lms 200 > "$dir/nvidia-smi.csv" 2> "$dir/nvidia-smi.stderr" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT

bash -euo pipefail "stages/$phase.sh" 2>&1 | tee "$dir/raw.log" |
  awk '{ print strftime("%Y-%m-%dT%H:%M:%SZ", systime()), $0; fflush() }' |
  tee "$dir/timestamped.log"
codes=("${PIPESTATUS[@]}")
nvidia-smi > "$dir/nvidia-smi-after.txt" 2>&1
printf 'ended_utc=%s\nphase_exit=%s\npipeline_codes=%s\n' \
  "$(date -u +%FT%TZ)" "${codes[0]}" "${codes[*]}" >> "$dir/status.txt"
for code in "${codes[@]}"; do
  if (( code != 0 )); then exit "$code"; fi
done
```

В `setup.sh` запиши `python --version`, `uname -a`, `free -h`, `df -h /content`, затем установи релиз Soup и bitsandbytes. Для Soup 0.77.2 используй Python 3.11–3.13 внутри VM. Сохрани `pip freeze`, `pip check`, версию Soup, версию PyTorch/CUDA и фактическое имя GPU. Проверь импорты `torch`, `transformers`, `peft`, `trl`, `accelerate` и `bitsandbytes` в новом процессе. Если Colab сообщает конфликт пакетов, сохрани вывод и проверь, затрагивает ли он используемый стек. [Релиз Soup](https://pypi.org/project/soup-cli/0.77.2/)

```bash
# Внутри stages/setup.sh.
python -m pip install 'soup-cli[train]==0.77.2' bitsandbytes
python -m pip freeze > "$PHASE_LOG_DIR/pip-freeze.txt"
pip_check_rc=0
python -m pip check || pip_check_rc=$?
printf 'pip_check_exit=%s\n' "$pip_check_rc"
python -c 'from importlib.metadata import version; print("Soup", version("soup-cli"))'
python -c 'import torch, transformers, peft, trl, accelerate, bitsandbytes'
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
```

В `measure_soup.py` найди `console_scripts` entry point `soup` у `importlib.metadata.distribution('soup-cli')` и вызови `entry_point.load()()` с переданным `sys.argv`. Для train в этом же процессе выполни `torch.cuda.reset_peak_memory_stats()` перед стартом и запиши `max_memory_allocated()` и `max_memory_reserved()` в `finally`. Сохрани исключение и первоначальный exit code. Подтверди single-GPU запуск в логах. Для quant добавь замеры в `quantize_nf4.py`. Для CPU merge используй `/usr/bin/time -v`. Замер в отдельном Colab kernel не отражает аллокации процесса `soup train`. [Релиз Soup](https://pypi.org/project/soup-cli/0.77.2/) · [Python entry points](https://docs.python.org/3/library/importlib.metadata.html#entry-points) · [CUDA allocator PyTorch](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management) · [NVIDIA SMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

## 5. Подготовить данные и заморозить тест

Возьми Amazon MASSIVE 1.1 `ru-RU`. Это открытый NLU-корпус. Преобразуй исходные `utt` и `intent` в instruction SFT. Сохрани атрибуцию Amazon, лицензию CC BY 4.0, ссылку на оригинал и описание преобразования. По модели зафиксируй resolved commit через `HfApi().model_info(model_id).sha` и скачай этот SHA в `./base_snapshot` через `snapshot_download(repo_id=model_id, revision=sha, local_dir='./base_snapshot')`. [Архив, формат и лицензия MASSIVE](https://github.com/alexa/massive#accessing-and-processing-the-data) · [Карточка датасета](https://huggingface.co/datasets/AmazonScience/massive) · [Статья MASSIVE](https://arxiv.org/abs/2204.08582) · [Скачивание ревизии HF Hub](https://huggingface.co/docs/huggingface_hub/guides/download)

```bash
mkdir -p data/raw
curl -fL 'https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz' \
  -o data/raw/massive-1.1.tar.gz
sha256sum data/raw/massive-1.1.tar.gz
tar -tzf data/raw/massive-1.1.tar.gz | awk '/ru-RU\.jsonl$|LICENSE$/'
```

1. Извлеки из архива найденные `ru-RU.jsonl` и `LICENSE`. По исходному `partition=train` выбери примерно 12 частых классов с достаточной поддержкой, зафиксируй определения классов на русском и выбери около 100 строк на класс с `seed=42`. Цель — 1 200 train-строк. Храни `id`, исходный split, `utt` и `intent` в `data/manifest.json`.
2. Получи около 100 dev и 240–300 test из соответствующих официальных разделов. Проверь совпадения ID и нормализованного текста между сплитами. Для поиска дублей используй NFKC, `casefold` и схлопывание пробелов. Сохрани исходный текст для обучения и инференса. Запиши выбывшие примеры, реальные размеры и распределения. Для одинакового текста с разными метками сохрани конфликт и исключи неоднозначность из основной метрики.
3. Для train запиши `messages` с `system` (русская инструкция и определения кодов), `user` (исходный `utt`), `assistant` (ровно исходный `intent`). Для dev/test запиши `id`, `messages` только с system/user и отдельное поле `expected`. Используй одинаковые определения для всех моделей. Скрипт сверки должен проверять целевые ответы в JSONL по исходному manifest.
4. Рендери шаблон Qwen и сохрани распределение длин после токенизации, число примеров, которые пришлось бы обрезать, и число обучаемых токенов. Проверь худшие по длине строки. Если ответ или конец хода потеряется, укороти общую инструкцию либо пересчитай бюджет под новый лимит. Держи test закрытым до выбора конфигурации и допусков на dev.

```jsonl
{"messages":[{"role":"system","content":"Верни один код намерения из списка. alarm_set — поставить будильник; alarm_remove — удалить будильник."},{"role":"user","content":"Разбуди меня в семь утра"},{"role":"assistant","content":"alarm_set"}]}
```

Строка иллюстрирует формат. Рабочий `system` должен содержать весь зафиксированный список классов. Маска важна отдельно от содержимого JSONL. Сопоставь в нескольких строках rendered IDs, позиции `labels != -100` и соответствующие токены. Проверь обычную строку, самые длинные и батч с padding. Учитывай, что stock template Qwen завершает assistant-ход токеном `<|im_end|>`, а поведение маски зависит от реализации Soup. [Chat templates в Transformers](https://huggingface.co/docs/transformers/main/chat_templating) · [Шаблон Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/tokenizer_config.json) · [Soup Data Doctor](https://trysoup.dev/docs/fine-tune-doctor)

## 6. Сохранить расчёт памяти до GPU-прогона

Зафиксируй исходный расчёт в `results/memory-estimate.md` с временем и конфигом. Для T4 используй FP16 и сравни бюджет с реальным `memory.total`, указанным драйвером. Для Qwen при `microbatch=1`, `S=512`, 28 слоях, hidden size 1 536, KV output 256 и vocab 151 936 получаем следующие оценки. [NVIDIA T4](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf) · [Конфиг Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json) · [PEFT LoRA](https://huggingface.co/docs/peft/main/package_reference/lora) · [Loss в Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py)

| Компонент | Предварительный расчёт |
|---|---|
| Замороженные веса | `1.54e9 × 2 / 2^30 ≈ 2.87 GiB` FP16. Для базы отсутствуют градиенты и состояния оптимизатора. |
| LoRA и AdamW | `28 × 16 × [(1536+1536)+(1536+256)] = 2 179 072` параметра. При FP32 весах, градиентах и двух моментах AdamW это `N × 16 B ≈ 33 MiB`. Сверь число параметров и dtype с фактическим тренером. |
| Активации | Входы checkpoint-блоков `28 × 1 × 512 × 1536 × 2 B ≈ 42 MiB`. Добавь временные тензоры текущего блока, attention и recompute. Рабочее допущение для этой строки — 0,08–0,20 GiB при checkpointing. |
| Логиты и loss | `1 × 512 × 151936` значений, ≈148 MiB FP16 или ≈297 MiB FP32. FP16 логиты, FP32 копия и дополнительный FP32 буфер вместе дают до ≈742 MiB. Фактические копии зависят от loss и версий. |
| Пик train | Плановый диапазон устройства **5–6,5 GiB**, включая запас на CUDA context, kernels и allocator. Время жизни буферов определяет реальный одновременный пик. |
| Пик merge | CPU. Одна плотная FP16 копия ≈2,87 GiB, две ≈5,74 GiB, плюс буферы чтения, LoRA, merge и запись. На GPU ожидается фоновое потребление. Сохрани RSS и запас RAM. |
| Пик NF4 export | Связанные embeddings/head `E = 151936 × 1536`. Оценка `E × 2 B + (1.54e9 − E) × 0.5 B ≈ 1.04 GiB` до scale metadata и прочих FP16 тензоров, ориентир **1,1 GiB** после загрузки. Около **4,0 GiB** при совместной резидентности полной FP16 и NF4 копий, плюс staging/workspace. Стадирование по тензорам может снизить пик. |

После прогона добавь для train, merge и quant `expected`, sampled `nvidia-smi max`, `max_memory_allocated`, `max_memory_reserved`, CPU max RSS и разницу. `nvidia-smi` показывает память устройства в моменты выборки. `allocated` отражает живые PyTorch-аллокации, `reserved` включает кэш его аллокатора. Короткий пик может оказаться между выборками. Сверь длину реально обработанных батчей, dtype и число резидентных моделей. Объясняй остаток допущениями и замерами. Отдельно определи tok/s. `soup bench train` измеряет supervised tokens, которые у задачи с короткой меткой составляют малую часть полного входа. Для полного прогона укажи время фазы и явно определи, какие токены и какой интервал вошли в деление. [PyTorch CUDA memory](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management) · [NVIDIA SMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html) · [Bench Train](https://trysoup.dev/docs/fine-tune-doctor#soup-bench-train-a-short-run-that-fails-if-the-model-was-not-training-v0760)

## 7. Запустить Soup pre-flight, baseline и LoRA

Стартовый `soup.yaml`. На T4 после загрузки подтверди реальный dtype базы и адаптера. При 1 200 строках, одной эпохе и накоплении 8 ожидается около 150 optimizer steps. `train_on_eot: true` нужен здесь для явного обучения остановке после кода. Проверь фактическую маску и передачу этой настройки в pre-flight. [Схема Soup 0.77.2](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.2/docs/config-reference.md) · [Soup CLI Reference](https://trysoup.dev/docs/cli-reference)

```yaml
base: ./base_snapshot
task: sft
backend: transformers
output: ./artifacts/adapter

data:
  train: ./data/train.jsonl
  format: chatml
  val_split: 0.0
  max_length: 512
  streaming: false
  train_on_responses_only: true

training:
  quantization: none
  stream_layers: false
  gradient_checkpointing: true
  packing: false
  fast_lora: false
  use_flash_attn: false
  train_on_eot: true
  batch_size: 1
  gradient_accumulation_steps: 8
  epochs: 1
  lr: 0.0002
  optimizer: adamw_torch
  seed: 42
  logging_steps: 10
  save_steps: 50
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: [q_proj, v_proj]
```

Разложи команды по `stages/preflight.sh`, `stages/bench.sh` и `stages/train.sh`. В `stages/data.sh` вызови подготовку данных и снимка базы. В `stages/baseline.sh` загрузи FP16 base и запиши результаты verifier в режиме `base-dev`.

```bash
# stages/preflight.sh
soup data validate ./data/train.jsonl --format chatml --min-valid-fraction 1.0
soup data doctor ./data/train.jsonl --model ./base_snapshot --max-length 512 \
  --sample 2000 --show-mask 5 --train-on-responses-only --train-on-eot \
  --output "$PHASE_LOG_DIR/data-doctor.json"
soup doctor --config soup.yaml
soup train --config soup.yaml --dry-run --yes

# stages/bench.sh
soup bench train --config soup.yaml --steps 20 --warmup 3 \
  --output "$PHASE_LOG_DIR/bench-train.json"

# stages/train.sh
mkdir -p artifacts
python -u scripts/measure_soup.py train --config soup.yaml --yes
```

```bash
# Локально в WSL, проверяя результат каждой команды перед следующей.
run_colab_phase data
run_colab_phase preflight
run_colab_phase baseline
run_colab_phase bench
run_colab_phase train
```

Проверь число steps, trainable params, ненулевые конечные градиенты и пики в bench. `soup bench train` делает реальные optimizer steps во временном каталоге и выдаёт диагностические метрики. При non-finite или нулевом градиенте сначала проверь активные labels, параметры и dtype. При OOM проверь фактическую длину, текущие GPU-процессы и рабочие настройки. Затем при необходимости зафиксируй переход на QLoRA и пересчитай память. Успешный bench разрешает полный прогон. Доказательство вклада сохраняемого адаптера появится после отдельной проверки. [Fine-tune Doctor и правила bench](https://trysoup.dev/docs/fine-tune-doctor)

Из полного прогона сохрани `adapter_model.safetensors`, `adapter_config.json`, tokenizer, финальный step, конфиг, CUDA-замеры и список хэшей. Перед merge скачай небольшой архив с адаптером и уже собранными логами. При разрыве используй `soup train --config soup.yaml --resume auto --yes` только при наличии полного совместимого `checkpoint-*`. Зафиксируй номер шага и оставь журналы обеих попыток. [Документация train и resume](https://trysoup.dev/docs/cli-reference#soup-train)

## 8. Слить и квантизировать

Запускай train, merge и quant последовательно в отдельных процессах. В `stages/merge.sh` сохрани max RSS и дай Soup явные пути к тому же снимку и финальному адаптеру. Проверь, что есть полные модельные веса, конфиг и tokenizer. Посчитай хэши полученных файлов. Если обычный CPU merge упирается в RAM, для этого plain LoRA можно проверить `--low-memory` и зафиксировать изменение маршрута. [Флаги merge Soup 0.77.2](https://trysoup.dev/docs/cli-reference#soup-merge) · [PEFT checkpoint format](https://huggingface.co/docs/peft/main/developer_guides/checkpoint)

```bash
# stages/merge.sh
/usr/bin/time -v -o "$PHASE_LOG_DIR/cpu-time.txt" \
  soup merge --base ./base_snapshot --adapter ./artifacts/adapter \
  --output ./artifacts/merged --dtype float16
```

В `scripts/quantize_nf4.py` загрузи сохранённый FP16 merge с `local_files_only=True`, `device_map={"": 0}` и `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)`. Задай `dtype=torch.float16` для неквантованных параметров на версии Transformers, поддерживающей `dtype`. Сохрани модель и tokenizer в `artifacts/nf4` через `save_pretrained`. В метаданных укажи SHA256 файлов merge, конфиг BNB, версию пакетов, размеры и пики GPU. Укажи, что 4-битными становятся поддерживаемые линейные веса, а часть слоёв остаётся FP16. [Transformers BitsandbytesConfig](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes) · [Статья QLoRA о NF4](https://arxiv.org/abs/2305.14314)

В отдельном процессе снова открой `artifacts/nf4` через `AutoModelForCausalLM.from_pretrained` по локальному пути и **с сохранённым quantization config**. Проверь `is_loaded_in_4bit`, наличие `bitsandbytes.nn.Linear4bit` и NF4 `quant_state`, фактическое размещение, совпадение tokenizer и наличие самостоятельных весов. Проверка должна отвергать FP16-каталог, переданный как NF4. Верификатор должен прочитать именно этот сохранённый каталог. [Документация Transformers по квантованным моделям](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes)

```bash
# Локально в WSL. В stages/quant.sh вызывается scripts/quantize_nf4.py.
run_colab_phase merge
run_colab_phase quant
```

## 9. Написать и запустить `verify.py`

Скрипт принимает пути к базе, адаптеру, merged и NF4, файлы dev/test, manifest и зафиксированные пороги. В начале проверяет входные SHA256. На выходе пишет JSON с числовыми метриками и причинами решения, JSONL с результатом каждого примера и краткий текстовый итог. Сохраняй те же результаты в контрольных прогонах. Для экономии памяти загружай модели по очереди и сохраняй малые массивы результатов на CPU.

**Активность адаптера.** Проверь пары `lora_A/lora_B` для `q_proj/v_proj`, норму конечной эффективной дельты `(alpha/r) × B @ A` и число слоёв с ненулевой дельтой. На фиксированном dev-наборе сравни `adapter-on` и `adapter-off` через `PeftModel.disable_adapter()` в `eval()` и `torch.inference_mode()`. В качестве сигнала возьми распределение абсолютной разницы log-prob целевых токенов. Чтобы оценить численный шум, повтори один и тот же режим и прогони свежий LoRA с нулевой B. Изменения, сравнимые с этими контролями, не дают доказательства вклада адаптера. Если обучал через QLoRA, сравни on/off на одной NF4 базе. Для проверки merge отдельно загрузи этот адаптер на исходной FP16 базе. [PEFT LoRA и инициализация](https://huggingface.co/docs/peft/main/package_reference/lora) · [PEFT `disable_adapter()`](https://huggingface.co/docs/peft/main/package_reference/peft_model#peft.PeftModel.disable_adapter)

**Паритет merge.** Сравни `FP16 base + adapter-on` и перезагруженный FP16 merge на одинаковых токенах. Для teacher forcing отрендери полную беседу с эталонным ответом, сверь границу assistant-ответа с рендером prompt, возьми `log_softmax(logits.float())` и собери `log p(token_i | tokens_<i)` из позиции `i-1`. Считай только токены метки, отдельно логируй EOT, длину и отсутствие обрезки. Сохрани median, p95 и max абсолютной разницы, greedy agreement и первые расхождения. На dev зафиксируй допуск, который покрывает FP16-округление и отделяет merge от заведомого base. Передай базу вместо merge и сохрани результат `FAIL` по численному или структурному критерию. Допуск и отказ должны работать на независимых примерах. [Шаблоны Transformers](https://huggingface.co/docs/transformers/main/chat_templating) · [PEFT merge](https://huggingface.co/docs/peft/main/package_reference/lora)

**Качество артефакта.** На test подай только system/user через tokenizer каждого сохранённого артефакта с `add_generation_prompt=True`. Сравни токены и спецтокены с базовым tokenizer. Установи единые `do_sample=False`, `num_beams=1`, `max_new_tokens` (например, 16 после аудита кодов), одинаковые batch и stop IDs. Декодируй только новые токены, сохрани сырой текст, token IDs и признак `EOS` или достижения лимита. Правило основного счёта — после удаления внешних пробелов строка целиком совпадает с одним допустимым кодом. Для `base / merged / nf4` запиши exact match, valid-format rate, macro-F1, число строк и ошибки по классам. Для разниц `merged − base` и `NF4 − merged` посчитай парный bootstrap CI по тем же ID, например 5 000 ресемплов с фиксированным seed. При возможности добавь NF4 base как контроль изменений от одной квантизации. [Парный bootstrap SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) · [Генерация Transformers](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

**Почему нужен отдельный скрипт.** Усреднённый loss меняется при разной сложности батчей и разных масках, даже если нужный артефакт остался базой. На train он также падает при запоминании частых ответов. Проверки выше дают отдельное свидетельство вклада адаптера, паритета merge и качества на holdout. Границы проверки — узкий набор классов и фиксированный runtime. Она оставляет открытыми contamination из pretrain, качество вне домена, другие языки, работу под нагрузкой и долгосрочную стабильность. Небольшой тест может дать широкий интервал оценки.

```text
Реальный вывод должен содержать поля такого вида
adapter_effect  <median/p95 on-off>  <repeat/no-op floor>  <status>
merge_parity    <median/p95 on-merge>  <tolerance>         <status>
base            <n>  <exact match>  <valid format>  <macro-F1>
merged          <n>  <exact match>  <valid format>  <macro-F1>
nf4             <n>  <exact match>  <valid format>  <macro-F1>
negative_base_as_merge   <exit code and failed check>
negative_dense_as_nf4    <exit code and failed check>
```

```bash
run_colab_phase verify
run_colab_phase probes
```

## 10. Воспроизвести тихие сбои и проверить Soup

В `stages/probes.sh` создай контрольные копии, сохрани их хэши и запусти команды с ожидаемыми отказами. Код ожидаемого отказа записывай отдельной строкой. Исходные артефакты оставляй доступными для повторной проверки. Для каждого опыта укажи, что предсказал, что команда реально увидела и ссылку на журнал. В таблице ниже перечислены гипотезы, которые предстоит проверить.

| Звено | Контрольный опыт | Какие свидетельства собрать |
|---|---|---|
| Данные | Подменить один целевой класс на другой существующий в копии train. | `data doctor` проверяет структуру и токены. Сверка по исходному manifest выявляет смысловую подмену. Записать оба результата. |
| Tokenizer и шаблон | Удлинить один user-ввод до обрезки ответа. | Записать предупреждения doctor, выход `--show-mask` и фактические supervised IDs после collator. Убедиться, что в штатных данных код и нужный EOT участвуют в loss. |
| Train | Подать свежий LoRA с нулевой B или оценить обученный с выключенным адаптером. | `bench train` проверяет обновления своей короткой тренировки. В Soup 0.77.2 live `ship --adapter` может отклонить локальный safetensors с нулевой B правилом `adapter_noop`. Verifier фиксирует нулевой сигнал и отсутствие вклада. |
| Merge | Подставить base вместо `artifacts/merged`. | Base загружается как валидная модель. Сравнение с adapter-on должно обнаружить потерю сигнала. Записать фактический результат `soup ship --tuned` или причину, по которой проверка неприменима. |
| Квантизация | Подставить FP16 merge вместо NF4. | Проверить формат и число квантованных модулей после reload. Результат `soup ship` для dense checkpoint относится к dense checkpoint. |
| Инференс | В копии убрать assistant generation prompt или сменить EOS. | Сравнить входные токены, причину остановки и формат ответов. Зафиксировать, какой prompt и runtime использовал каждый из инструментов. |

Для `soup ship` сгенерируй `data/ship-dev.jsonl` из dev. Каждая строка содержит `prompt` с инструкцией, классами и запросом, `expected` с кодом и `scoring: exact`. Это вспомогательное сравнение. Встроенный scorer и способ рендера могут отличаться от verifier, поэтому запиши, какие именно входы он получил. Сделай live-проверку dense merged против базы с общим suite. Отдельно подай локальный свежий адаптер в `--adapter` и проверь, что правило `adapter_noop` сработало. Коды `ship` в 0.77.2 — `0` SHIP, `2` DON'T SHIP, `3` ошибка аргументов или входа, `1` runtime error. [Soup Ship и `adapter_noop`](https://trysoup.dev/docs/soup-ship) · [Описание формата custom eval](https://github.com/MakazhanAlpamys/Soup/blob/main/docs/evaluation.md#custom-eval-format) · [Data Doctor](https://trysoup.dev/docs/fine-tune-doctor)

```bash
# stages/ship.sh, из /content/soup_takehome
soup ship --base ./base_snapshot --tuned ./artifacts/merged \
  --task-eval ./data/ship-dev.jsonl --task-mode metric \
  --output "$PHASE_LOG_DIR/ship-merged.json"
```

Для финального отчёта по каждому сбою заполни `data doctor / pre-flight / soup ship / verifier`. Используй фактические категории `поймал`, `пропустил`, `неприменимо`, `ошибка запуска` с объяснением. Отдельно запиши verdict встроенного инструмента и фактическое качество перезагруженного NF4.

## 11. Забрать результаты и остановить Colab

Архивируй логи и адаптер сразу после train, затем отдельным архивом после проверки. Команды ниже предполагают, что `data`, `results`, `artifacts/adapter` и логи уже существуют. При сбое архивируй то, что удалось сохранить. Храни каждый экспорт под новым именем. Перед передачей проверь содержимое архива и отсутствие токенов.

```bash
run_colab_phase ship

# Локально в WSL. Создать архив на VM и вывести его уникальное имя.
colab exec -s "$SESSION" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import subprocess

root = Path('/content/soup_takehome')
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
name = f'evidence-{stamp}.tar.gz'
subprocess.run(['tar', '-czf', name, '--exclude=data/raw',
                '--exclude=artifacts/adapter/checkpoint-*', 'logs', 'results',
                'data', 'soup.yaml', 'scripts', 'stages',
                'run_phase.sh', 'artifacts/adapter'], cwd=root, check=True)
with (root / f'{name}.sha256').open('w') as checksum:
    subprocess.run(['sha256sum', name], cwd=root,
                   stdout=checksum, check=True)
print(name)
PY

mkdir -p retrieved
read -r -p 'Имя архива из вывода Colab: ' archive
colab download -s "$SESSION" "content/soup_takehome/$archive" "retrieved/$archive"
colab download -s "$SESSION" "content/soup_takehome/$archive.sha256" "retrieved/$archive.sha256"
(cd retrieved && sha256sum -c "$archive.sha256" && tar -tzf "$archive")
colab log -s "$SESSION" -o "retrieved/${archive%.tar.gz}-session.ipynb"
```

Для весов NF4 заранее проверь возможность скачивания. Colab CLI использует Jupyter Contents API, где крупные бинарные файлы могут быть неудобны. Пример ниже выгружает финальный артефакт частями по 32 MiB. При лимите конкретной передачи уменьши размер части. В браузерном Colab при доступном Google Drive можно скопировать туда артефакт штатным способом. Если нужен resume после потери VM, сохрани последний целый `checkpoint-*` отдельно. [Colab file operations](https://github.com/googlecolab/google-colab-cli/blob/main/docs/03_file_management.md)

```bash
colab exec -s "$SESSION" <<'PY'
import subprocess

subprocess.run(['bash', '-euc', '''
mkdir delivery
tar -cf delivery/nf4.tar artifacts/nf4
cd delivery
sha256sum nf4.tar > nf4.tar.sha256
split -b 32M -d -a 4 nf4.tar nf4.part-
printf '%s\\n' nf4.part-* > nf4.parts.txt
'''], cwd='/content/soup_takehome', check=True)
PY

mkdir -p retrieved/nf4-parts
(
  set -e
  cd retrieved/nf4-parts
  remote=content/soup_takehome/delivery
  colab download -s "$SESSION" "$remote/nf4.parts.txt" nf4.parts.txt
  colab download -s "$SESSION" "$remote/nf4.tar.sha256" nf4.tar.sha256
  while IFS= read -r part; do
    [[ $part =~ ^nf4\.part-[0-9]{4}$ ]] || exit 1
    colab download -s "$SESSION" "$remote/$part" "$part"
  done < nf4.parts.txt
  while IFS= read -r part; do cat -- "$part"; done < nf4.parts.txt > nf4.tar
  sha256sum -c nf4.tar.sha256
)
```

После проверки локальных копий освободи VM. [Сессии и остановка Colab](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md)

```bash
colab stop -s "$SESSION"
colab sessions
```

## 12. Вынести один вердикт и собрать сдачу

Зафиксируй критерии до открытия test в `results/acceptance.json`. Ниже стартовое правило для автоматической маршрутизации запросов из выбранных классов. При корректировке по dev запиши причину и новую версию до финального теста.

- Verifier подтверждает вклад LoRA, паритет merge и 4-битную загрузку. Контрольные подстановки воспроизводимо отклоняются.
- Merged улучшает exact match относительно base минимум на **5 процентных пунктов**, нижняя граница парного 95% CI разницы выше нуля.
- Перезагруженный NF4 выдаёт минимум **99%** валидных ответов, достигает **90%** exact match и уступает merged максимум **3 процентных пункта**. Укажи интервал для потери после квантизации. Если данные совместимы с существенным ухудшением, опиши неопределённость и выбери `DON’T SHIP`.
- Прогон воспроизводится на T4 в заявленном режиме с измеренным запасом памяти. Функция и границы целевого runtime названы явно.

Если какой-либо выпускной критерий нарушен, выбери `DON’T SHIP`. Назови ближайшее изменение и конкретный повторный тест. Например, при потере паритета исправь связку base revision/adapter и повтори проверку merge. При потере качества NF4 оцени другую схему квантизации на dev и повтори сравнение на свежем holdout. Если нижняя граница улучшения не проходит порог, сначала собери дополнительную независимую оценку или улучши данные на train/dev.

**Отчёт до двух страниц.** В начале укажи вердикт и целевой сценарий. На первой странице помести выбор стека, таблицу прогноз/факт VRAM по трём фазам и сравнение артефактов. На второй помести доказательство вклада адаптера, ключевые тихие сбои и какие проверки их поймали, ограничения, абзац о неожиданном и о рисках, короткую заметку об ИИ.

**Пакет для ответа на письмо `team@trysoup.dev`**

- Отчёт до двух страниц.
- Сырые stdout/stderr, timestamped логи, исходные выводы `nvidia-smi`, CSV и логи неудачных попыток.
- `verify.py`, команды запуска, per-example результаты и таблица метрик.
- Готовый notebook или код со всеми скриптами, конфигом, версиями и инструкцией запуска.
- Абзац о том, что удивило и что беспокоит. В заметке об ИИ укажи, где использовал его, какие предложения принял и что проверил, изменил или переделал самостоятельно.

Для 40-минутного обсуждения сохрани конкретные логи и строки кода, к которым можно вернуться. Задание просит разбирать ход эксперимента в разговоре.

## Указатель источников

- Google Colab CLI, версии и режимы запуска. [Анонс Google](https://developers.googleblog.com/introducing-the-google-colab-cli/) · [Официальный репозиторий](https://github.com/googlecolab/google-colab-cli) · [PyPI 0.8.0](https://pypi.org/project/google-colab-cli/0.8.0/) · [Colab FAQ](https://research.google.com/colaboratory/faq.html) · [Авторизация CLI](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md) · [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
- Soup. [Релиз 0.77.2](https://pypi.org/project/soup-cli/0.77.2/) · [Схема конфигурации](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.2/docs/config-reference.md) · [CLI Reference](https://trysoup.dev/docs/cli-reference) · [Fine-tune Doctor](https://trysoup.dev/docs/fine-tune-doctor) · [Soup Ship](https://trysoup.dev/docs/soup-ship) · [Layer streaming](https://trysoup.dev/docs/layer-streaming)
- Модель и данные. [Карточка Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) · [Qwen config](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json) · [Qwen tokenizer](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/tokenizer_config.json) · [MASSIVE](https://github.com/alexa/massive) · [Карточка MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) · [Статья MASSIVE](https://arxiv.org/abs/2204.08582)
- LoRA, измерения и оценка. [PEFT LoRA](https://huggingface.co/docs/peft/main/package_reference/lora) · [QLoRA и NF4](https://arxiv.org/abs/2305.14314) · [Transformers bitsandbytes](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes) · [Chat templates](https://huggingface.co/docs/transformers/main/chat_templating) · [NVIDIA T4](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf) · [NVIDIA SMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html) · [PyTorch CUDA memory](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management) · [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
