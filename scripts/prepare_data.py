#!/usr/bin/env python3
"""
Prepare Amazon MASSIVE 1.1 (ru-RU) instruction dataset and download pinned base model snapshot.
Follows Section 5 of soup-take-home-colab-cli-plan.md.
"""

import argparse
import hashlib
import json
import os
import random
import tarfile
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Concise Russian definitions for common MASSIVE 1.1 intents.
INTENT_DEFINITIONS_RU: Dict[str, str] = {
    "alarm_query": "проверить установленные будильники",
    "alarm_remove": "удалить или отключить будильник",
    "alarm_set": "поставить будильник",
    "audio_volume_down": "уменьшить громкость звука",
    "audio_volume_mute": "выключить звук",
    "audio_volume_up": "увеличить громкость звука",
    "calendar_query": "узнать события и расписание в календаре",
    "calendar_remove": "удалить событие из календаря",
    "calendar_set": "создать событие или напоминание в календаре",
    "cooking_recipe": "найти рецепт или совет по готовке",
    "datetime_convert": "перевести время между часовыми поясами",
    "datetime_query": "узнать текущее время или дату",
    "email_query": "проверить входящие письма",
    "email_querycontact": "найти контактные данные для почты",
    "email_sendemail": "отправить электронное письмо",
    "general_greet": "поприветствовать ассистента",
    "general_joke": "попросить рассказать шутку",
    "general_quirky": "нестандартная болтовня или общий разговорный запрос",
    "iot_cleaning": "управлять роботом-пылесосом или уборкой",
    "iot_coffee": "управлять кофеваркой",
    "iot_hue_lightchange": "изменить цвет или режим освещения",
    "iot_hue_lightdim": "приглушить свет",
    "iot_hue_lightoff": "выключить свет",
    "iot_hue_lighton": "включить свет",
    "iot_hue_lightup": "сделать свет ярче",
    "iot_wemo_off": "выключить умную розетку или устройство",
    "iot_wemo_on": "включить умную розетку или устройство",
    "lists_createoradd": "создать список или добавить пункт в список",
    "lists_query": "посмотреть содержимое списка",
    "lists_remove": "удалить пункт или список",
    "music_likeness": "поставить оценку или отметить понравившийся трек",
    "music_query": "узнать информацию о музыке или исполнителе",
    "music_settings": "изменить настройки воспроизведения музыки",
    "news_query": "узнать новости",
    "play_audiobook": "включить аудиокнигу",
    "play_game": "запустить игру",
    "play_music": "включить музыку или трек",
    "play_podcasts": "включить подкаст",
    "play_radio": "включить радио",
    "qa_currency": "конвертировать валюту или узнать курс",
    "qa_definition": "узнать определение слова или понятия",
    "qa_factoid": "ответить на фактологический вопрос",
    "qa_maths": "решить математический пример",
    "qa_stock": "узнать котировки акций",
    "recommendation_events": "порекомендовать мероприятия или события",
    "recommendation_locations": "порекомендовать места, кафе или заведения",
    "recommendation_movies": "порекомендовать фильм",
    "social_post": "опубликовать пост в соцсети",
    "social_query": "проверить ленту или уведомления в соцсети",
    "takeaway_order": "заказать еду с доставкой или навынос",
    "takeaway_query": "узнать статус заказа еды",
    "transport_query": "узнать расписание транспорта или маршрут",
    "transport_taxi": "вызвать такси",
    "transport_ticket": "купить или проверить билет",
    "transport_traffic": "узнать ситуацию с пробками на дорогах",
    "weather_query": "узнать прогноз погоды",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(text: str) -> str:
    nfkc = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(nfkc.split())


def download_massive(raw_dir: Path) -> Tuple[Path, Path, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "massive-1.1.tar.gz"
    if not archive_path.exists():
        print(f"Downloading {MASSIVE_URL} -> {archive_path}")
        urllib.request.urlretrieve(MASSIVE_URL, archive_path)
    else:
        print(f"Using existing archive: {archive_path}")

    ru_path = raw_dir / "ru-RU.jsonl"
    license_path = raw_dir / "LICENSE"
    if not ru_path.exists() or not license_path.exists():
        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf.getmembers():
                if member.name.endswith("ru-RU.jsonl"):
                    src = tf.extractfile(member)
                    assert src is not None
                    ru_path.write_bytes(src.read())
                elif member.name.endswith("/LICENSE") or member.name == "LICENSE":
                    src = tf.extractfile(member)
                    assert src is not None
                    license_path.write_bytes(src.read())

    if not ru_path.exists():
        raise FileNotFoundError("ru-RU.jsonl not found in MASSIVE archive")
    return archive_path, ru_path, license_path


def build_system_prompt(selected_intents: List[str]) -> str:
    defs = []
    for intent in selected_intents:
        desc = INTENT_DEFINITIONS_RU.get(intent, intent.replace("_", " "))
        defs.append(f"{intent} — {desc}")
    return (
        "Определи намерение пользователя и верни ровно один код из списка без пояснений.\n"
        "Допустимые коды:\n" + "\n".join(defs)
    )


def sample_balanced(
    rows: List[Dict[str, Any]],
    intents: List[str],
    target_total: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    by_intent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_intent[r["intent"]].append(r)

    per_class = max(1, target_total // len(intents))
    selected: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []

    for intent in intents:
        pool = list(by_intent[intent])
        rng.shuffle(pool)
        selected.extend(pool[:per_class])
        leftovers.extend(pool[per_class:])

    if len(selected) < target_total and leftovers:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: target_total - len(selected)])

    rng.shuffle(selected)
    return selected[:target_total]


def download_base_snapshot(model_id: str, local_dir: Path) -> str:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    info = api.model_info(model_id)
    sha = info.sha
    if not sha:
        raise RuntimeError(f"Could not resolve commit SHA for {model_id}")
    print(f"Downloading {model_id} @ {sha} -> {local_dir}")
    snapshot_download(repo_id=model_id, revision=sha, local_dir=str(local_dir))
    return sha


def audit_token_lengths(
    train_rows: List[Dict[str, Any]],
    dev_rows: List[Dict[str, Any]],
    model_dir: Path,
    max_length: int,
) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)

    def summarize(lengths: List[int]) -> Dict[str, Any]:
        s = sorted(lengths)
        n = len(s)
        if n == 0:
            return {"count": 0}
        return {
            "count": n,
            "min": s[0],
            "p50": s[int(0.50 * (n - 1))],
            "p95": s[int(0.95 * (n - 1))],
            "p99": s[int(0.99 * (n - 1))],
            "max": s[-1],
            "truncated_at_max_length": sum(1 for x in s if x > max_length),
        }

    def extract_ids(res: Any) -> List[int]:
        """Унифицирует выход apply_chat_template: в transformers 5.22+ по умолчанию возвращается BatchEncoding (dict), а не list[int]."""
        if isinstance(res, list):
            return res
        if hasattr(res, "get") or isinstance(res, dict):
            ids = res["input_ids"]
            return list(ids[0]) if ids and isinstance(ids[0], list) else list(ids)
        return list(res)

    # Считаем реальную длину токенов (prompt + ответ) и отдельно длину supervised-ответа,
    # чтобы заранее убедиться: max_length=512 не обрежет ни один ответ (truncation=0).
    train_full_lens = []
    train_resp_lens = []
    for row in train_rows:
        msgs = row["messages"]
        full_ids = extract_ids(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, return_dict=False))
        prompt_ids = extract_ids(tokenizer.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True, return_dict=False))
        train_full_lens.append(len(full_ids))
        train_resp_lens.append(max(0, len(full_ids) - len(prompt_ids)))

    dev_prompt_lens = []
    for row in dev_rows:
        prompt_ids = extract_ids(tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True, return_dict=False))
        dev_prompt_lens.append(len(prompt_ids))

    return {
        "model_dir": str(model_dir),
        "max_length": max_length,
        "train_full_tokens": summarize(train_full_lens),
        "train_response_tokens": summarize(train_resp_lens),
        "dev_prompt_tokens": summarize(dev_prompt_lens),
        "total_supervised_tokens_in_train": sum(train_resp_lens),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MASSIVE ru-RU splits and base snapshot.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--dev-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=264)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--skip-model-download", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    archive_path, ru_path, license_path = download_massive(raw_dir)
    archive_sha256 = sha256_file(archive_path)
    ru_sha256 = sha256_file(ru_path)

    all_rows = [json.loads(line) for line in ru_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_pool = [r for r in all_rows if r.get("partition") == "train"]
    dev_pool = [r for r in all_rows if r.get("partition") == "dev"]
    test_pool = [r for r in all_rows if r.get("partition") == "test"]

    # Pick top-N frequent intents from train that have >= train_per_class examples
    counts = Counter(r["intent"] for r in train_pool)
    eligible = [intent for intent, cnt in counts.most_common() if cnt >= args.train_per_class]
    selected_intents = sorted(eligible[: args.num_classes])
    if len(selected_intents) < args.num_classes:
        raise RuntimeError(f"Only {len(selected_intents)} intents have >= {args.train_per_class} train examples.")

    selected_set = set(selected_intents)
    system_prompt = build_system_prompt(selected_intents)

    # Ищем неоднозначные тексты (один и тот же нормализованный текст с разными метками класса)
    # и полностью удаляем их из обучения и оценки, чтобы не учить модель противоречиям.
    norm_to_intents: Dict[str, set] = defaultdict(set)
    for r in all_rows:
        if r.get("intent") in selected_set:
            norm_to_intents[normalize_text(r["utt"])].add(r["intent"])
    conflicting_texts = {t for t, intents in norm_to_intents.items() if len(intents) > 1}

    # Отбираем train: строго 100 уникальных нормализованных фраз на класс, без конфликтов и дублей
    train_selected: List[Dict[str, Any]] = []
    seen_train_ids = set()
    seen_train_norms = set()
    dropped_records: List[Dict[str, Any]] = []

    for intent in selected_intents:
        candidates = [r for r in train_pool if r["intent"] == intent]
        rng.shuffle(candidates)
        taken = 0
        for r in candidates:
            norm = normalize_text(r["utt"])
            if norm in conflicting_texts:
                dropped_records.append({"id": r["id"], "partition": "train", "reason": "label_conflict"})
                continue
            if norm in seen_train_norms or r["id"] in seen_train_ids:
                dropped_records.append({"id": r["id"], "partition": "train", "reason": "duplicate_in_train"})
                continue
            seen_train_ids.add(r["id"])
            seen_train_norms.add(norm)
            train_selected.append(r)
            taken += 1
            if taken >= args.train_per_class:
                break
        if taken < args.train_per_class:
            raise RuntimeError(f"Intent {intent} only yielded {taken} clean train examples.")

    rng.shuffle(train_selected)

    def filter_eval_pool(pool: List[Dict[str, Any]], split_name: str, seen_norms: set, seen_ids: set) -> List[Dict[str, Any]]:
        clean = []
        for r in pool:
            if r["intent"] not in selected_set:
                continue
            norm = normalize_text(r["utt"])
            if norm in conflicting_texts:
                dropped_records.append({"id": r["id"], "partition": split_name, "reason": "label_conflict"})
                continue
            if r["id"] in seen_ids or norm in seen_norms:
                dropped_records.append({"id": r["id"], "partition": split_name, "reason": "overlap_or_duplicate"})
                continue
            seen_ids.add(r["id"])
            seen_norms.add(norm)
            clean.append(r)
        return clean

    seen_eval_norms = set(seen_train_norms)
    seen_eval_ids = set(seen_train_ids)
    dev_clean = filter_eval_pool(dev_pool, "dev", seen_eval_norms, seen_eval_ids)
    test_clean = filter_eval_pool(test_pool, "test", seen_eval_norms, seen_eval_ids)

    dev_selected = sample_balanced(dev_clean, selected_intents, min(args.dev_size, len(dev_clean)), rng)
    test_selected = sample_balanced(test_clean, selected_intents, min(args.test_size, len(test_clean)), rng)

    # Write JSONL files
    train_jsonl = data_dir / "train.jsonl"
    dev_jsonl = data_dir / "dev.jsonl"
    test_jsonl = data_dir / "test.jsonl"
    ship_dev_jsonl = data_dir / "ship-dev.jsonl"

    with train_jsonl.open("w", encoding="utf-8") as f:
        for r in train_selected:
            obj = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": r["utt"]},
                    {"role": "assistant", "content": r["intent"]},
                ]
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    for path, split_rows in [(dev_jsonl, dev_selected), (test_jsonl, test_selected)]:
        with path.open("w", encoding="utf-8") as f:
            for r in split_rows:
                obj = {
                    "id": str(r["id"]),
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": r["utt"]},
                    ],
                    "expected": r["intent"],
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with ship_dev_jsonl.open("w", encoding="utf-8") as f:
        for r in dev_selected:
            obj = {
                "prompt": f"{system_prompt}\n\nЗапрос пользователя: {r['utt']}",
                "expected": r["intent"],
                "scoring": "exact",
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    model_sha = None
    model_dir = Path("base_snapshot")
    if not args.skip_model_download:
        model_sha = download_base_snapshot(MODEL_ID, model_dir)
        token_audit = audit_token_lengths(
            [json.loads(l) for l in train_jsonl.read_text(encoding="utf-8").splitlines()],
            [json.loads(l) for l in dev_jsonl.read_text(encoding="utf-8").splitlines()],
            model_dir,
            args.max_length,
        )
        (results_dir / "data-token-audit.json").write_text(
            json.dumps(token_audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("Token audit saved to results/data-token-audit.json")

    manifest = {
        "source": {
            "dataset": "Amazon MASSIVE 1.1 (ru-RU)",
            "url": MASSIVE_URL,
            "license": "CC BY 4.0",
            "license_file": str(license_path),
            "archive_sha256": archive_sha256,
            "ru_jsonl_sha256": ru_sha256,
        },
        "model": {
            "model_id": MODEL_ID,
            "resolved_sha": model_sha,
            "local_dir": str(model_dir),
        },
        "seed": args.seed,
        "selected_intents": selected_intents,
        "intent_definitions_ru": {k: INTENT_DEFINITIONS_RU.get(k, k) for k in selected_intents},
        "system_prompt": system_prompt,
        "splits": {
            "train": {
                "count": len(train_selected),
                "sha256": sha256_file(train_jsonl),
                "class_distribution": dict(Counter(r["intent"] for r in train_selected)),
                "ids": [str(r["id"]) for r in train_selected],
            },
            "dev": {
                "count": len(dev_selected),
                "sha256": sha256_file(dev_jsonl),
                "class_distribution": dict(Counter(r["intent"] for r in dev_selected)),
                "ids": [str(r["id"]) for r in dev_selected],
            },
            "test": {
                "count": len(test_selected),
                "sha256": sha256_file(test_jsonl),
                "class_distribution": dict(Counter(r["intent"] for r in test_selected)),
                "ids": [str(r["id"]) for r in test_selected],
            },
        },
        "conflicting_normalized_texts_count": len(conflicting_texts),
        "dropped_records_count": len(dropped_records),
        "dropped_records_sample": dropped_records[:50],
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[OK] Prepared train={len(train_selected)}, dev={len(dev_selected)}, "
        f"test={len(test_selected)} across {len(selected_intents)} intents."
    )


if __name__ == "__main__":
    main()
