#!/usr/bin/env python3
"""
Blind 4-model evaluation across 3 independent topics (100 + 100 + 100 = 300):
  - Topic 1 (in_domain_productivity): 6 trained work/calendar/email/time classes
  - Topic 2 (in_domain_media_facts):  6 trained media/weather/news/facts/chat classes
  - Topic 3 (unseen_daily_domains):   6 UNSEEN classes not present in training

Evaluates all 4 models on the exact same 300 prompts (no cherry-picking):
  1. base
  2. adapter (base + LoRA adapter)
  3. merged (FP16 merged model)
  4. nf4 (4-bit bitsandbytes quantized model)
"""

import argparse
import csv
import gc
import json
import random
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TOPIC1_INTENTS = [
    "calendar_query",
    "calendar_remove",
    "calendar_set",
    "datetime_query",
    "email_query",
    "email_sendemail",
]

TOPIC2_INTENTS = [
    "play_music",
    "play_radio",
    "news_query",
    "weather_query",
    "qa_factoid",
    "general_quirky",
]

TOPIC3_UNSEEN_INTENTS = [
    "alarm_set",
    "alarm_query",
    "iot_hue_lightoff",
    "transport_query",
    "cooking_recipe",
    "takeaway_order",
]

UNSEEN_DEFS_RU = {
    "alarm_set": "поставить будильник",
    "alarm_query": "проверить установленные будильники",
    "iot_hue_lightoff": "выключить свет",
    "transport_query": "узнать расписание транспорта или маршрут",
    "cooking_recipe": "найти рецепт или совет по готовке",
    "takeaway_order": "заказать еду с доставкой или навынос",
}


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_prompt(intents: List[str], defs_map: Dict[str, str]) -> str:
    lines = [f"{k} — {defs_map[k]}" for k in intents]
    return (
        "Определи намерение пользователя и верни ровно один код из списка без пояснений.\n"
        "Допустимые коды:\n" + "\n".join(lines)
    )


def sample_topic(
    pool: List[Dict[str, Any]],
    intents: List[str],
    n_target: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    by_intent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in pool:
        if row["intent"] in intents:
            by_intent[row["intent"]].append(row)

    per_class = n_target // len(intents)
    chosen: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []
    for intent in intents:
        rows = list(by_intent[intent])
        rows.sort(key=lambda x: str(x.get("id", "")))
        rng.shuffle(rows)
        chosen.extend(rows[:per_class])
        leftovers.extend(rows[per_class:])

    if len(chosen) < n_target:
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: n_target - len(chosen)])

    rng.shuffle(chosen)
    return chosen[:n_target]


def build_blind_dataset(
    ru_jsonl_path: Path,
    manifest_path: Path,
    train_path: Path,
    dev_path: Path,
    out_path: Path,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Формирует слепую выборку из 300 примеров (3 темы × 100) с жёсткой защитой от утечек:
    любая нормализованная фраза, встречавшаяся в train.jsonl или dev.jsonl, полностью исключается.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trained_prompt = manifest["system_prompt"]
    trained_intents = list(manifest["selected_intents"])

    # Собираем чёрный список всех текстов из train и dev
    forbidden_texts: Set[str] = set()
    for p in (train_path, dev_path):
        for r in load_jsonl(p):
            user_txt = r["messages"][-1]["content"]
            forbidden_texts.add(normalize_text(user_txt))

    # Приоритет отдаём строкам из test/dev-разделов исходного MASSIVE, добирая при необходимости из остальных
    raw_rows = load_jsonl(ru_jsonl_path)
    seen_texts: Set[str] = set()
    clean_pool_test_first: List[Dict[str, Any]] = []
    clean_pool_other: List[Dict[str, Any]] = []

    for r in raw_rows:
        norm = normalize_text(r["utt"])
        if norm in forbidden_texts or norm in seen_texts:
            continue
        seen_texts.add(norm)
        if r.get("partition") in ("test", "dev"):
            clean_pool_test_first.append(r)
        else:
            clean_pool_other.append(r)

    rng = random.Random(seed + 777)

    # For trained classes, prefer test/dev partitions; top up from unused rows if needed.
    t1_pool = [r for r in clean_pool_test_first if r["intent"] in TOPIC1_INTENTS]
    if len(t1_pool) < 100:
        t1_pool += [r for r in clean_pool_other if r["intent"] in TOPIC1_INTENTS]
    t2_pool = [r for r in clean_pool_test_first if r["intent"] in TOPIC2_INTENTS]
    if len(t2_pool) < 100:
        t2_pool += [r for r in clean_pool_other if r["intent"] in TOPIC2_INTENTS]

    # Unseen intents were never in training at all; still prefer test/dev first.
    t3_pool = [r for r in clean_pool_test_first if r["intent"] in TOPIC3_UNSEEN_INTENTS]
    if len(t3_pool) < 100:
        t3_pool += [r for r in clean_pool_other if r["intent"] in TOPIC3_UNSEEN_INTENTS]

    topic1_rows = sample_topic(t1_pool, TOPIC1_INTENTS, 100, rng)
    topic2_rows = sample_topic(t2_pool, TOPIC2_INTENTS, 100, rng)
    topic3_rows = sample_topic(t3_pool, TOPIC3_UNSEEN_INTENTS, 100, rng)

    unseen_prompt = build_prompt(TOPIC3_UNSEEN_INTENTS, UNSEEN_DEFS_RU)

    dataset: List[Dict[str, Any]] = []
    specs = [
        ("topic1_work_comms", trained_prompt, trained_intents, True, topic1_rows),
        ("topic2_media_facts", trained_prompt, trained_intents, True, topic2_rows),
        ("topic3_unseen_daily", unseen_prompt, TOPIC3_UNSEEN_INTENTS, False, topic3_rows),
    ]
    for topic_name, sys_prompt, allowed, is_trained_topic, rows in specs:
        for r in rows:
            dataset.append(
                {
                    "topic": topic_name,
                    "is_trained_topic": is_trained_topic,
                    "id": str(r["id"]),
                    "partition_src": r.get("partition", ""),
                    "user_text": r["utt"],
                    "expected": r["intent"],
                    "allowed_labels": allowed,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": r["utt"]},
                    ],
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return dataset


def compute_macro_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(sum(f1s) / len(labels)) if labels else 0.0


def paired_bootstrap_ci(
    a_scores: List[int],
    b_scores: List[int],
    n_resamples: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    a = np.asarray(a_scores, dtype=np.float64)
    b = np.asarray(b_scores, dtype=np.float64)
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    boot_means = diff[idx].mean(axis=1)
    return {
        "mean_diff": round(float(diff.mean()), 6),
        "ci95_low": round(float(np.quantile(boot_means, 0.025)), 6),
        "ci95_high": round(float(np.quantile(boot_means, 0.975)), 6),
        "n": int(len(diff)),
        "n_resamples": n_resamples,
    }


def evaluate_loaded_model(
    model,
    tokenizer,
    examples: List[Dict[str, Any]],
    trained_labels: Set[str],
    max_new_tokens: int = 16,
) -> List[Dict[str, Any]]:
    """
    Прогоняет модель на всех примерах в детерминированном greedy-режиме (do_sample=False),
    сохраняя сырой ответ без редакции, факт точного совпадения и проверку утечки старых меток на Теме 3.
    """
    device = next(model.parameters()).device
    model.eval()
    preds: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for ex in examples:
            prompt_text = tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            prompt_len = inputs["input_ids"].shape[1]

            # Жёстко выключаем сэмплирование: одинаковый вход должен давать воспроизводимый выход
            gen_out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_ids = gen_out[0, prompt_len:].tolist()
            raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            first_line = raw.splitlines()[0].strip() if raw else ""
            cleaned = first_line.strip("`\"'.,:; ")
            allowed = set(ex["allowed_labels"])
            if cleaned not in allowed:
                for lbl in sorted(allowed, key=len, reverse=True):
                    if lbl in first_line:
                        cleaned = lbl
                        break

            valid = cleaned in allowed
            em = cleaned == ex["expected"]
            # Проверяем, не пытается ли дообученная модель на новой теме по привычке выдать старый класс из обучения
            leaked_old = (not ex["is_trained_topic"]) and (cleaned in trained_labels)
            preds.append(
                {
                    "raw": raw,
                    "pred": cleaned,
                    "valid": valid,
                    "em": em,
                    "leaked_old_label": leaked_old,
                }
            )
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="base_snapshot")
    parser.add_argument("--adapter-dir", default="artifacts/adapter")
    parser.add_argument("--merged-dir", default="artifacts/merged")
    parser.add_argument("--nf4-dir", default="artifacts/nf4")
    parser.add_argument("--ru-jsonl", default="data/raw/ru-RU.jsonl")
    parser.add_argument("--manifest", default="data/manifest.json")
    parser.add_argument("--train-jsonl", default="data/train.jsonl")
    parser.add_argument("--dev-jsonl", default="data/dev.jsonl")
    parser.add_argument("--dataset-out", default="data/blind-300.jsonl")
    parser.add_argument("--paired-jsonl", default="results/blind-300-paired.jsonl")
    parser.add_argument("--paired-csv", default="results/blind-300-paired.csv")
    parser.add_argument("--summary-out", default="results/blind-300-summary.json")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    trained_labels = set(manifest["selected_intents"])

    dataset = build_blind_dataset(
        ru_jsonl_path=Path(args.ru_jsonl),
        manifest_path=Path(args.manifest),
        train_path=Path(args.train_jsonl),
        dev_path=Path(args.dev_jsonl),
        out_path=Path(args.dataset_out),
        seed=42,
    )
    print(f"Prepared blind dataset: {len(dataset)} rows -> {args.dataset_out}", flush=True)
    print("Topic counts:", dict(Counter(d["topic"] for d in dataset)), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_preds: Dict[str, List[Dict[str, Any]]] = {}
    timings: Dict[str, float] = {}

    # 1. base
    t0 = time.perf_counter()
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_dir, dtype=torch.float16, device_map={"": 0}
    )
    all_preds["base"] = evaluate_loaded_model(base_model, tokenizer, dataset, trained_labels)
    timings["base_sec"] = round(time.perf_counter() - t0, 2)
    del base_model
    cleanup_cuda()
    print(f"Evaluated base in {timings['base_sec']}s", flush=True)

    # 2. base + adapter (on the fly)
    t0 = time.perf_counter()
    base_for_adapter = AutoModelForCausalLM.from_pretrained(
        args.base_dir, dtype=torch.float16, device_map={"": 0}
    )
    adapter_model = PeftModel.from_pretrained(base_for_adapter, args.adapter_dir)
    all_preds["adapter"] = evaluate_loaded_model(adapter_model, tokenizer, dataset, trained_labels)
    timings["adapter_sec"] = round(time.perf_counter() - t0, 2)
    del adapter_model, base_for_adapter
    cleanup_cuda()
    print(f"Evaluated base+adapter in {timings['adapter_sec']}s", flush=True)

    # 3. merged (FP16)
    t0 = time.perf_counter()
    merged_model = AutoModelForCausalLM.from_pretrained(
        args.merged_dir, dtype=torch.float16, device_map={"": 0}
    )
    all_preds["merged"] = evaluate_loaded_model(merged_model, tokenizer, dataset, trained_labels)
    timings["merged_sec"] = round(time.perf_counter() - t0, 2)
    del merged_model
    cleanup_cuda()
    print(f"Evaluated merged in {timings['merged_sec']}s", flush=True)

    # 4. nf4 (4-bit)
    t0 = time.perf_counter()
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    nf4_model = AutoModelForCausalLM.from_pretrained(
        args.nf4_dir, quantization_config=bnb_cfg, device_map={"": 0}
    )
    all_preds["nf4"] = evaluate_loaded_model(nf4_model, tokenizer, dataset, trained_labels)
    timings["nf4_sec"] = round(time.perf_counter() - t0, 2)
    del nf4_model
    cleanup_cuda()
    print(f"Evaluated nf4 in {timings['nf4_sec']}s", flush=True)

    # Write unedited paired JSONL & CSV
    paired_rows: List[Dict[str, Any]] = []
    Path(args.paired_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.paired_jsonl).open("w", encoding="utf-8") as fj, Path(args.paired_csv).open(
        "w", encoding="utf-8", newline=""
    ) as fc:
        fieldnames = [
            "idx",
            "topic",
            "id",
            "expected",
            "user_text",
            "base_pred",
            "base_ok",
            "adapter_pred",
            "adapter_ok",
            "merged_pred",
            "merged_ok",
            "nf4_pred",
            "nf4_ok",
        ]
        writer = csv.DictWriter(fc, fieldnames=fieldnames)
        writer.writeheader()
        for i, ex in enumerate(dataset):
            row = {
                "idx": i,
                "topic": ex["topic"],
                "is_trained_topic": ex["is_trained_topic"],
                "id": ex["id"],
                "expected": ex["expected"],
                "user_text": ex["user_text"],
                "models": {m: all_preds[m][i] for m in ("base", "adapter", "merged", "nf4")},
            }
            paired_rows.append(row)
            fj.write(json.dumps(row, ensure_ascii=False) + "\n")
            writer.writerow(
                {
                    "idx": i,
                    "topic": ex["topic"],
                    "id": ex["id"],
                    "expected": ex["expected"],
                    "user_text": ex["user_text"],
                    "base_pred": all_preds["base"][i]["pred"],
                    "base_ok": int(all_preds["base"][i]["em"]),
                    "adapter_pred": all_preds["adapter"][i]["pred"],
                    "adapter_ok": int(all_preds["adapter"][i]["em"]),
                    "merged_pred": all_preds["merged"][i]["pred"],
                    "merged_ok": int(all_preds["merged"][i]["em"]),
                    "nf4_pred": all_preds["nf4"][i]["pred"],
                    "nf4_ok": int(all_preds["nf4"][i]["em"]),
                }
            )

    # Aggregate summary per topic + trained (200) + all (300)
    groups: List[Tuple[str, List[int]]] = [
        ("topic1_work_comms", [i for i, d in enumerate(dataset) if d["topic"] == "topic1_work_comms"]),
        ("topic2_media_facts", [i for i, d in enumerate(dataset) if d["topic"] == "topic2_media_facts"]),
        ("topic3_unseen_daily", [i for i, d in enumerate(dataset) if d["topic"] == "topic3_unseen_daily"]),
        ("in_domain_total_200", [i for i, d in enumerate(dataset) if d["is_trained_topic"]]),
        ("all_300", list(range(len(dataset)))),
    ]

    summary: Dict[str, Any] = {"total_examples": len(dataset), "timings_sec": timings, "slices": {}}
    for gname, idxs in groups:
        y_true = [dataset[i]["expected"] for i in idxs]
        labels = sorted(set(y_true))
        slice_res: Dict[str, Any] = {"n": len(idxs), "models": {}, "comparisons": {}}

        for m in ("base", "adapter", "merged", "nf4"):
            y_pred = [all_preds[m][i]["pred"] for i in idxs]
            ems = [all_preds[m][i]["em"] for i in idxs]
            valids = [all_preds[m][i]["valid"] for i in idxs]
            leaks = [all_preds[m][i]["leaked_old_label"] for i in idxs]
            slice_res["models"][m] = {
                "exact_match": round(sum(ems) / len(idxs), 6),
                "valid_format_rate": round(sum(valids) / len(idxs), 6),
                "macro_f1": round(compute_macro_f1(y_true, y_pred, labels), 6),
                "leaked_old_label_rate": round(sum(leaks) / len(idxs), 6),
            }

        # Exact prediction-string agreement rates + paired bootstrap CIs
        def agree(m1: str, m2: str) -> float:
            return round(
                sum(1 for i in idxs if all_preds[m1][i]["pred"] == all_preds[m2][i]["pred"]) / len(idxs),
                6,
            )

        slice_res["comparisons"] = {
            "agreement_adapter_vs_merged": agree("adapter", "merged"),
            "agreement_base_vs_merged": agree("base", "merged"),
            "agreement_merged_vs_nf4": agree("merged", "nf4"),
            "merged_minus_base_em": paired_bootstrap_ci(
                [int(all_preds["merged"][i]["em"]) for i in idxs],
                [int(all_preds["base"][i]["em"]) for i in idxs],
            ),
            "nf4_minus_merged_em": paired_bootstrap_ci(
                [int(all_preds["nf4"][i]["em"]) for i in idxs],
                [int(all_preds["merged"][i]["em"]) for i in idxs],
            ),
            "adapter_minus_merged_em": paired_bootstrap_ci(
                [int(all_preds["adapter"][i]["em"]) for i in idxs],
                [int(all_preds["merged"][i]["em"]) for i in idxs],
            ),
        }
        summary["slices"][gname] = slice_res

    Path(args.summary_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
