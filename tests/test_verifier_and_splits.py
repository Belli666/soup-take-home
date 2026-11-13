"""Локальные регрессионные тесты для проверки отрицательных контролей verify --mode full и фильтрации утечек."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

# Заглушки тяжёлых GPU-зависимостей (torch, peft, transformers, safetensors, numpy),
# чтобы тест логики CLI и фильтрации работал в лёгком CPU-окружении без Colab.
def install_ml_stubs() -> None:
    if "numpy" not in sys.modules:
        np_mod = types.ModuleType("numpy")
        np_mod.float64 = float
        np_mod.asarray = lambda x, dtype=None: list(x)
        np_mod.array = lambda x, dtype=None: list(x)
        np_mod.random = types.SimpleNamespace(default_rng=lambda seed=None: None)
        sys.modules["numpy"] = np_mod
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        torch_mod.nn = types.SimpleNamespace(functional=types.ModuleType("torch.nn.functional"))
        torch_mod.Tensor = object
        torch_mod.float16 = "float16"
        sys.modules["torch"] = torch_mod
        sys.modules["torch.nn"] = torch_mod.nn
        sys.modules["torch.nn.functional"] = torch_mod.nn.functional
    if "peft" not in sys.modules:
        peft_mod = types.ModuleType("peft")
        peft_mod.LoraConfig = object
        peft_mod.PeftModel = object
        peft_mod.get_peft_model = lambda *a, **kw: None
        sys.modules["peft"] = peft_mod
    if "safetensors.torch" not in sys.modules:
        st_pkg = types.ModuleType("safetensors")
        st_torch = types.ModuleType("safetensors.torch")
        st_torch.load_file = lambda *a, **kw: {}
        sys.modules["safetensors"] = st_pkg
        sys.modules["safetensors.torch"] = st_torch
    if "transformers" not in sys.modules:
        tr_mod = types.ModuleType("transformers")
        tr_mod.AutoModelForCausalLM = object
        class DummyTok:
            @staticmethod
            def from_pretrained(*a, **kw):
                return ["tok"] * 100
        tr_mod.AutoTokenizer = DummyTok
        tr_mod.BitsAndBytesConfig = object
        sys.modules["transformers"] = tr_mod


install_ml_stubs()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import verify  # noqa: E402
import eval_3topics_4models as ev  # noqa: E402


class TestVerifyModeFullExitCodes(unittest.TestCase):
    def _run_full_with_statuses(self, act_status: str, mp_status: str) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev_p = root / "dev.jsonl"
            test_p = root / "test.jsonl"
            manifest_p = root / "manifest.json"
            out_dir = root / "out"
            dev_p.write_text('{"id":"1","messages":[{"role":"user","content":"hi"}],"expected":"news_query"}\n', encoding="utf-8")
            test_p.write_text('{"id":"2","messages":[{"role":"user","content":"bye"}],"expected":"news_query"}\n', encoding="utf-8")
            manifest_p.write_text(json.dumps({"selected_intents": ["news_query"]}), encoding="utf-8")

            fake_sum = {"artifact": "x", "model_dir": "y", "n": 1, "exact_match": 1.0, "valid_format_rate": 1.0, "macro_f1": 1.0, "errors_by_expected_class": {}}
            fake_rec = [{"artifact": "x", "id": "2", "expected": "news_query", "raw_output": "news_query", "parsed_pred": "news_query", "valid_format": True, "exact_match": True, "stop_reason": "eos", "new_token_ids": [1]}]

            argv = [
                "verify.py", "--mode", "full",
                "--base-dir", str(root), "--adapter-dir", str(root),
                "--merged-dir", str(root), "--nf4-dir", str(root),
                "--manifest", str(manifest_p), "--dev", str(dev_p), "--test", str(test_p),
                "--out-dir", str(out_dir),
            ]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(verify, "check_adapter_activity", return_value={"status": act_status}), \
                 mock.patch.object(verify, "check_merge_parity", return_value={"status": mp_status}), \
                 mock.patch.object(verify, "evaluate_artifact", return_value=(fake_sum, fake_rec)), \
                 mock.patch.object(verify, "paired_bootstrap_ci", return_value={"mean_diff": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n": 1, "n_resamples": 10}):
                try:
                    verify.main()
                    return 0
                except SystemExit as exc:
                    return int(exc.code)

    def test_failed_adapter_activity_returns_nonzero(self):
        self.assertEqual(self._run_full_with_statuses("FAILED", "PASSED"), 2)

    def test_failed_merge_parity_returns_nonzero(self):
        self.assertEqual(self._run_full_with_statuses("PASSED", "FAILED"), 3)

    def test_all_passed_returns_zero(self):
        self.assertEqual(self._run_full_with_statuses("PASSED", "PASSED"), 0)


class TestBlindSplitZeroLeakage(unittest.TestCase):
    def test_excludes_train_dev_and_test_user_utterances(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_p, dev_p, test_p = root / "train.jsonl", root / "dev.jsonl", root / "test.jsonl"
            ru_p, man_p, out_p = root / "ru.jsonl", root / "manifest.json", root / "blind.jsonl"

            man_p.write_text(json.dumps({
                "system_prompt": "sys",
                "selected_intents": ev.TOPIC1_INTENTS + ev.TOPIC2_INTENTS,
            }), encoding="utf-8")

            # В train последним сообщением идёт метка assistant — фильтр обязан проверять role == "user"
            train_p.write_text(json.dumps({
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "уникальная фраза из train"},
                    {"role": "assistant", "content": "calendar_query"},
                ]
            }) + "\n", encoding="utf-8")
            dev_p.write_text(json.dumps({
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "уникальная фраза из dev"},
                ]
            }) + "\n", encoding="utf-8")
            test_p.write_text(json.dumps({
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "уникальная фраза из test"},
                ]
            }) + "\n", encoding="utf-8")

            raw_rows = [
                {"id": "leak_tr", "partition": "test", "intent": "calendar_query", "utt": "Уникальная фраза из train"},
                {"id": "leak_dv", "partition": "test", "intent": "calendar_query", "utt": "уникальная фраза из dev"},
                {"id": "leak_ts", "partition": "test", "intent": "calendar_query", "utt": "уникальная фраза из test"},
                {"id": "ok_1", "partition": "test", "intent": "calendar_query", "utt": "чистая фраза т1"},
                {"id": "ok_2", "partition": "test", "intent": "play_music", "utt": "чистая фраза т2"},
                {"id": "ok_3", "partition": "test", "intent": "alarm_set", "utt": "чистая фраза т3"},
            ]
            ru_p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in raw_rows) + "\n", encoding="utf-8")

            ds = ev.build_blind_dataset(ru_p, man_p, train_p, dev_p, out_p, seed=42, test_path=test_p)
            ids = {r["id"] for r in ds}
            self.assertEqual(ids, {"ok_1", "ok_2", "ok_3"})


if __name__ == "__main__":
    unittest.main()
