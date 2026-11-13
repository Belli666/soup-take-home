#!/usr/bin/env python3
"""
Wrapper to run `soup` CLI in the same Python process while recording PyTorch CUDA peak memory.
Follows Section 4 of soup-take-home-colab-cli-plan.md.
"""

import importlib.metadata
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch


def find_soup_entry_point():
    """Находит точку входа консольной команды `soup` внутри установленного пакета soup-cli."""
    dist = importlib.metadata.distribution("soup-cli")
    for ep in dist.entry_points:
        if ep.group == "console_scripts" and ep.name == "soup":
            return ep
    raise RuntimeError("Could not find console_scripts entry point 'soup' in soup-cli")


def main() -> None:
    # Сбрасываем счётчики пиковой памяти PyTorch до запуска команды, чтобы замер был чистым
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    exit_code = 0
    exc_info_str = None

    # Подменяем sys.argv и вызываем `soup` напрямую в ЭТОМ ЖЕ процессе Python:
    # это важно, потому что CUDA max_memory_allocated() живёт только внутри своего процесса.
    old_argv = sys.argv[:]
    sys.argv = ["soup"] + sys.argv[1:]

    try:
        ep = find_soup_entry_point()
        fn = ep.load()
        ret = fn()
        if isinstance(ret, int):
            exit_code = ret
    except SystemExit as e:
        # Click/Typer завершают CLI через SystemExit, сохраняем реальный код возврата
        if isinstance(e.code, int):
            exit_code = e.code
        elif e.code is None:
            exit_code = 0
        else:
            print(e.code, file=sys.stderr)
            exit_code = 1
    except Exception:
        exc_info_str = traceback.format_exc()
        print(exc_info_str, file=sys.stderr)
        exit_code = 1
    finally:
        sys.argv = old_argv
        elapsed_sec = time.perf_counter() - t0
        cuda_stats = {
            "argv": ["soup"] + old_argv[1:],
            "elapsed_sec": round(elapsed_sec, 3),
            "exit_code": exit_code,
            "exception": exc_info_str,
            "cuda_available": torch.cuda.is_available(),
        }
        # Фиксируем раздельно:
        # - allocated: реальная память тензоров (веса + градиенты + активации)
        # - reserved: память, зарезервированная пулом кэширующего аллокатора PyTorch
        if torch.cuda.is_available():
            alloc_bytes = torch.cuda.max_memory_allocated()
            res_bytes = torch.cuda.max_memory_reserved()
            cuda_stats.update(
                {
                    "device_name": torch.cuda.get_device_name(0),
                    "max_memory_allocated_bytes": alloc_bytes,
                    "max_memory_allocated_mib": round(alloc_bytes / (1024**2), 2),
                    "max_memory_allocated_gib": round(alloc_bytes / (1024**3), 4),
                    "max_memory_reserved_bytes": res_bytes,
                    "max_memory_reserved_mib": round(res_bytes / (1024**2), 2),
                    "max_memory_reserved_gib": round(res_bytes / (1024**3), 4),
                }
            )

        out_dir = Path(os.environ.get("PHASE_LOG_DIR", "results"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "cuda-memory.json"
        out_path.write_text(json.dumps(cuda_stats, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[measure_soup] Saved CUDA stats to {out_path}: {json.dumps(cuda_stats)}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
