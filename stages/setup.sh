#!/usr/bin/env bash
set -euo pipefail

python --version
uname -a
free -h
df -h /content || df -h .

python -m pip install 'soup-cli[train]==0.77.2' bitsandbytes
python -m pip uninstall -y torchao || true
python -m pip freeze > "$PHASE_LOG_DIR/pip-freeze.txt"

pip_check_rc=0
python -m pip check || pip_check_rc=$?
printf 'pip_check_exit=%s\n' "$pip_check_rc"

python -c 'from importlib.metadata import version; print("Soup", version("soup-cli"))'
python -c 'import torch, transformers, peft, trl, accelerate, bitsandbytes'
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
