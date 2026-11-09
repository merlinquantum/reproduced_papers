#!/usr/bin/env bash
# Run the sweep coordinator in the foreground of this shell.
#
# Launched from Windows through a hidden, long-lived wsl.exe so WSL cannot
# idle-stop the distribution while the sweep runs:
#   Start-Process -WindowStyle Hidden -FilePath wsl.exe -ArgumentList `
#     '-e env PQKD_PYTHON=<venv>/bin/python bash <repo>/papers/photonic_QKD/utils/launch_sweep.sh <spec.json | --resume DIR>'
# Arguments are passed to utils/run_sweep.py unchanged. PQKD_PYTHON selects the
# interpreter of the reproduction environment (default: python3 on PATH).
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
python_bin="${PQKD_PYTHON:-python3}"
console="$HOME/pqkd_sweep_$(date +%Y%m%d-%H%M%S).log"
exec "$python_bin" papers/photonic_QKD/utils/run_sweep.py "$@" > "$console" 2>&1
