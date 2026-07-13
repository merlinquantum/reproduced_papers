#!/usr/bin/env bash
# run_medmnist_suite.sh — Full MedMNIST benchmark campaign:
# paper baselines + the generic/butterfly × full/lite grid over 12 datasets.
#
# CPU_FRIENDLY=1 restricts the paper benchmark to A/B/D — this replaces the
# former run_medmnist_cpu_suite.sh.
#
# Skip behavior:
#   Each underlying runner skips any run whose output folder already exists.
#   Re-running this suite is therefore safe after interruptions.
#
# Usage:
#   bash scripts/suites/run_medmnist_suite.sh
#   bash scripts/suites/run_medmnist_suite.sh --device cuda:0
#   CPU_FRIENDLY=1 bash scripts/suites/run_medmnist_suite.sh --device cpu
#   FORCE_RERUN=1 bash scripts/suites/run_medmnist_suite.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CPU_FRIENDLY="${CPU_FRIENDLY:-0}"

echo "== MedMNIST paper benchmark =="
if [ "$CPU_FRIENDLY" = "1" ]; then
    MODELS="A B D" bash scripts/reproduction/run_paper_medmnist_benchmark.sh "$@"
else
    bash scripts/reproduction/run_paper_medmnist_benchmark.sh "$@"
fi
echo ""
echo "== MedMNIST generic full =="
bash scripts/experiments/run_all_medmnist.sh "$@"
echo ""
echo "== MedMNIST butterfly full =="
CIRCUIT_FAMILY=butterfly bash scripts/experiments/run_all_medmnist.sh "$@"
echo ""
echo "== MedMNIST butterfly lite =="
CIRCUIT_FAMILY=butterfly PROFILE=lite bash scripts/experiments/run_all_medmnist.sh "$@"

echo ""
echo "MedMNIST suite complete."
echo "Generate figures with: python scripts/analysis/generate_figures.py outdir/"
