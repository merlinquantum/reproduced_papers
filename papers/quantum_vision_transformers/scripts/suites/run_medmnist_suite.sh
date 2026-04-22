#!/usr/bin/env bash
# run_medmnist_suite.sh — Run the full MedMNIST benchmark suite.
#
# Includes:
#   - paper MedMNIST benchmark
#   - generic full variants
#   - butterfly full variants
#   - butterfly lite variants
#
# Usage:
#   bash scripts/suites/run_medmnist_suite.sh
#   bash scripts/suites/run_medmnist_suite.sh --device cuda:0
#   FORCE_RERUN=1 bash scripts/suites/run_medmnist_suite.sh
#
# Skip behavior:
#   Each underlying runner skips any run whose output folder already exists.
#   Re-running this suite is therefore safe after interruptions.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

echo "== MedMNIST paper benchmark =="
bash scripts/reproduction/run_paper_medmnist_benchmark.sh "$@"

echo ""
echo "== MedMNIST generic full =="
bash scripts/experiments/run_all_medmnist.sh "$@"

echo ""
echo "== MedMNIST butterfly full =="
bash scripts/experiments/run_all_medmnist_butterfly.sh "$@"

echo ""
echo "== MedMNIST butterfly lite =="
bash scripts/experiments/run_all_medmnist_butterfly_lite.sh "$@"

echo ""
echo "MedMNIST suite complete."
echo "Generate figures with: python scripts/analysis/generate_figures.py outdir/"
