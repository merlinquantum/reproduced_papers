#!/usr/bin/env bash
# run_medmnist_cpu_suite.sh — CPU-friendly MedMNIST suite.
#
# The MedMNIST aggregate runners already exclude the heavier extension models E and F,
# so this wrapper is mainly a clearer CPU-oriented entry point.
#
# Includes:
#   - paper MedMNIST benchmark
#   - generic full variants: A B D D_full
#   - butterfly full variants: A B D D_full
#   - butterfly lite variants: A B D D_full
#
# Usage:
#   bash scripts/suites/run_medmnist_cpu_suite.sh
#   bash scripts/suites/run_medmnist_cpu_suite.sh --device cpu
#   FORCE_RERUN=1 bash scripts/suites/run_medmnist_cpu_suite.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

echo "== MedMNIST paper benchmark =="
MODELS="A B D" bash scripts/reproduction/run_paper_medmnist_benchmark.sh "$@"

echo ""
echo "== MedMNIST generic full (CPU-friendly set) =="
bash scripts/experiments/run_all_medmnist.sh "$@"

echo ""
echo "== MedMNIST butterfly full (CPU-friendly set) =="
bash scripts/experiments/run_all_medmnist_butterfly.sh "$@"

echo ""
echo "== MedMNIST butterfly lite (CPU-friendly set) =="
bash scripts/experiments/run_all_medmnist_butterfly_lite.sh "$@"

echo ""
echo "MedMNIST CPU-friendly suite complete."
echo "Generate figures with: python scripts/analysis/generate_figures.py outdir/"
