#!/usr/bin/env bash
# run_all_benchmarks.sh — Run the full RetinaMNIST and MedMNIST suites.
#
# Usage:
#   bash scripts/suites/run_all_benchmarks.sh
#   bash scripts/suites/run_all_benchmarks.sh --device cuda:0
#   FORCE_RERUN=1 bash scripts/suites/run_all_benchmarks.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

echo "== Retina suite =="
bash scripts/suites/run_retina_suite.sh "$@"

echo ""
echo "== MedMNIST suite =="
bash scripts/suites/run_medmnist_suite.sh "$@"

echo ""
echo "All benchmarks complete."
echo "Generate figures with: python scripts/analysis/generate_figures.py outdir/"
