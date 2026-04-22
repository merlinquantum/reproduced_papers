#!/usr/bin/env bash
# run_cpu_friendly_benchmarks.sh — Run the CPU-friendly RetinaMNIST and MedMNIST suites.
#
# This excludes the heaviest extension models E and F from the Retina aggregate runs.
#
# Usage:
#   bash scripts/suites/run_cpu_friendly_benchmarks.sh
#   bash scripts/suites/run_cpu_friendly_benchmarks.sh --device cpu
#   FORCE_RERUN=1 bash scripts/suites/run_cpu_friendly_benchmarks.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

echo "== Retina CPU-friendly suite =="
bash scripts/suites/run_retina_cpu_suite.sh "$@"

echo ""
echo "== MedMNIST CPU-friendly suite =="
bash scripts/suites/run_medmnist_cpu_suite.sh "$@"

echo ""
echo "CPU-friendly benchmark suites complete."
echo "Generate figures with: python scripts/analysis/generate_figures.py outdir/"
