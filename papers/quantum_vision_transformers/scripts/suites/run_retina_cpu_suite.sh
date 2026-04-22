#!/usr/bin/env bash
# run_retina_cpu_suite.sh — CPU-friendly RetinaMNIST reproduction suite.
#
# Includes:
#   - paper Retina benchmark
#   - butterfly full variants: A B D D_full
#   - butterfly lite variants: A B D D_full
#
# Usage:
#   bash scripts/suites/run_retina_cpu_suite.sh
#   bash scripts/suites/run_retina_cpu_suite.sh --device cpu
#   FORCE_RERUN=1 bash scripts/suites/run_retina_cpu_suite.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

echo "== Retina paper benchmark =="
MODELS="vision_transformer orthofnn" bash scripts/reproduction/run_paper_retina_benchmark.sh "$@"

echo ""
echo "== Retina butterfly full (CPU-friendly set) =="
MODELS="A B D D_full" bash scripts/experiments/run_all_retina_butterfly.sh "$@"

echo ""
echo "== Retina butterfly lite (CPU-friendly set) =="
MODELS="A B D D_full" bash scripts/experiments/run_all_retina_butterfly_lite.sh "$@"

echo ""
echo "Retina CPU-friendly suite complete."
echo "Generate figures with: python scripts/analysis/generate_figures.py outdir/"
