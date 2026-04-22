#!/usr/bin/env bash
# run_retina_suite.sh — Run the full RetinaMNIST benchmark suite.
#
# Includes:
#   - paper Retina benchmark
#   - generic full variants
#   - butterfly full variants
#   - generic lite variants
#   - butterfly lite variants
#
# Usage:
#   bash scripts/suites/run_retina_suite.sh
#   bash scripts/suites/run_retina_suite.sh --device cuda:0
#   FORCE_RERUN=1 bash scripts/suites/run_retina_suite.sh
#
# Skip behavior:
#   Each underlying runner skips any run whose output folder already exists.
#   Re-running this suite is therefore safe after interruptions.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

echo "== Retina paper benchmark =="
bash scripts/reproduction/run_paper_retina_benchmark.sh "$@"

echo ""
echo "== Retina generic full =="
bash scripts/experiments/run_all_retina.sh "$@"

echo ""
echo "== Retina butterfly full =="
bash scripts/experiments/run_all_retina_butterfly.sh "$@"

echo ""
echo "== Retina generic lite =="
bash scripts/experiments/run_all_retina_lite.sh "$@"

echo ""
echo "== Retina butterfly lite =="
bash scripts/experiments/run_all_retina_butterfly_lite.sh "$@"

echo ""
echo "Retina suite complete."
echo "Generate figures with: python scripts/analysis/generate_figures.py outdir/"
