#!/usr/bin/env bash
# run_retina_suite.sh — Full RetinaMNIST benchmark campaign:
# paper baselines + the generic/butterfly × full/lite grid.
#
# CPU_FRIENDLY=1 runs the reduced CPU workflow (fewer models, no generic
# passes) — this replaces the former run_retina_cpu_suite.sh.
#
# Skip behavior:
#   Each underlying runner skips any run whose output folder already exists.
#   Re-running this suite is therefore safe after interruptions.
#
# Usage:
#   bash scripts/suites/run_retina_suite.sh
#   bash scripts/suites/run_retina_suite.sh --device cuda:0
#   CPU_FRIENDLY=1 bash scripts/suites/run_retina_suite.sh --device cpu
#   FORCE_RERUN=1 bash scripts/suites/run_retina_suite.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CPU_FRIENDLY="${CPU_FRIENDLY:-0}"

if [ "$CPU_FRIENDLY" = "1" ]; then
    echo "== Retina paper benchmark (CPU set) =="
    MODELS="vision_transformer orthofnn" bash scripts/reproduction/run_paper_retina_benchmark.sh "$@"
    echo ""
    echo "== Retina butterfly full (CPU set) =="
    MODELS="A B D D_full" CIRCUIT_FAMILY=butterfly bash scripts/experiments/run_all_retina.sh "$@"
    echo ""
    echo "== Retina butterfly lite (CPU set) =="
    MODELS="A B D D_full" CIRCUIT_FAMILY=butterfly PROFILE=lite bash scripts/experiments/run_all_retina.sh "$@"
else
    echo "== Retina paper benchmark =="
    bash scripts/reproduction/run_paper_retina_benchmark.sh "$@"
    echo ""
    echo "== Retina generic full =="
    bash scripts/experiments/run_all_retina.sh "$@"
    echo ""
    echo "== Retina butterfly full =="
    CIRCUIT_FAMILY=butterfly bash scripts/experiments/run_all_retina.sh "$@"
    echo ""
    echo "== Retina generic lite =="
    PROFILE=lite bash scripts/experiments/run_all_retina.sh "$@"
    echo ""
    echo "== Retina butterfly lite =="
    CIRCUIT_FAMILY=butterfly PROFILE=lite bash scripts/experiments/run_all_retina.sh "$@"
fi

echo ""
echo "Retina suite complete."
echo "Generate figures with: python scripts/analysis/generate_figures.py outdir/"
