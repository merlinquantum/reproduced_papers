#!/usr/bin/env bash
# validate.sh — Full validation before committing to a benchmark run.
#
# 1. verify.py  — unit tests all models (shapes, gradients, sectors, param tying)
# 2. smoke_test — 3 epochs × 1 layer for every model type (A–F)
#
# If either step fails, the script exits with an error.
#
# Usage:  bash scripts/validation/validate.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║           QVT Validation Suite                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Unit-level checks ──
echo "━━━ Step 1/2: verify.py (unit checks) ━━━"
echo ""
python verify.py
VERIFY_EXIT=$?
if [ $VERIFY_EXIT -ne 0 ]; then
    echo ""
    echo "✗ verify.py failed.  Fix errors above before running benchmarks."
    exit 1
fi

echo ""
echo "━━━ Step 2/2: smoke test (3 epochs per model) ━━━"
echo ""
bash scripts/validation/smoke_test.sh
SMOKE_EXIT=$?
if [ $SMOKE_EXIT -ne 0 ]; then
    echo ""
    echo "✗ Smoke test failed.  Check logs in outdir/smoke_*/"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✓ All validation passed.  Safe to run benchmarks.      ║"
echo "║                                                          ║"
echo "║  Next:  bash scripts/suites/run_retina_suite.sh          ║"
echo "╚══════════════════════════════════════════════════════════╝"
