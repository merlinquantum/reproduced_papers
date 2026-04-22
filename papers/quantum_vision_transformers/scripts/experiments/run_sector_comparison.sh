#!/usr/bin/env bash
# run_sector_comparison.sh — Compare Model D variants on RetinaMNIST.
#
# Runs:
#   D cross_only  (paper default — discard same-register sectors)
#   D full_sector (extension — use all three sectors)
#   B             (baseline with explicit V + W interferometers)
#
# Usage:
#   bash scripts/experiments/run_sector_comparison.sh
#   bash scripts/experiments/run_sector_comparison.sh --device cuda:0

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

EXTRA="${*}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SEEDS=(42 123 7)

declare -A CONFIGS
CONFIGS["D_cross"]="configs/model_d_retina.json"
CONFIGS["D_full"]="configs/model_d_full_retina.json"
CONFIGS["B"]="configs/model_b_retina.json"

for tag_prefix in "D_cross" "D_full" "B"; do
    cfg="${CONFIGS[$tag_prefix]}"
    [ -f "$cfg" ] || { echo "SKIP $cfg"; continue; }

    for s in "${SEEDS[@]}"; do
        tag="${tag_prefix}_retina_s${s}"
        out="outdir/${tag}"
        [ "$FORCE_RERUN" != "1" ] && [ -d "$out" ] && { echo "SKIP ${tag} (folder exists; set FORCE_RERUN=1 to rerun)"; continue; }

        echo ""
        echo "════════════════════════════════════════"
        echo "  ${tag_prefix}  seed ${s}"
        echo "════════════════════════════════════════"
        python implementation.py \
            --config "$cfg" --seed "$s" --outdir "$out" \
            $EXTRA 2>&1 | tee "${out}.log"
    done
done

echo ""
echo "Done.  Generate comparison figures with:"
echo "  python scripts/analysis/generate_figures.py outdir/ --dataset retinamnist"
