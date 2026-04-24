#!/usr/bin/env bash
# run_extensions_comparison.sh — Compare our extensions against paper baselines.
#
# Paper models:
#   B       — Quantum Orthogonal Transformer (V + W, two interferometers)
#   D       — Compound Transformer (cross-partition only)
#
# Our extensions:
#   D_full  — Compound with full-sector readout (emergent attention from pp sector)
#   E       — Multi-sector attention (shared circuit, 1ph features + 2ph attention)
#
# Usage:
#   bash scripts/experiments/run_extensions_comparison.sh
#   bash scripts/experiments/run_extensions_comparison.sh --device cuda:0

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

EXTRA="${*}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SEEDS=(42 123 7)

declare -A CONFIGS
CONFIGS["B"]="configs/model_b_retina.json"
CONFIGS["D"]="configs/model_d_retina.json"
CONFIGS["D_full"]="configs/model_d_full_retina.json"
CONFIGS["E"]="configs/model_e_retina.json"

for tag_prefix in "B" "D" "D_full" "E"; do
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
        python ../../implementation.py --paper quantum_vision_transformers \
            --config "$cfg" --seed "$s" --outdir "$out" \
            $EXTRA 2>&1 | tee "${out}.log"
    done
done

echo ""
echo "Done.  Generate comparison figures with:"
echo "  python scripts/analysis/generate_figures.py outdir/ --dataset retinamnist"
