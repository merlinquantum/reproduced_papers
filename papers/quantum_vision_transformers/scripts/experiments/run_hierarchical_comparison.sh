#!/usr/bin/env bash
# run_hierarchical_comparison.sh — Compare 3-photon hierarchical model
# against 2-photon compound and explicit-attention baseline.
#
# Models:
#   B       — Quantum Orthogonal Transformer (V + W, 1-photon, paper baseline)
#   D       — Compound Transformer (2-photon, paper)
#   D_full  — Compound with full-sector readout (2-photon, extension)
#   E       — Multi-sector attention (shared circuit, 1+2 photon, extension)
#   F       — Hierarchical Compound (3-photon, region+patch+feature, extension)
#
# NOTE: Model F with 24 modes × 3 photons has 2600 Fock states.
#       Expect ~5-10× slower per step than Model D.
#
# Usage:
#   bash scripts/experiments/run_hierarchical_comparison.sh
#   bash scripts/experiments/run_hierarchical_comparison.sh --device cuda:0

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
CONFIGS["F"]="configs/model_f_retina.json"

for tag_prefix in "B" "D" "D_full" "E" "F"; do
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
echo "Done.  Generate figures with:"
echo "  python scripts/analysis/generate_figures.py outdir/ --dataset retinamnist"
