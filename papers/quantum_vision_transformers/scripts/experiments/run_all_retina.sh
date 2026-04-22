#!/usr/bin/env bash
# run_all_retina.sh — All 4 QVT models on RetinaMNIST, 3 seeds (paper protocol).
#
# Usage:
#   bash scripts/experiments/run_all_retina.sh
#   bash scripts/experiments/run_all_retina.sh --device cuda:0
#
# Results: outdir/{model}_retina_s{seed}/results.json
# Figures: python scripts/analysis/generate_figures.py outdir/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

EXTRA="${*}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MODELS="${MODELS:-A B C D D_full E F}"
SEEDS=(42 123 7)

for m in $MODELS; do
    if [ "$m" = "D_full" ]; then
        cfg="configs/model_d_full_retina.json"
    else
        cfg="configs/model_$(echo $m | tr A-Z a-z)_retina.json"
    fi
    [ -f "$cfg" ] || { echo "SKIP $cfg (not found)"; continue; }

    for s in "${SEEDS[@]}"; do
        tag="${m}_retina_generic_s${s}"
        out="outdir/${tag}"
        [ "$FORCE_RERUN" != "1" ] && [ -d "$out" ] && { echo "SKIP ${tag} (folder exists; set FORCE_RERUN=1 to rerun)"; continue; }

        echo ""
        echo "════════════════════════════════════════"
        echo "  Model ${m}  seed ${s}  (generic)"
        echo "════════════════════════════════════════"
        python implementation.py \
            --config "$cfg" --seed "$s" --outdir "$out" \
            $EXTRA 2>&1 | tee "${out}.log"
    done
done

echo ""
echo "Done.  Now run:  python scripts/analysis/generate_figures.py outdir/"
