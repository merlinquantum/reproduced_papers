#!/usr/bin/env bash
# run_all_retina_butterfly.sh — A, B, D QVT models on RetinaMNIST with butterfly layout.
#
# Usage:
#   bash scripts/experiments/run_all_retina_butterfly.sh
#   bash scripts/experiments/run_all_retina_butterfly.sh --device cuda:0
#
# Results: outdir/{model}_retina_butterfly_s{seed}/results.json
# Figures: python scripts/analysis/generate_figures.py outdir/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

EXTRA="${*}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MODELS="${MODELS:-A B D D_full E F}"
SEEDS=(42)

for m in $MODELS; do
    if [ "$m" = "D_full" ]; then
        cfg="configs/model_d_full_retina.json"
    else
        cfg="configs/model_$(echo $m | tr A-Z a-z)_retina.json"
    fi
    [ -f "$cfg" ] || { echo "SKIP $cfg (not found)"; continue; }

    for s in "${SEEDS[@]}"; do
        tag="${m}_retina_butterfly_s${s}"
        out="outdir/${tag}"
        [ "$FORCE_RERUN" != "1" ] && [ -d "$out" ] && { echo "SKIP ${tag} (folder exists; set FORCE_RERUN=1 to rerun)"; continue; }

        echo ""
        echo "════════════════════════════════════════"
        echo "  Model ${m}  seed ${s}  (butterfly)"
        echo "════════════════════════════════════════"

        # Create a temporary config with the butterfly circuit family
        tmp_cfg=$(mktemp)
        python -c "
import json, sys
with open('${cfg}') as f: c = json.load(f)
c['circuit_family'] = 'butterfly'
# Model D and D_full require use_cls_token=False for butterfly if n=16, d=16
if '${m}' in ('D', 'D_full'):
    c['use_cls_token'] = False
json.dump(c, sys.stdout, indent=2)
" > "$tmp_cfg"

        python ../../implementation.py --paper quantum_vision_transformers \
            --config "$tmp_cfg" --seed "$s" --outdir "$out" \
            $EXTRA 2>&1 | tee "${out}.log"

        rm -f "$tmp_cfg"
    done
done

echo ""
echo "Done.  Now run:  python scripts/analysis/generate_figures.py outdir/"
