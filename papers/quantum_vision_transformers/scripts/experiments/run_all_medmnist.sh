#!/usr/bin/env bash
# run_all_medmnist.sh — Full MedMNIST benchmark: all 4 models × 12 datasets × 3 seeds.
#
# This is the complete reproduction of Table 3 / Table 6 from the paper.
# WARNING: Model D is slow (~minutes/epoch for n=16,d=16).  Start with A and B.
#
# Usage:
#   bash scripts/experiments/run_all_medmnist.sh
#   bash scripts/experiments/run_all_medmnist.sh --device cuda:0
#   MODELS="A B" bash scripts/experiments/run_all_medmnist.sh   # subset of models

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

EXTRA="${*}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MODELS="${MODELS:-A B D D_full}"
SEEDS=(42 123 7)

DATASETS=(
    pathmnist chestmnist dermamnist octmnist pneumoniamnist retinamnist
    breastmnist bloodmnist tissuemnist organamnist organcmnist organsmnist
)

for ds in "${DATASETS[@]}"; do
    for m in $MODELS; do
        # use model_b_retina.json as template, override dataset
        if [ "$m" = "D_full" ]; then
            cfg="configs/model_d_full_retina.json"
        else
            cfg="configs/model_$(echo $m | tr A-Z a-z)_retina.json"
        fi
        [ -f "$cfg" ] || { echo "SKIP $cfg"; continue; }

        for s in "${SEEDS[@]}"; do
            tag="${m}_${ds}_generic_s${s}"
            out="outdir/${tag}"
            [ "$FORCE_RERUN" != "1" ] && [ -d "$out" ] && { echo "SKIP ${tag} (folder exists; set FORCE_RERUN=1 to rerun)"; continue; }

            echo ""
            echo "════ Model ${m}  ${ds}  generic  seed ${s} ════"

            # Create a temporary config with the right dataset name
            tmp_cfg=$(mktemp)
            python -c "
import json, sys
with open('${cfg}') as f: c = json.load(f)
c['dataset'] = '${ds}'
c['model_type'] = 'D' if '${m}' == 'D_full' else '${m}'
c['circuit_family'] = 'generic'
if '${m}' == 'D_full':
    c['compound_readout'] = 'full_sector'
json.dump(c, sys.stdout, indent=2)
" > "$tmp_cfg"

            python implementation.py \
                --config "$tmp_cfg" --seed "$s" --outdir "$out" \
                $EXTRA 2>&1 | tee "${out}.log"

            rm -f "$tmp_cfg"
        done
    done
done

echo ""
echo "Done.  Now run:  python scripts/analysis/generate_figures.py outdir/"
