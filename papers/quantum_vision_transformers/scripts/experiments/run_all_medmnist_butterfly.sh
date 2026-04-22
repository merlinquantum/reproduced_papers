#!/usr/bin/env bash
# run_all_medmnist_butterfly.sh — A, B, D models on 12 MedMNIST datasets with butterfly layout.
#
# Usage:
#   bash scripts/experiments/run_all_medmnist_butterfly.sh
#   bash scripts/experiments/run_all_medmnist_butterfly.sh --device cuda:0
#
# Results: outdir/{model}_{dataset}_butterfly_s{seed}/results.json

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

EXTRA="${*}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MODELS=(A B D D_full)
SEEDS=(42 123 7)

DATASETS=(
    pathmnist chestmnist dermamnist octmnist pneumoniamnist retinamnist
    breastmnist bloodmnist tissuemnist organamnist organcmnist organsmnist
)

for ds in "${DATASETS[@]}"; do
    for m in "${MODELS[@]}"; do
        # use model_b_retina.json as template, override dataset
        if [ "$m" = "D_full" ]; then
            cfg="configs/model_d_full_retina.json"
        else
            cfg="configs/model_$(echo $m | tr A-Z a-z)_retina.json"
        fi
        [ -f "$cfg" ] || { echo "SKIP $cfg"; continue; }

        for s in "${SEEDS[@]}"; do
            tag="${m}_${ds}_butterfly_s${s}"
            out="outdir/${tag}"
            [ "$FORCE_RERUN" != "1" ] && [ -d "$out" ] && { echo "SKIP ${tag} (folder exists; set FORCE_RERUN=1 to rerun)"; continue; }

            echo ""
            echo "════ Model ${m}  ${ds}  butterfly  seed ${s} ════"

            # Create a temporary config with the right dataset and butterfly layout
            tmp_cfg=$(mktemp)
            python -c "
import json, sys
with open('${cfg}') as f: c = json.load(f)
c['dataset'] = '${ds}'
c['model_type'] = 'D' if '${m}' == 'D_full' else '${m}'
c['circuit_family'] = 'butterfly'
if '${m}' == 'D_full':
    c['compound_readout'] = 'full_sector'
# Model D and D_full require use_cls_token=False for butterfly if n=16, d=16
if '${m}' in ('D', 'D_full'):
    c['use_cls_token'] = False
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
