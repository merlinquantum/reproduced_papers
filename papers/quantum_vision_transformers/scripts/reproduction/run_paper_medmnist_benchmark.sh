#!/usr/bin/env bash
# run_paper_medmnist_benchmark.sh — Paper benchmark family on all 12 MedMNIST datasets.
#
# Models:
#   VisionTransformer, OrthoFNN, A, B, D
#
# Usage:
#   bash scripts/reproduction/run_paper_medmnist_benchmark.sh
#   bash scripts/reproduction/run_paper_medmnist_benchmark.sh --device cuda:0

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

EXTRA="${*}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SEEDS=(42 123 7)
DATASETS=(
    pathmnist chestmnist dermamnist octmnist pneumoniamnist retinamnist
    breastmnist bloodmnist tissuemnist organamnist organcmnist organsmnist
)
MODELS="${MODELS:-vision_transformer orthofnn A B D}"
CONFIGS=()

for model in $MODELS; do
    case "$model" in
        vision_transformer|VisionTransformer)
            CONFIGS+=("vision_transformer configs/paper/model_vision_transformer_retina.json")
            ;;
        orthofnn|OrthoFNN)
            CONFIGS+=("orthofnn configs/paper/model_orthofnn_retina.json")
            ;;
        A|a)
            CONFIGS+=("A configs/paper/model_a_retina.json")
            ;;
        B|b)
            CONFIGS+=("B configs/paper/model_b_retina.json")
            ;;
        D|d)
            CONFIGS+=("D configs/paper/model_d_retina.json")
            ;;
        *)
            echo "SKIP unknown paper benchmark model '$model'"
            ;;
    esac
done

for ds in "${DATASETS[@]}"; do
    for entry in "${CONFIGS[@]}"; do
        tag_prefix=$(echo "$entry" | awk '{print $1}')
        cfg=$(echo "$entry" | awk '{print $2}')

        for s in "${SEEDS[@]}"; do
            tag="${tag_prefix}_${ds}_paper_s${s}"
            out="outdir/${tag}"
            [ "$FORCE_RERUN" != "1" ] && [ -d "$out" ] && { echo "SKIP ${tag} (folder exists; set FORCE_RERUN=1 to rerun)"; continue; }

            echo ""
            echo "========================================"
            echo "  Paper benchmark ${tag_prefix}  ${ds}  seed ${s}"
            echo "========================================"

            tmp_cfg=$(mktemp)
            python - <<PY > "$tmp_cfg"
import json
with open("${cfg}") as f:
    c = json.load(f)
c["dataset"] = "${ds}"
c["seed"] = ${s}
json.dump(c, fp=open("/dev/stdout", "w"), indent=2)
PY

            python implementation.py \
                --config "$tmp_cfg" --seed "$s" --outdir "$out" \
                $EXTRA 2>&1 | tee "${out}.log"

            rm -f "$tmp_cfg"
        done
    done
done

echo ""
echo "Done. Now run: python scripts/analysis/generate_figures.py outdir/"
