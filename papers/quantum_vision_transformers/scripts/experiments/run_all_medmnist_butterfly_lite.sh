#!/usr/bin/env bash
# run_all_medmnist_butterfly_lite.sh — MedMNIST butterfly benchmark with lite-sized models.
#
# Usage:
#   bash scripts/experiments/run_all_medmnist_butterfly_lite.sh
#   bash scripts/experiments/run_all_medmnist_butterfly_lite.sh --device cuda:0
#   MODELS="A B D" bash scripts/experiments/run_all_medmnist_butterfly_lite.sh
#   SEEDS="42" bash scripts/experiments/run_all_medmnist_butterfly_lite.sh
#
# Results: outdir/{model}_{dataset}_butterfly_lite_s{seed}/results.json
# Figures: python scripts/analysis/generate_figures.py outdir/ --profile lite --circuit-family butterfly

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

EXTRA="$*"
FORCE_RERUN="${FORCE_RERUN:-0}"
MODELS="${MODELS:-A B D D_full}"
SEEDS_STRING="${SEEDS:-42 123 7}"
read -r -a SEEDS <<< "$SEEDS_STRING"

DATASETS=(
    pathmnist chestmnist dermamnist octmnist pneumoniamnist retinamnist
    breastmnist bloodmnist tissuemnist organamnist organcmnist organsmnist
)

for ds in "${DATASETS[@]}"; do
    for m in $MODELS; do
        if [ "$m" = "D_full" ]; then
            cfg="configs/model_d_full_retina.json"
        else
            cfg="configs/model_$(echo "$m" | tr A-Z a-z)_retina.json"
        fi
        [ -f "$cfg" ] || { echo "SKIP $cfg"; continue; }

        for s in "${SEEDS[@]}"; do
            tag="${m}_${ds}_butterfly_lite_s${s}"
            out="outdir/${tag}"
            [ "$FORCE_RERUN" != "1" ] && [ -d "$out" ] && { echo "SKIP ${tag} (folder exists; set FORCE_RERUN=1 to rerun)"; continue; }

            echo ""
            echo "════ Model ${m}  ${ds}  butterfly  lite  seed ${s} ════"

            mkdir -p "$out"
            tmp_cfg=$(mktemp)

            python - <<PY > "$tmp_cfg"
import json

with open("${cfg}") as f:
    c = json.load(f)

c["dataset"] = "${ds}"
c["model_type"] = "D" if "${m}" == "D_full" else "${m}"
c["seed"] = ${s}
c["profile"] = "lite"
c["circuit_family"] = "butterfly"

if "${m}" == "D_full":
    c["compound_readout"] = "full_sector"

# lite profile: reduced model size, full 100-epoch training trace
c["epochs"] = 100
c["n_layers"] = 1
c["batch_size"] = 16

# Butterfly compound variants need n_patches + cls + d to stay a power of two.
if c.get("model_type") in ("D", "E"):
    c["embed_dim"] = 16
else:
    c["embed_dim"] = 8

if c.get("model_type") in ("D", "E"):
    c["use_cls_token"] = False

json.dump(c, fp=open("/dev/stdout", "w"), indent=2)
PY

            python ../../implementation.py --paper quantum_vision_transformers \
                --config "$tmp_cfg" --seed "$s" --outdir "$out" \
                $EXTRA 2>&1 | tee "${out}.log"

            rm -f "$tmp_cfg"
        done
    done
done

echo ""
echo "Done. Now run: python scripts/analysis/generate_figures.py outdir/ --profile lite --circuit-family butterfly"
