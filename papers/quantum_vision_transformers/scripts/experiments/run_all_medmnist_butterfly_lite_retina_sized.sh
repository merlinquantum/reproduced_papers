#!/usr/bin/env bash
# run_all_medmnist_butterfly_lite_retina_sized.sh — Butterfly lite MedMNIST benchmark
# with RetinaMNIST-sized training subsets and full validation/test splits.
#
# Usage:
#   bash scripts/experiments/run_all_medmnist_butterfly_lite_retina_sized.sh
#   bash scripts/experiments/run_all_medmnist_butterfly_lite_retina_sized.sh --device cpu
#   SEEDS="42" bash scripts/experiments/run_all_medmnist_butterfly_lite_retina_sized.sh
#   DATASETS="pathmnist octmnist bloodmnist" bash scripts/experiments/run_all_medmnist_butterfly_lite_retina_sized.sh
#
# Results:
#   outdir/{model}_{dataset}_butterfly_lite_retina_sized_s{seed}/results.json
#
# Notes:
#   - Trains on a stratified 1080-example subset, matching RetinaMNIST train size.
#   - Validation and test use the full official splits.
#   - chestmnist is excluded by default because it is multi-label and not supported
#     by the current single-label stratified subset path.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

EXTRA="$*"
FORCE_RERUN="${FORCE_RERUN:-0}"
MODELS="${MODELS:-A B D D_full}"
SEEDS_STRING="${SEEDS:-42 123 7}"
read -r -a SEEDS <<< "$SEEDS_STRING"
TRAIN_SUBSET_SIZE="${TRAIN_SUBSET_SIZE:-1080}"
TRAIN_SUBSET_SEED="${TRAIN_SUBSET_SEED:-0}"

DATASETS_STRING="${DATASETS:-pathmnist dermamnist octmnist pneumoniamnist retinamnist breastmnist bloodmnist tissuemnist organamnist organcmnist organsmnist}"
read -r -a DATASETS <<< "$DATASETS_STRING"

for ds in "${DATASETS[@]}"; do
    for m in $MODELS; do
        if [ "$m" = "D_full" ]; then
            cfg="configs/model_d_full_retina.json"
        else
            cfg="configs/model_$(echo "$m" | tr A-Z a-z)_retina.json"
        fi
        [ -f "$cfg" ] || { echo "SKIP $cfg"; continue; }

        for s in "${SEEDS[@]}"; do
            tag="${m}_${ds}_butterfly_lite_retina_sized_s${s}"
            out="outdir/${tag}"
            [ "$FORCE_RERUN" != "1" ] && [ -d "$out" ] && { echo "SKIP ${tag} (folder exists; set FORCE_RERUN=1 to rerun)"; continue; }

            echo ""
            echo "════ Model ${m}  ${ds}  butterfly  lite  retina-sized train  seed ${s} ════"

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
c["data_regime"] = "retina_sized_train"
c["train_subset_size"] = int("${TRAIN_SUBSET_SIZE}")
c["train_subset_seed"] = int("${TRAIN_SUBSET_SEED}")
c["train_subset_mode"] = "stratified"

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
