#!/usr/bin/env bash
# run_all_retina_lite.sh — Lightweight RetinaMNIST benchmark for quick triage runs.
#
# Usage:
#   bash scripts/experiments/run_all_retina_lite.sh
#   bash scripts/experiments/run_all_retina_lite.sh --device cpu
#   MODELS="A B D E F" bash scripts/experiments/run_all_retina_lite.sh
#   SEEDS="42" bash scripts/experiments/run_all_retina_lite.sh
#
# Results: outdir/{model}_retina_lite_s{seed}/results.json
# Figures: python scripts/analysis/generate_figures.py outdir/ --profile lite

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

EXTRA="$*"
FORCE_RERUN="${FORCE_RERUN:-0}"
MODELS="${MODELS:-A B C D D_full E F}"
SEEDS_STRING="${SEEDS:-42 123 7}"
read -r -a SEEDS <<< "$SEEDS_STRING"

for m in $MODELS; do
    if [ "$m" = "D_full" ]; then
        cfg="configs/model_d_full_retina.json"
    else
        cfg="configs/model_$(echo "$m" | tr A-Z a-z)_retina.json"
    fi
    [ -f "$cfg" ] || { echo "SKIP $cfg (not found)"; continue; }

    for s in "${SEEDS[@]}"; do
        tag="${m}_retina_generic_lite_s${s}"
        out="outdir/${tag}"
        [ "$FORCE_RERUN" != "1" ] && [ -d "$out" ] && { echo "SKIP ${tag} (folder exists; set FORCE_RERUN=1 to rerun)"; continue; }

        echo ""
        echo "========================================"
        echo "  Model ${m}  RetinaMNIST  LITE  generic  seed ${s}"
        echo "========================================"

        mkdir -p "$out"
        tmp_cfg=$(mktemp)

        python - <<PY > "$tmp_cfg"
import json

with open("${cfg}") as f:
    c = json.load(f)

c["dataset"] = "retinamnist"
c["model_type"] = "D" if "${m}" == "D_full" else "${m}"
c["seed"] = ${s}
c["profile"] = "lite"
c["circuit_family"] = "generic"

if "${m}" == "D_full":
    c["compound_readout"] = "full_sector"

# lighter training budget
c["epochs"] = 100
c["n_layers"] = 1
c["batch_size"] = 16
c["embed_dim"] = 8

# keep patch_size=7 to preserve token structure (16 patches at 28x28)
# disable CLS for some heavier variants to shave off a little quantum cost
if c.get("model_type") in ("D", "E"):
    c["use_cls_token"] = False

json.dump(c, fp=open("/dev/stdout", "w"), indent=2)
PY

        python implementation.py \
            --config "$tmp_cfg" --seed "$s" --outdir "$out" \
            $EXTRA 2>&1 | tee "${out}.log"

        rm -f "$tmp_cfg"
    done
done

echo ""
echo "Done. Now run: python scripts/analysis/generate_figures.py outdir/ --profile lite"
