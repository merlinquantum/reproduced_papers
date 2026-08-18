#!/usr/bin/env bash
# run_all_retina.sh — QVT models on RetinaMNIST across circuit family × profile.
#
# One runner for the whole retina grid. Absorbs the former
# run_all_retina_lite.sh, run_all_retina_butterfly.sh,
# run_all_retina_butterfly_lite.sh, run_full_benchmark.sh,
# run_extensions_comparison.sh, run_sector_comparison.sh and
# run_hierarchical_comparison.sh via env vars.
#
# Env vars:
#   CIRCUIT_FAMILY  generic (default) | butterfly
#   PROFILE         full (default) | lite  (lite: 100 ep, 1 layer, small embed)
#   MODELS          model list; default depends on family/profile (see below)
#   SEEDS           default "42 123 7" (butterfly full: "42")
#   FORCE_RERUN     1 to rerun existing outdirs
#
# Usage:
#   bash scripts/experiments/run_all_retina.sh                                  # generic full, all models
#   CIRCUIT_FAMILY=butterfly bash scripts/experiments/run_all_retina.sh
#   PROFILE=lite bash scripts/experiments/run_all_retina.sh --device cpu
#   MODELS="B D D_full E F" bash scripts/experiments/run_all_retina.sh          # extensions comparison
#   MODELS="B D D_full"     bash scripts/experiments/run_all_retina.sh          # sector comparison
#
# NOTE: Model F (3 photons, 2600 Fock states) is ~5-10× slower per step than D.
#
# Results: outdir/{model}_retina_{family}[_lite]_s{seed}/results.json
# Figures: python scripts/analysis/generate_figures.py outdir/ [--profile lite] [--circuit-family butterfly]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

EXTRA="$*"
FORCE_RERUN="${FORCE_RERUN:-0}"
CIRCUIT_FAMILY="${CIRCUIT_FAMILY:-generic}"
PROFILE="${PROFILE:-full}"

case "$CIRCUIT_FAMILY" in generic|butterfly) ;; *) echo "CIRCUIT_FAMILY must be 'generic' or 'butterfly'"; exit 1 ;; esac
case "$PROFILE" in full|lite) ;; *) echo "PROFILE must be 'full' or 'lite'"; exit 1 ;; esac

# Historical per-variant defaults (butterfly needs power-of-two mode counts,
# so C is excluded; butterfly full defaults to a single seed for cost).
if [ -z "${MODELS:-}" ]; then
    if [ "$CIRCUIT_FAMILY" = "generic" ]; then
        MODELS="A B C D D_full E F"
    elif [ "$PROFILE" = "full" ]; then
        MODELS="A B D D_full E F"
    else
        MODELS="A B D D_full"
    fi
fi
if [ -z "${SEEDS:-}" ]; then
    if [ "$CIRCUIT_FAMILY" = "butterfly" ] && [ "$PROFILE" = "full" ]; then
        SEEDS="42"
    else
        SEEDS="42 123 7"
    fi
fi
read -r -a SEEDS_ARR <<< "$SEEDS"

SUFFIX="$CIRCUIT_FAMILY"
[ "$PROFILE" = "lite" ] && SUFFIX="${SUFFIX}_lite"

for m in $MODELS; do
    if [ "$m" = "D_full" ]; then
        cfg="configs/model_d_full_retina.json"
    else
        cfg="configs/model_$(echo "$m" | tr A-Z a-z)_retina.json"
    fi
    [ -f "$cfg" ] || { echo "SKIP $cfg (not found)"; continue; }

    for s in "${SEEDS_ARR[@]}"; do
        tag="${m}_retina_${SUFFIX}_s${s}"
        out="outdir/${tag}"
        [ "$FORCE_RERUN" != "1" ] && [ -d "$out" ] && { echo "SKIP ${tag} (folder exists; set FORCE_RERUN=1 to rerun)"; continue; }

        echo ""
        echo "════════════════════════════════════════"
        echo "  Model ${m}  RetinaMNIST  ${CIRCUIT_FAMILY}  ${PROFILE}  seed ${s}"
        echo "════════════════════════════════════════"

        mkdir -p "$out"
        tmp_cfg=$(mktemp --suffix=.json)

        python - <<PY > "$tmp_cfg"
import json

with open("${cfg}") as f:
    c = json.load(f)

c["dataset"] = "retinamnist"
c["model_type"] = "D" if "${m}" == "D_full" else "${m}"
c["seed"] = ${s}
c["circuit_family"] = "${CIRCUIT_FAMILY}"

if "${m}" == "D_full":
    c["compound_readout"] = "full_sector"

if "${PROFILE}" == "lite":
    # lite profile: reduced model size, full 100-epoch training trace
    c["profile"] = "lite"
    c["epochs"] = 100
    c["n_layers"] = 1
    c["batch_size"] = 16
    if "${CIRCUIT_FAMILY}" == "butterfly":
        # Butterfly compound variants need n_patches + cls + d to stay a power of two.
        c["embed_dim"] = 16 if c.get("model_type") in ("D", "E") else 8
    else:
        c["embed_dim"] = 8

# Compound variants drop the CLS token on butterfly (power-of-two mode count)
# and on lite (shaves quantum cost).
if ("${CIRCUIT_FAMILY}" == "butterfly" or "${PROFILE}" == "lite") and c.get("model_type") in ("D", "E"):
    c["use_cls_token"] = False

json.dump(c, fp=open("/dev/stdout", "w"), indent=2)
PY

        python ../../implementation.py --paper quantum_vision_transformers \
            --config "$tmp_cfg" --seed "$s" --outdir "$out" \
            $EXTRA 2>&1 | tee "${out}.log"

        rm -f "$tmp_cfg"
    done
done

echo ""
FIG_FLAGS=""
[ "$PROFILE" = "lite" ] && FIG_FLAGS=" --profile lite"
[ "$CIRCUIT_FAMILY" = "butterfly" ] && FIG_FLAGS="${FIG_FLAGS} --circuit-family butterfly"
echo "Done.  Now run:  python scripts/analysis/generate_figures.py outdir/${FIG_FLAGS}"
