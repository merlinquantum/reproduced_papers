#!/usr/bin/env bash
# run_full_benchmark.sh — Run every model and config, then generate figures.
#
# Models:
#   A, B, C, D         — paper reproduction
#   D_full, E, F       — extensions beyond the paper
#
# Each model is trained on RetinaMNIST with 3 seeds (42, 123, 7).
# Results land in outdir/{tag}/results.json.
# Figures are generated at the end in outdir/figures/.
#
# Already-completed runs are skipped (safe to re-run after interruption).
#
# Usage:
#   bash scripts/experiments/run_full_benchmark.sh
#   bash scripts/experiments/run_full_benchmark.sh --device cuda:0
#
# Estimated time (CPU, 100 epochs, 4 layers):
#   A, B, C:  ~5–15 min each
#   D:        ~30–60 min each
#   D_full:   ~30–60 min each
#   E:        ~45–90 min each  (two SLOS passes per step)
#   F:        ~2–4 hr each     (2600 Fock states)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

EXTRA="${*}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SEEDS=(42 123 7)
STARTED=$(date +%s)

# ── Define all runs ──
# Format: TAG CONFIG
RUNS=(
    "A          configs/model_a_retina.json"
    "B          configs/model_b_retina.json"
    "C          configs/model_c_retina.json"
    "D          configs/model_d_retina.json"
    "D_full     configs/model_d_full_retina.json"
    "E          configs/model_e_retina.json"
    "F          configs/model_f_retina.json"
)

TOTAL=0
DONE=0
SKIPPED=0
FAILED=0

echo "╔══════════════════════════════════════════════════════════╗"
echo "║           QVT Full Benchmark                            ║"
echo "║  ${#RUNS[@]} models × ${#SEEDS[@]} seeds = $(( ${#RUNS[@]} * ${#SEEDS[@]} )) runs                             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

for entry in "${RUNS[@]}"; do
    tag_prefix=$(echo "$entry" | awk '{print $1}')
    cfg=$(echo "$entry" | awk '{print $2}')

    if [ ! -f "$cfg" ]; then
        echo "WARN: $cfg not found, skipping $tag_prefix"
        continue
    fi

    for s in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        tag="${tag_prefix}_retina_s${s}"
        out="outdir/${tag}"

        if [ "$FORCE_RERUN" != "1" ] && [ -d "$out" ]; then
            SKIPPED=$((SKIPPED + 1))
            echo "SKIP  ${tag}  (folder exists; set FORCE_RERUN=1 to rerun)"
            continue
        fi

        echo ""
        echo "════════════════════════════════════════════════════"
        echo "  [${TOTAL}] ${tag_prefix}  seed=${s}"
        echo "════════════════════════════════════════════════════"

        mkdir -p "$out"
        if python implementation.py \
            --config "$cfg" --seed "$s" --outdir "$out" \
            $EXTRA 2>&1 | tee "${out}/train.log"; then
            if [ -f "${out}/results.json" ]; then
                DONE=$((DONE + 1))
                echo "  ✓ ${tag} complete"
            else
                FAILED=$((FAILED + 1))
                echo "  ✗ ${tag} — no results.json produced"
            fi
        else
            FAILED=$((FAILED + 1))
            echo "  ✗ ${tag} — crashed (see ${out}/train.log)"
        fi
    done
done

# ── Generate figures ──
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Generating figures..."
echo "════════════════════════════════════════════════════════"
python scripts/analysis/generate_figures.py outdir/ --out outdir/figures

# ── Summary ──
ELAPSED=$(( $(date +%s) - STARTED ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Benchmark complete                                      ║"
echo "║  Done: ${DONE}  Skipped: ${SKIPPED}  Failed: ${FAILED}  Total: ${TOTAL}             ║"
echo "║  Time: ${HOURS}h ${MINS}m                                          ║"
echo "║                                                          ║"
echo "║  Results:  outdir/*/results.json                         ║"
echo "║  Figures:  outdir/figures/                               ║"
echo "║  Summary:  outdir/figures/summary.csv                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
