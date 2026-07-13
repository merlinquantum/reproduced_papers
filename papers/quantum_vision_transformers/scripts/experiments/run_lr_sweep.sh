#!/usr/bin/env bash
# run_lr_sweep.sh — Learning rate sweep with separate classical/quantum lr.
#
# Tests a grid of (lr_classical, lr_quantum) combinations for 30 epochs.
#
# Usage:
#   bash scripts/experiments/run_lr_sweep.sh
#   bash scripts/experiments/run_lr_sweep.sh --device cuda
#   MODELS="A B" bash scripts/experiments/run_lr_sweep.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

EXTRA="${*}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MODELS="${MODELS:-A B D}"
LR_CLASSICAL="0.001 0.0003"
LR_QUANTUM="0.1 0.05 0.01 0.003"
SEED=42
EPOCHS=30

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LR Sweep: ${EPOCHS} epochs per config                              ║"
echo "║  Models: ${MODELS}                                              ║"
echo "║  lr_classical: ${LR_CLASSICAL}                              ║"
echo "║  lr_quantum:   ${LR_QUANTUM}                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

for m in $MODELS; do
    cfg="configs/model_$(echo $m | tr A-Z a-z)_retina.json"
    [ -f "$cfg" ] || { echo "SKIP $cfg"; continue; }

    for lrc in $LR_CLASSICAL; do
        for lrq in $LR_QUANTUM; do
            tag="sweep_${m}_lrc${lrc}_lrq${lrq}"
            out="outdir/${tag}"
            [ "$FORCE_RERUN" != "1" ] && [ -d "$out" ] && { echo "SKIP ${tag} (folder exists; set FORCE_RERUN=1 to rerun)"; continue; }

            echo "════ ${m}  lr=${lrc}  lr_q=${lrq} ════"

            tmp=$(mktemp --suffix=.json)
            python3 -c "
import json
with open('${cfg}') as f: c = json.load(f)
c['epochs'] = ${EPOCHS}
c['lr'] = ${lrc}
c['lr_quantum'] = ${lrq}
c['lr_milestones'] = [20, 25]
json.dump(c, open('${tmp}', 'w'), indent=2)
"
            python ../../implementation.py --paper quantum_vision_transformers --config "$tmp" --seed $SEED --outdir "$out" \
                $EXTRA 2>&1 | tail -5
            rm -f "$tmp"
        done
    done
done

# ── Summary ──
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  LR Sweep Results"
echo "══════════════════════════════════════════════════════════════"

for m in $MODELS; do
    echo ""
    echo "  Model ${m}:"
    printf "  %-10s %-10s %-10s %-10s %-10s %-8s\n" "lr_class" "lr_quant" "val_auc" "test_auc" "test_acc" "best_ep"
    echo "  $(printf '%.0s-' {1..60})"
    for lrc in $LR_CLASSICAL; do
        for lrq in $LR_QUANTUM; do
            out="outdir/sweep_${m}_lrc${lrc}_lrq${lrq}"
            if [ -f "${out}/results.json" ]; then
                python3 -c "
import json
with open('${out}/results.json') as f: r = json.load(f)
print(f'  {${lrc}:<10}  {${lrq}:<10}  {r[\"best_val_auc\"]:<10.4f}  {r[\"test_auc\"]:<10.4f}  {r[\"test_acc\"]:<10.4f}  {r[\"best_epoch\"]}')
"
            fi
        done
    done
done

echo ""
echo "Update configs/ with the best (lr, lr_quantum) pair, then run:"
echo "  bash scripts/experiments/run_all_retina.sh && python scripts/analysis/generate_figures.py outdir/"
