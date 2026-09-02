#!/usr/bin/env bash
# Launches the 6 statistically-powered CV runs (100 folds each) in THREE
# memory-safe batches of 2 (revised 2026-09-01 after a 3-concurrent attempt
# caused severe swap thrashing on this 7.5GB container -- 3x ~1.7-2GB RSS
# processes pushed the machine into swap, slowing splits from ~55-65s to
# ~180s each and risking another OOM kill). 2 concurrent gate/photonic-scale
# jobs (~1.5-2GB RSS each once warmed up, front-loaded during the first
# couple of folds from library/JIT/device-construction overhead, not a
# per-fold leak) stay comfortably under the RAM limit.
#
# Each run uses its own --outdir to avoid the run_dir timestamp-collision
# bug discovered 2026-09-01 (concurrent processes starting within the same
# wall-clock second land on the same outdir/run_<timestamp>/ path and
# corrupt each other's config_snapshot.json / run.log / metrics.json).
#
# Usage: bash utils/launch_powered_runs.sh   (run from papers/MoE_fraud_detection/)
# Or:    bash papers/MoE_fraud_detection/utils/launch_powered_runs.sh  (from repo root)
#
# Batches paired by similar expected duration to minimize total makespan
# (sum of per-batch max, ~90+67+42 min):
#   Batch 1 (~90 min): moe_gate_tuned_xgboost_powered, moe_gate_powered
#   Batch 2 (~67 min): moe_gate_validation_router_powered, moe_photonic_trainable_powered
#   Batch 3 (~42 min): moe_photonic_fixed_powered, moe_classical_powered
#
# Total expected wall-clock: ~3.3 hours. Logs stream to
# outdir/<config_name>/run_<timestamp>/run.log per job; stdout/stderr also
# captured under $LOGDIR below for quick tailing.

set -euo pipefail
cd "$(dirname "$0")/.."   # papers/MoE_fraud_detection/

LOGDIR=/tmp/reproductions/mix_experts/scratch/powered_runs_logs
mkdir -p "$LOGDIR"

BATCH_1=(moe_gate_tuned_xgboost_powered moe_gate_powered)
BATCH_2=(moe_gate_validation_router_powered moe_photonic_trainable_powered)
BATCH_3=(moe_photonic_fixed_powered moe_classical_powered)

run_batch() {
    local pids=()
    for cfg in "$@"; do
        OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 \
            python ../../implementation.py --paper MoE_fraud_detection \
            --config "configs/${cfg}.json" --outdir "outdir/${cfg}" \
            > "$LOGDIR/${cfg}.out" 2>&1 &
        pids+=($!)
        echo "launched $cfg pid=$!"
    done
    echo "waiting on batch: ${pids[*]}"
    wait "${pids[@]}"
}

echo "=== Batch 1 ==="
run_batch "${BATCH_1[@]}"
echo "=== Batch 1 done, starting Batch 2 ==="
run_batch "${BATCH_2[@]}"
echo "=== Batch 2 done, starting Batch 3 ==="
run_batch "${BATCH_3[@]}"
echo "=== All 6 powered runs complete ==="
