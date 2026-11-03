#!/bin/bash
# Multi-seed driver for the heat-equation matched-effort study and the
# consistency-loss ablation. Launches up to PARALLEL jobs at a time.
#
# Usage:
#     bash utils/launch_seeds.sh
#
# Final aggregation: python utils/aggregate_seeds.py.

set -u
cd "$(dirname "$0")/.."

PARALLEL=${PARALLEL:-2}
SEEDS=(42 7 123 256 1024)
RUN_LOG_DIR=/tmp/cv_qpinn_seed_runs
mkdir -p "$RUN_LOG_DIR"

queue=()

run_one() {
    local tag=$1
    local config=$2
    local seed=$3
    shift 3
    local extra="$*"
    # Subdir per (tag, seed, extra hash) so parallel runs don't collide on the
    # second-resolution timestamp in runtime_lib's default outdir naming.
    local extra_hash
    extra_hash=$(echo -n "$extra" | md5sum | cut -c1-6)
    local outdir="outdir/${tag}_seed${seed}_${extra_hash}"
    local log="$RUN_LOG_DIR/${tag}_seed${seed}_${extra_hash}.log"
    echo "[launch] tag=$tag seed=$seed config=$config extra='$extra' outdir=$outdir log=$log"
    (cd /reproduced_papers/papers/CV_QPINN_PDE && nohup python ../../implementation.py \
        --config "$config" --seed "$seed" --outdir "$outdir" $extra \
        > "$log" 2>&1) &
    queue+=($!)
    # Drain when queue exceeds PARALLEL.
    while (( ${#queue[@]} >= PARALLEL )); do
        wait -n
        # Compact queue by re-checking which are still alive.
        local fresh=()
        for pid in "${queue[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then fresh+=("$pid"); fi
        done
        queue=("${fresh[@]}")
    done
}

# Heat QPINN matched-effort study (5 seeds).
for s in "${SEEDS[@]}"; do
    run_one heat_qpinn configs/heat_smoke.json "$s"
done

# Heat PINN matched-effort study (5 seeds). Cheap, run after the QPINNs finish.
wait
for s in "${SEEDS[@]}"; do
    run_one heat_pinn configs/heat_pinn.json "$s"
done
wait

# Consistency-loss ablation: Poisson QPINN with nested autograd, two
# cutoffs to probe the paper's "nested gradients blow up memory" claim.
# Cutoff 15 nested takes too long under contention; we measure it
# separately via utils/measure_nested_overhead.py if needed.
for cutoff in 8 12; do
    run_one poisson_qpinn_nested configs/poisson_qpinn_nested.json 42 --cutoff "$cutoff"
done

# Companion: consistency Poisson QPINN at matching cutoffs.
for cutoff in 8 12; do
    run_one poisson_qpinn_cons configs/poisson_smoke.json 42 --cutoff "$cutoff"
done

# Heat ablation: nested vs consistency, smaller cutoff to keep the smoke fast.
run_one heat_qpinn_nested configs/heat_qpinn_nested.json 42
wait

echo "All runs finished."
