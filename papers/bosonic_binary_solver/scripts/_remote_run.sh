#!/bin/bash
# Runs on the GPU host, launched detached by run_gpu.sh. cwd = package root.
# Every config is one step; a failed step is recorded and the queue continues,
# so one bad config cannot cancel a long run.
set -u
STAMP=$1; shift
CHUNK=$1; shift
mkdir -p logs results
echo $$ > "logs/run_$STAMP.pid"      # our own pid, not nohup's or setsid's
status=0
failed=""
for cfg in "$@"; do
    echo "=== $cfg   ($(date -u +%H:%M:%S) UTC)"
    python3 -u utils/run_gpu.py "$cfg" --chunk "$CHUNK" --device cuda \
        --out results/gpu_results.jsonl \
        || { rc=$?; status=$rc; failed="$failed $cfg($rc)"; echo "STEP FAILED rc=$rc: $cfg"; }
done
echo "failed steps:${failed:- none}"
echo "rows in results/gpu_results.jsonl: $(wc -l < results/gpu_results.jsonl 2>/dev/null || echo 0)"
echo "EXIT $status"
echo "$status" > "logs/run_$STAMP.status"
