#!/bin/bash
# One launch: sync this package to a GPU host, verify the GPU is free, run the
# anchors while you wait, then start the whole queue detached.
#
#   scripts/run_gpu.sh --host user@gpu --key ~/.ssh/id_ed25519 \
#       --clifford ~/DEV/claude_experiments/clifford_gpu/gpu-clifford
#
#   scripts/run_gpu.sh --host user@gpu --key ~/.ssh/id_ed25519 --fetch
#
# With no config arguments the queue in QUEUE below is used, ordered so that the
# most informative experiments finish first.
set -u

HOST=${HOST:-}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}
CLIFFORD=${CLIFFORD:-$HOME/DEV/claude_experiments/clifford_gpu/gpu-clifford}
REMOTE=${REMOTE:-bbs_repro}
CHUNK=${CHUNK:-20}
MODE=launch
STAMP=""
CONFIGS=()

QUEUE=(
  # 1. the paper rows that cannot be run on CPU (9 h and 43 h per row there)
  configs/knapsack_m25_original.json
  configs/knapsack_m30_original.json
  configs/tsp_m29_original.json
  # 2. the control the paper never runs, at the only sizes that can separate anything
  configs/knapsack_m25_source_shuffled_boson.json
  configs/knapsack_m25_source_distinguishable.json
  configs/knapsack_m25_source_bernoulli.json
  configs/knapsack_m30_source_shuffled_boson.json
  configs/knapsack_m30_source_distinguishable.json
  configs/knapsack_m30_source_bernoulli.json
  # 3. click density, the known confound between those sources
  configs/knapsack_m30_density_r087.json
  # rate 1.0 is knapsack_m30_source_bernoulli above -- same arm, so it is not repeated
  configs/knapsack_m30_density_r115.json
  configs/knapsack_m30_density_r13.json
  configs/knapsack_m30_density_r15.json
  # 4. the paper's own ablation
  configs/knapsack_m25_frozen_theta.json
  configs/knapsack_m30_frozen_theta.json
  # 5. the two conventions the paper leaves open (see README.md)
  configs/knapsack_m25_shift_pi6.json
  configs/knapsack_m25_shift_scale05.json
  # 6. the circuit the paper describes against the one ORCA ships
  configs/knapsack_m20_topology_loop.json
  configs/knapsack_m20_original.json
  configs/knapsack_m25_topology_loop.json
  # 7. TSP controls, where the ordering may differ from knapsack
  configs/tsp_m29_source_shuffled_boson.json
  configs/tsp_m29_source_distinguishable.json
  configs/tsp_m29_source_bernoulli.json
)

usage() {
    sed -n '2,12p' "$0" | sed 's/^# \?//'
    echo "  --host USER@HOST   GPU host (or set HOST=)"
    echo "  --key PATH         private key for ssh -i (or set SSH_KEY=)"
    echo "  --clifford PATH    local gpu-clifford checkout (or set CLIFFORD=)"
    echo "  --remote PATH      remote directory under \$HOME, default bbs_repro"
    echo "  --chunk N          instances solved in lockstep, default 20"
    echo "  --fetch            retrieve log and results instead of launching"
    echo "  --stamp STAMP      with --fetch, a specific run; default the latest"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --host) HOST=$2; shift 2 ;;
        --key) SSH_KEY=$2; shift 2 ;;
        --clifford) CLIFFORD=$2; shift 2 ;;
        --remote) REMOTE=$2; shift 2 ;;
        --chunk) CHUNK=$2; shift 2 ;;
        --fetch) MODE=fetch; shift ;;
        --stamp) STAMP=$2; shift 2 ;;
        -h|--help) usage 0 ;;
        --*) echo "unknown option: $1" >&2; usage 2 ;;
        *) CONFIGS+=("$1"); shift ;;
    esac
done

[ -n "$HOST" ] || { echo "no host: pass --host user@server or set HOST=" >&2; exit 2; }
[ -r "$SSH_KEY" ] || { echo "key not readable: $SSH_KEY (pass --key PATH)" >&2; exit 2; }
case "$SSH_KEY" in *.pub) echo "warning: --key looks like a public key; ssh -i wants the private key" >&2 ;; esac

SSH="ssh -o IdentitiesOnly=yes -i $SSH_KEY"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/logs" "$ROOT/results"

if [ "$MODE" = fetch ]; then
    if [ -z "$STAMP" ]; then
        STAMP=$($SSH "$HOST" "ls -1 ~/$REMOTE/logs/run_*.log 2>/dev/null | tail -1" | sed 's|.*/run_||; s|\.log$||')
    fi
    [ -n "$STAMP" ] || { echo "no runs found in ~$REMOTE/logs on $HOST" >&2; exit 1; }
    rsync -az -e "$SSH" "$HOST:~/$REMOTE/logs/run_$STAMP.log" "$ROOT/logs/" 2>/dev/null
    rsync -az -e "$SSH" "$HOST:~/$REMOTE/results/gpu_results.jsonl" "$ROOT/results/gpu_results_$STAMP.jsonl" 2>/dev/null
    state=$($SSH "$HOST" "cd ~/$REMOTE && if [ -f logs/run_$STAMP.status ]; then echo \"finished \$(cat logs/run_$STAMP.status)\"; elif kill -0 \$(cat logs/run_$STAMP.pid 2>/dev/null) 2>/dev/null; then echo running; else echo 'gone (no status file, process not alive)'; fi")
    rows=$(wc -l < "$ROOT/results/gpu_results_$STAMP.jsonl" 2>/dev/null || echo 0)
    echo "run:     $STAMP"
    echo "state:   $state"
    echo "log:     $ROOT/logs/run_$STAMP.log"
    echo "results: $ROOT/results/gpu_results_$STAMP.jsonl  ($rows rows)"
    echo "--- tail of log:"
    tail -25 "$ROOT/logs/run_$STAMP.log" 2>/dev/null
    exit 0
fi

[ ${#CONFIGS[@]} -gt 0 ] || CONFIGS=("${QUEUE[@]}")
for cfg in "${CONFIGS[@]}"; do
    case "$cfg" in /*) echo "config must be package-relative: $cfg" >&2; exit 2 ;; esac
    [ -r "$ROOT/$cfg" ] || { echo "config not found: $ROOT/$cfg" >&2; exit 2; }
done
[ -d "$CLIFFORD/cliffordgpu" ] || { echo "cliffordgpu not found under $CLIFFORD (pass --clifford PATH)" >&2; exit 2; }

STAMP=$(date +%Y%m%d_%H%M%S)
echo "host:    $HOST"
echo "configs: ${#CONFIGS[@]}"

rsync -az -e "$SSH" --exclude __pycache__ --exclude logs --exclude outdir --exclude '.venv' \
      --exclude '_package.tgz' --exclude '_template_leftovers' "$ROOT/" "$HOST:~/$REMOTE/"
rsync -az -e "$SSH" --exclude __pycache__ "$CLIFFORD/cliffordgpu" "$HOST:~/$REMOTE/"

# Synchronous preflight: the GPU must be idle and the anchors must pass. Sampled
# three times rather than once because a job between kernels looks idle in a
# single snapshot, and starting on top of someone else's run wastes both.
$SSH "$HOST" 'bash -s' <<PREFLIGHT || { echo "preflight failed - nothing launched" >&2; exit 3; }
set -e
cd ~/$REMOTE && mkdir -p logs results
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
for i in 1 2 3; do
    PROCS=\$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
    USED=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    UTIL=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    if [ -n "\$PROCS" ]; then echo "GPU BUSY - compute process present:"; echo "\$PROCS"; exit 3; fi
    if [ "\${USED:-0}" -gt 1024 ]; then echo "GPU BUSY - \${USED} MiB in use"; exit 3; fi
    if [ "\${UTIL:-0}" -gt 5 ]; then echo "GPU BUSY - utilisation \${UTIL}%"; exit 3; fi
    [ \$i -lt 3 ] && sleep 2
done
echo "GPU idle across 3 samples (0 processes, \${USED} MiB, \${UTIL}% util)"
python3 -c "import torch, triton, numpy" 2>/dev/null || python3 -m pip install -q --break-system-packages torch triton numpy 2>&1 | tail -3
python3 -c "import perceval" 2>/dev/null || python3 -m pip install -q --break-system-packages perceval-quandela 2>&1 | tail -3
python3 -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
# Report what is present before running anything: the queue needs torch, numpy and
# cliffordgpu only. Perceval is optional and used by one cross-check; MerLin is
# never needed here. A missing optional package should read as a skipped check,
# not as a launch failure.
python3 - <<'DEPS'
import importlib.util
for name, role in (("torch", "required"), ("numpy", "required"), ("cliffordgpu", "required: GPU sampler"),
                   ("perceval", "optional: boson-vs-Perceval cross-check"),
                   ("merlin", "not needed on this host")):
    found = importlib.util.find_spec(name) is not None
    print(f"  {name:12} {'present' if found else 'ABSENT ':8} ({role})")
DEPS
python3 scripts/gpu_anchors.py --device cuda
PREFLIGHT

# Detached production run. Written as a heredoc because in
#   cd dir && cmd & echo \$! > dir/file
# the & backgrounds the whole && list and the pid file lands in the wrong place.
$SSH "$HOST" 'bash -s' <<LAUNCH
set -e
cd ~/$REMOTE
mkdir -p logs results
rm -f logs/run_$STAMP.status logs/run_$STAMP.pid
nohup setsid bash scripts/_remote_run.sh $STAMP $CHUNK ${CONFIGS[*]} > logs/run_$STAMP.log 2>&1 < /dev/null &
disown || true
for _ in 1 2 3 4 5; do sleep 1; [ -s logs/run_$STAMP.pid ] && break; done
if [ -s logs/run_$STAMP.pid ]; then
    echo "started, pid \$(cat logs/run_$STAMP.pid)"
else
    echo "WARNING: no pid file after 5s - check logs/run_$STAMP.log" >&2
    tail -5 logs/run_$STAMP.log 2>/dev/null
    exit 1
fi
LAUNCH

echo
echo "launched, detached. stamp: $STAMP"
echo "fetch with:"
echo "  scripts/run_gpu.sh --host $HOST --key $SSH_KEY --fetch"
