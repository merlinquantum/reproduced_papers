#!/bin/bash
# Beyond the paper: one launch for the scaling extension.
#
#   scripts/run_scaling_gpu.sh --host user@gpu --key ~/.ssh/id_ed25519 \
#       --clifford ~/DEV/claude_experiments/clifford_gpu/gpu-clifford
#
# Thin wrapper over run_gpu.sh with the scaling queue and a smaller lockstep
# chunk: at m=42 one chunk holds instances x 227 circuits x 50 shots, so 10
# instances per launch keeps the sampler's working set modest. About 3.8 GPU
# hours, dominated by the m=42 boson arm (2.9 h), which is deliberately last.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)

QUEUE=(
  # m=34 and m=38: all four arms, boson still cheap enough to pair by instance
  configs/scaling_m34_source_boson.json
  configs/scaling_m34_source_distinguishable.json
  configs/scaling_m34_source_bernoulli.json
  configs/scaling_m34_source_bernoulli_dense.json
  configs/scaling_m38_source_boson.json
  configs/scaling_m38_source_distinguishable.json
  configs/scaling_m38_source_bernoulli.json
  configs/scaling_m38_source_bernoulli_dense.json
  # the interference-free floor, nearly free, out to where nothing is found
  configs/scaling_m42_source_distinguishable.json
  configs/scaling_m42_source_bernoulli.json
  configs/scaling_m42_source_bernoulli_dense.json
  configs/scaling_m48_source_bernoulli.json
  configs/scaling_m48_source_bernoulli_dense.json
  configs/scaling_m54_source_bernoulli.json
  configs/scaling_m54_source_bernoulli_dense.json
  configs/scaling_m60_source_bernoulli.json
  configs/scaling_m60_source_bernoulli_dense.json
  # last, and on its own: 2.9 h. If it does not finish, nothing else is lost
  # and it resumes -- every completed run is skipped by key.
  configs/scaling_m42_source_boson.json
)

exec "$ROOT/scripts/run_gpu.sh" --chunk "${CHUNK:-10}" "$@" "${QUEUE[@]}"
