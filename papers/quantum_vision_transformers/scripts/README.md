# Script Index

The `scripts/` tree is grouped by purpose rather than kept as one flat directory.

## Recommended entrypoints

- `scripts/validation/validate.sh`
  Run validation before long benchmarks.
- `scripts/reproduction/run_paper_retina_benchmark.sh`
  Paper Retina benchmark family.
- `scripts/suites/run_retina_suite.sh`
  Retina benchmark campaign (`CPU_FRIENDLY=1` for the reduced CPU workflow).
- `scripts/suites/run_medmnist_suite.sh`
  MedMNIST campaign (`CPU_FRIENDLY=1` for the reduced CPU workflow).
- `scripts/analysis/generate_figures.py`
  Regenerate figures from `outdir/`.

## Runner contract

The paper is now discoverable through the repo-root shared CLI:

```bash
python implementation.py --paper quantum_vision_transformers --config papers/quantum_vision_transformers/configs/paper/model_a_retina.json
```

The shell wrappers in this directory already invoke that root runner internally.

## Layout

- `analysis/`
  Figure generation and result summaries.
- `benchmarks/`
  Profiling and backend benchmark utilities.
- `experiments/`
  Grid runners (`run_all_retina.sh`, `run_all_medmnist.sh` — parameterized by
  `CIRCUIT_FAMILY`/`PROFILE`/`MODELS`/`SEEDS`), the capped-subset study, and the LR sweep.
- `reproduction/`
  Paper-facing benchmark runners.
- `suites/`
  Campaign wrappers that chain the paper benchmark with the grid runners.
- `validation/`
  Pre-benchmark checks and smoke tests.

## Quick examples

```bash
bash scripts/validation/validate.sh
bash scripts/reproduction/run_paper_retina_benchmark.sh --device cpu
CPU_FRIENDLY=1 bash scripts/suites/run_retina_suite.sh --device cpu
CIRCUIT_FAMILY=butterfly PROFILE=lite bash scripts/experiments/run_all_retina.sh --device cpu
python scripts/analysis/generate_figures.py outdir/
python scripts/benchmarks/benchmark_device_profile.py --devices cpu cuda:0 --precision-mode gpu_friendly
```
