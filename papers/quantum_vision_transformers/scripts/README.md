# Script Index

The `scripts/` tree is grouped by purpose rather than kept as one flat directory.

## Recommended entrypoints

- `scripts/validation/validate.sh`
  Run validation before long benchmarks.
- `scripts/reproduction/run_paper_retina_benchmark.sh`
  Paper Retina benchmark family.
- `scripts/suites/run_retina_cpu_suite.sh`
  CPU-friendly Retina reproduction workflow.
- `scripts/suites/run_medmnist_cpu_suite.sh`
  CPU-friendly MedMNIST workflow.
- `scripts/analysis/generate_figures.py`
  Regenerate figures from `outdir/`.

## Layout

- `analysis/`
  Figure generation and result summaries.
- `benchmarks/`
  Profiling and backend benchmark utilities.
- `experiments/`
  Exploratory runners, subset studies, and extension comparisons.
- `reproduction/`
  Paper-facing benchmark runners.
- `suites/`
  Higher-level wrappers that chain multiple benchmark families.
- `validation/`
  Pre-benchmark checks and smoke tests.

## Quick examples

```bash
bash scripts/validation/validate.sh
bash scripts/reproduction/run_paper_retina_benchmark.sh --device cpu
bash scripts/suites/run_retina_cpu_suite.sh --device cpu
python scripts/analysis/generate_figures.py outdir/
python scripts/benchmarks/benchmark_device_profile.py --devices cpu cuda:0 --precision-mode gpu_friendly
```
