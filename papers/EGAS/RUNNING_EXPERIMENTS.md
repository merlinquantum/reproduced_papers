# Running EGAS Reproduction Experiments

This directory contains scripts to easily run all EGAS reproduction experiments with progress tracking.

## Quick Start

### Python Script (Recommended)
```bash
# Run from the EGAS directory
cd papers/EGAS

# Run all experiments (tests + Wasserstein + Fig 1 + EGAS on 3 datasets + photonic)
python run_all_experiments.py

# Run only quick smoke test (~80s)
python run_all_experiments.py --quick

# Run only photonic experiments
python run_all_experiments.py --only-photonic

# Run only gate-based EGAS (skip photonic)
python run_all_experiments.py --only-gate

# Skip tests, run everything else
python run_all_experiments.py --skip-tests
```

### Bash Script
```bash
# Make executable
chmod +x run_all_experiments.sh

# Run all experiments
./run_all_experiments.sh

# Run with specific options (tests, gate EGAS, photonic)
./run_all_experiments.sh yes yes yes  # all enabled (default)
./run_all_experiments.sh no yes no    # skip tests and photonic
./run_all_experiments.sh yes no yes   # skip gate EGAS
```

## What Each Experiment Does

| Step | Command | Output | Time |
|------|---------|--------|------|
| Tests | `pytest tests/test_photonic_impl.py` | 7 test functions validating photonic impl | ~30s |
| Wasserstein | `configs/wasserstein.json` | Table I (input-space W1 distances) | ~5min |
| Fig 1 | `configs/fig1.json` | Trace distance vs W1 saturation curve | ~5min |
| EGAS (PW) | `configs/egas_PW.json` | Phishing dataset results (Figs 3–7) | ~15min |
| EGAS (WQ) | `configs/egas_WQ.json` | Wine Quality dataset results | ~10min |
| EGAS (MGT) | `configs/egas_MGT.json` | MAGIC Gamma Telescope dataset results | ~10min |
| Photonic | `configs/photonic_MGT.json` | MerLin photonic QKSVM results | ~20min |

**Total time (all experiments):** ~65 minutes (CPU-only, 10 cores)

## Output Locations

After running, results are saved to:
```
papers/EGAS/
├── outdir/
│   ├── wasserstein/         # Table I results
│   ├── fig1/                # Fig 1 results
│   ├── PW/                  # EGAS on Phishing
│   ├── WQ/                  # EGAS on Wine Quality
│   ├── MGT/                 # EGAS on MAGIC Gamma Telescope
│   └── photonic_MGT/        # Photonic results
└── results/                 # Generated plots (when available)
```

Each `outdir/*/` directory contains:
- `run_0/`, `run_1/`, ... — individual experiment runs
- `metrics.json` in each run directory with detailed results
- `*.txt` logs for reproducibility

## Individual Experiments

You can also run individual experiments manually:

```bash
# From repo root
cd ../..

# Wasserstein diagnostic
python implementation.py --paper generative_quantum_embeddings --config papers/EGAS/configs/wasserstein.json

# Fig 1
python implementation.py --paper generative_quantum_embeddings --config papers/EGAS/configs/fig1.json

# EGAS on specific dataset
python implementation.py --paper generative_quantum_embeddings --config papers/EGAS/configs/egas_PW.json --outdir papers/EGAS/outdir/PW

# Photonic implementation
python implementation.py --paper generative_quantum_embeddings --config papers/EGAS/configs/photonic_MGT.json --outdir papers/EGAS/outdir/photonic_MGT
```

## Configuration Files

All experiment configurations are in `configs/`:
- `wasserstein.json` — Wasserstein diagnostic (Table I)
- `fig1.json` — Trace distance saturation curve (Fig 1)
- `egas_PW.json`, `egas_WQ.json`, `egas_MGT.json` — EGAS on each dataset
- `photonic_MGT.json` — Photonic QKSVM on MGT
- `defaults.json` — Quick smoke test

## Tests

Run photonic implementation tests:
```bash
# All photonic tests
pytest tests/test_photonic_impl.py -v

# Specific test
pytest tests/test_photonic_impl.py::test_default_input_state_alternates_modes -v

# All EGAS tests (statevector + photonic)
pytest -q
```

## Troubleshooting

**Python not found:** Update the Python path in the script to match your environment:
```bash
which python  # Find your Python executable
# Update script accordingly
```

**Experiments run very slowly:** EGAS experiments are computationally intensive. Expected runtimes:
- Wasserstein: ~5 min
- EGAS on one dataset: ~10–15 min
- Photonic (all): ~20 min
- Reduce compute by using `--quick` option or modifying config files

**Out of memory:** Reduce batch sizes or number of iterations in the respective config file.

## Configuration Reference

Edit `configs/*.json` to customize experiments:
```json
{
  "seed": 0,
  "dataset": {"name": "phishing", "root": "data/..."},
  "egas": {
    "n_iters": 120,           // Number of search iterations
    "n_candidates": 12,       // Candidates per iteration
    "select_k": 6,            // Top/bottom selection size
    "n_repeats": 8            // Train/test splits
  },
  "gpt": {
    "d_model": 32,            // Hidden dimension
    "n_layers": 1             // Number of attention layers
  }
}
```

See `cli.json` for full authoritative schema.
