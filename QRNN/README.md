# QRNN — Quantum Recurrent Neural Networks (arXiv:2302.03244)

This project bootstraps a reproduction of the paper on quantum recurrent neural networks (QRNN, arXiv:2302.03244). It currently ships a classical RNN baseline over weather time-series data and a scaffold for extending the work toward the quantum architecture.

## Reference and Attribution

- Paper: Quantum Recurrent Neural Networks for sequential modelling (arXiv preprint, 2023)
- Authors: See arXiv:2302.03244 for the full list
- DOI/ArXiv: https://arxiv.org/abs/2302.03244
- Original repository (if any): not referenced here
- License and attribution notes: cite the arXiv preprint when using results derived from this code.

## Overview

The reproduction is staged:

- **Stage 1 (implemented here):** classical RNN baseline for sequence forecasting on a meteorological dataset.
- **Stage 2:** swap in the QRNN architecture described in the paper and compare against the baseline metrics.

Defaults target the Kaggle weather prediction dataset (`thedevastator/weather-prediction`) using a sliding-window regression setup to predict future temperature from recent temperature and humidity readings. A small synthetic CSV (`data/sample_weather.csv`) is included so the pipeline runs end-to-end without network access.

## How to Run

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Command-line interface

Main entry point: `implementation.py`.

```bash
python implementation.py --help
```

Common options (see `configs/cli.json` for the full schema):

- `--config PATH` Load an additional JSON config (merged over defaults).
- `--epochs INT` Override `training.epochs`.
- `--batch-size INT` Override `dataset.batch_size`.
- `--sequence-length INT` History length used for forecasting.
- `--hidden-dim INT` RNN hidden dimension.
- `--use-kagglehub` Auto-download the Kaggle dataset when the configured path is missing.
- Standard global flags: `--seed`, `--dtype`, `--device`, `--log-level`, `--outdir`.

Example runs:

```bash
# Use the bundled synthetic CSV for a quick smoke test
python implementation.py --config configs/example.json --epochs 1

# Train on the Kaggle dataset (downloads on first run if needed)
python implementation.py --config configs/example.json --use-kagglehub true --outdir runs/qrnn_baseline
```

### Dataset setup

The default configuration points to `data/sample_weather.csv` for convenience. To work with the Kaggle data:

1. Install `kagglehub` (already listed in `requirements.txt`).
2. Set `dataset.use_kagglehub` to `true` (via config or `--use-kagglehub`).
3. Optionally change `dataset.path` to a specific CSV inside the downloaded folder; when left as-is and the path is missing, the runner picks the first CSV it finds under the Kaggle download directory.

### Outputs

Each run writes to `<outdir>/run_YYYYMMDD-HHMMSS/` and includes:

- `config_snapshot.json` — resolved configuration used for the run
- `metrics.json` — train/validation loss history
- `metadata.json` — dataset and preprocessing metadata
- `rnn_baseline.pt` — PyTorch checkpoint for the baseline model
- `done.txt` — completion marker

## Configuration

Key files under `configs/`:

- `defaults.json` — baseline hyperparameters and dataset paths
- `example.json` — example experiment overriding sequence length and batch size
- `cli.json` — CLI schema consumed by the shared runner

Precision control: include `"dtype"` (e.g., `"float32"`) at the top level or under `model` to run in a specific torch dtype.

## Results and Next Steps

- Baseline metrics: mean squared error on validation splits of the weather sequences (see `metrics.json`).
- Planned extensions: implement the QRNN cell described in the paper, run ablations versus the classical RNN, and add visualization notebooks for sequence reconstruction.

## Testing

Run tests from inside the `QRNN/` directory:

```bash
cd QRNN
pytest -q
```

Tests cover the CLI, config loading, and a smoke run of the training loop on the synthetic dataset.
