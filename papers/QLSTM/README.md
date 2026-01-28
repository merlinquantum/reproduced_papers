# Quantum Long Short-Term Memory (QLSTM) — Reproduction

Clean, modular PyTorch + PennyLane re-implementation of the "Quantum Long Short-Term Memory" (QLSTM) model (arXiv:2009.01783) extended with a MerLin photonic variant.

## Introduction

Classical LSTMs are a staple for sequence modeling: their gates (input, forget, output, candidate) regulate the flow of temporal information. The paper "Quantum Long Short-Term Memory" introduces a hybrid quantum-classical design where each gate is replaced by a variational quantum circuit (VQC). The core idea is to leverage superposition and entanglement to mix temporal dependencies more richly while retaining the familiar LSTM control structure.

Key points emphasized in the paper:

1. Convergence: on certain synthetic temporal datasets, a QLSTM can reach better accuracy or converge faster than a similarly sized classical LSTM.
2. Parameter parsimony: gate VQCs may use fewer effective trainable parameters (rotation angles) than dense matrices, especially when qubits are re-used through a shallow ansatz.
3. NISQ suitability: the variational setup limits depth and qubit count, making the approach feasible for near-term noisy devices.
4. Structural expressivity: mapping (x_t, h_{t-1}) into measurement space (Pauli expectations) yields a nonlinear feature space induced by quantum interference.
5. Modular integration: the QLSTM block slots in as a drop-in replacement for the classical gate affine transforms without changing the broader training loop.

In this reproduction we:

- Integrate and lightly refactor the authors’ original gate-based (qubit) implementation (see linked repository); changes focus on modular structure, reproducibility, and configuration handling.
- Add a photonic QLSTM implementation illustrating how interferometers + phase shifts could serve as gate VQCs.
- Include a real-world time-series dataset (`data/airline-passengers.csv`) alongside synthetic generators to evaluate generalization on actual time serie.
- Standardize interface (CLI + JSON configs) for reproducible comparisons.

## Reference and Attribution

- Paper: Quantum Long Short-Term Memory (arXiv, 2020)
- Authors: Samuel Yen-Chi Chen, Shinjae Yoo, Yao-Lung L. Fang
- DOI: https://doi.org/10.48550/arXiv.2009.01783
- Link: https://arxiv.org/abs/2009.01783
- Original repository (inspiration): https://github.com/ycchen1989/Quantum_Long_Short_Term_Memory
- License: MIT-compatible. Please cite the original article (BibTeX below).

## Overview

Goal: reproduce a quantum LSTM by replacing classical gates (input/forget/cell/output) with VQCs (variational quantum circuits). This reproduction provides:

- A standardized CLI (JSON config + inline overrides)
- A timestamped per-run output folder under `outdir/run_YYYYMMDD-HHMMSS`
- Logging to `run.log` and a configuration snapshot `config_snapshot.json`
- Minimal artifacts (checkpoint, loss curves, simulation plots)
- Simple smoke tests for the CLI and artifact creation

Stack: PyTorch, PennyLane, MerLin, NumPy/SciPy, scikit‑learn, matplotlib.

## Folder Structure

- `../implementation.py --paper QLSTM` — Repository-level CLI entry point powered by this folder's `cli.json`
- `lib/model.py` — Classical LSTM, QLSTM (gate VQCs), sequence wrapper
- `lib/dataset.py` — Synthetic generators + CSV loader (delegates to `papers/shared/QLSTM/dataset.py`)
- `lib/rendering.py` — Plotting and pickle helpers
- `configs/` — JSON configs (`defaults.json`, dataset presets, `cli.json`)
- `data/` — Input datasets (CSV, etc.)
- `outdir/` — Output base directory (ignored by Git), contains timestamped runs
- `results/` — Collected figures from helper scripts (e.g., `utils/run_all_configs.sh`)
- `tests/` — Unit tests (CLI and artifacts smoke tests)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Command-line Interface

Main entry point: repository root `implementation.py` with `--paper QLSTM`. The available arguments live in `cli.json`, so you can extend the CLI by editing JSON rather than Python.

```bash
python implementation.py --paper QLSTM --help
```

Highlights (see `cli.json` for the authoritative list):

- `--config PATH`: merge an additional JSON config into `configs/defaults.json` (deep merge).
- `--seed INT`, `--outdir DIR`, `--device STR`: generic runtime overrides.
- `--model {qlstm,lstm,qlstm_photonic}` with `--use-preencoders` / `--use-photonic-head` flags.
- Dataset controls: `--generator {sin,damped_shm,logsine,ma_noise,csv}`, `--csv-path PATH`, `--seq-length INT`, `--train-split FLOAT`.
- Model hyperparameters: `--hidden-size INT`, `--vqc-depth INT`, `--shots INT`.
- Training overrides: `--epochs INT`, `--batch-size INT`, `--lr FLOAT`.
- Visualization: `--fmt {png,pdf}`, `--snapshot-epochs 1,15,...`, `--plot-width FLOAT`.

Example runs:

```bash
# From a JSON config
python implementation.py --paper QLSTM --config configs/example.json

# Inline overrides
python implementation.py --paper QLSTM --config configs/example.json --epochs 50 --lr 1e-3

# Minimal quantum “smoke” run
python implementation.py --paper QLSTM --model qlstm --generator sin --epochs 3 --seq-length 8 --hidden-size 4 --vqc-depth 2

# Quantum with 6‑VQC pre-encoders (closer to paper’s Sec. B)
python implementation.py --paper QLSTM --model qlstm --use-preencoders --generator sin --epochs 3 --seq-length 8 --hidden-size 4 --vqc-depth 2

# Classical baseline
python implementation.py --paper QLSTM --model lstm --generator damped_shm --epochs 10 --seq-length 6 --hidden-size 8

# Quick snapshot demo (2 epochs, save both e1 and e2, custom width)
python implementation.py --paper QLSTM --model lstm --generator sin --epochs 2 --seq-length 8 --hidden-size 4 \
  --snapshot-epochs 1,2 --plot-width 7

# Experimental: Photonic QLSTM (requires merlinquantum)
# Note: this can be slow in simulation; recommended for targeted tests
python implementation.py --paper QLSTM --model qlstm_photonic --generator sin --epochs 1 --seq-length 4 --hidden-size 2 --device cpu
```

### Data location

- Default data root is the shared repo `data/` directory (resolved via `paper_data_dir("QLSTM")`).
- On first use the loader copies the bundled `data/airline-passengers.csv` from this paper folder into the shared `data/QLSTM/airline-passengers.csv` location if missing.
- Relative `--csv-path` values are resolved under the shared `data/QLSTM/` folder; absolute paths are honored.
- Override the base with `DATA_DIR=/abs/path` or `--data-root /abs/path` when invoking `implementation.py`.
- If the resolver still cannot find the CSV, provide `--csv-path` explicitly or place the file in the chosen data root.

## Output Directory and Artifacts

Each run creates a timestamped folder under `outdir/`:

```
outdir/run_YYYYMMDD-HHMMSS/
├── config_snapshot.json         # Resolved configuration for the run
├── run.log                      # Execution logs
├── model_final.pth              # Final checkpoint
├── last_TRAINING_LOSS_<ts>.pkl  # Training loss history (timestamped)
├── last_TESTING_LOSS_<ts>.pkl   # Testing loss history (timestamped)
├── train_plot_<ts>.png          # Loss curve (timestamped)
├── simulation_<ts>.png          # Predictions vs ground truth at final epoch (timestamped)
├── simulation_e{epoch}_<ts>.png # Optional intermediate simulations (when snapshot epochs are set)
└── simulation_data_e{epoch}.npz # Raw arrays for each saved epoch (y, y_pred, n_train, epoch, losses)
```

Notes:
- Change the base directory via `--outdir` or in `configs/example.json` (`outdir`).

## Configuration

Place JSON files in `configs/`.

- `defaults.json` is the base experiment (sine generator with QLSTM).
- Dataset/model presets (e.g., `sine_qlstm.json`) can be layered via `--config`.
- `cli.json` defines every CLI argument, its type, and the config key it overrides.
- The shared runtime automatically reads `configs/defaults.json`, `cli.json`, and imports `lib/runner.py::train_and_evaluate`.
- Keys follow the same structure as before: `experiment`, `model`, `training`, `logging`, etc. CLI overrides mutate these dictionaries via the dot-paths defined in `cli.json`.

## Available Generators

Current keys (via `--generator`): `sin`, `damped_shm`, `logsine`, `ma_noise`, `csv`.
- CSV mode: provide `--csv-path` (uses the 2nd numeric column if available). All signals are scaled to `[-1, 1]`. A window of length `seq_length` predicts the next value.

### Airline Passengers (Real-world)

When using `--generator csv` with `data/airline-passengers.csv`, you are loading the classic monthly international airline passengers time series (1949–1960), often referenced as the "AirPassengers" dataset. It exhibits:

- Strong multiplicative seasonality (peak summer months grow proportionally with trend).
- Long-term upward trend tied to post-war expansion of air travel.
- Monthly frequency, 144 observations (12 years × 12 months), values expressed as passenger counts (originally in thousands).

Original source attribution (as cited in the R `datasets` package documentation):

> Box, G. E. P.; Jenkins, G. M.; Reinsel, G. C. (1994). *Time Series Analysis: Forecasting and Control* (3rd ed.). Prentice Hall. ISBN 978-0130607744.

Useful online references:

- R manual entry: https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/AirPassengers.html
- Mirror (CSV) / public distribution: Kaggle dataset "International airline passengers" — https://www.kaggle.com/datasets/andreazzini/international-airline-passengers

Recommended citation (short form) if reporting results on this dataset:

> AirPassengers dataset (Monthly International Airline Passengers, 1949–1960), originally compiled in Box & Jenkins time series studies; accessed via R `datasets` package.

Preprocessing notes in this repo: values are min–max scaled to [-1, 1] before windowed sequence construction; no log or seasonal differencing is applied by default (you may add those in a future extension if benchmarking classical ARIMA vs QLSTM).

## Modeling Notes

### Gate-Based QLSTM (Qubits)
Each LSTM gate (input, forget, cell/update, output) is replaced by a variational quantum circuit (VQC). A typical ansatz in `gatebased_quantum_cell.py`:

1. Hadamard layer on all qubits (creates |+> superpositions).
2. Feature embedding via single-qubit RY rotations on the first `n_features` wires (remaining wires stay in |+> if pre-encoders are enabled and hidden size > input size).
3. For each depth layer: alternating CNOT entangling pattern (even pairs then odd pairs) + another round of per-qubit RY rotations with trainable angles.
4. Measurement: expectation values of PauliZ on the first `hidden_size` wires (or all wires if encoders straddle concatenated embeddings). These expectation values form the gate activation vector prior to classical nonlinearity (sigmoid/tanh).

Modes:
- 4‑VQC (default): gates consume concat([x_t, h_{t-1}]).
- 6‑VQC (pre-encoders): two additional VQC map x and h separately into R^{hidden_size}; their outputs are concatenated and fed to the four gate VQCs. This mirrors the factorization (Eqs. 5a–5g) and can improve representation decoupling at the cost of extra parameters.

Shots: by default expectations are analytic (`shots=None`); enabling finite shots introduces sampling noise akin to running on real hardware.

### Photonic QLSTM (Linear Optics Simulation)
Implemented in `photonic_quantum_cell.py` using Merlin + Perceval libraries to model interferometers:

Architecture per gate (for concatenated input `[x_t, h_{t-1}]` of size `n_v = input_size + hidden_size`):

1. Mode construction: `m = 2 * n_v` optical modes, with an input Fock-like occupation pattern `[1,0,1,0,...]` (one photon in every other mode) supplied by Merlin.
2. Initial layer: parallel beam splitters on disjoint mode pairs (0,1), (2,3), … to generate superpositions.
3. Feature encoding: phase shifters on the first `n_v` modes with symbolic parameters `px{i}` that receive classical inputs (x and h concatenated) mapped to phases.
4. Trainable interferometer: rectangular mesh (`GenericInterferometer`) composed of repeating (BS → PS(theta_in_k) → BS → PS(theta_out_k)) cells; parameters prefixed `theta_U_*` are optimized. This implements a universal linear optics transformation (unitary on modes).
5. Measurement: Merlin measurement strategy `MODE_EXPECTATIONS` extracts per-mode expectation / intensity-like features (probability simplex). A learned linear layer optionally projects these to the gate hidden size if raw output dimension differs.

Gate outputs then pass through classical nonlinearities: `i = sigmoid(vqc_i(v_t))`, `f = sigmoid(...)`, `g = tanh(...)`, `o = sigmoid(...)`, followed by standard LSTM updates `c_t = f * c_{t-1} + i * g`, `h_t = o * tanh(c_t)`.

Head options:
- Classical linear head: `Linear(hidden_size -> output_size)` (default).
- Photonic head: an additional photonic VQC on `2 * hidden_size` modes encodes `h_t` using the same pattern, yielding output features directly.

Parameterization & size:
- Each gate VQC has its own interferometer parameters (`theta_` in/out) plus encoding phases; shared architectural hyperparameters: number of modes `2 * (input_size + hidden_size)` and photon count ≈ `input_size + hidden_size`.
- Head VQC (if enabled) adds parameters scaling with `2 * hidden_size` modes.

### Classical Baseline
Standard LSTM cell with linear layers for gates and identical hidden/state update equations for comparison and ablation.

### Shared Cell Update (All Variants)
Regardless of gate implementation (classical dense, qubit VQC expectations, photonic mode expectations), the high-level recurrence is:

$$\begin{aligned}
i_t &= \sigma(\tilde{i}_t)\\
f_t &= \sigma(\tilde{f}_t)\\
g_t &= \tanh(\tilde{g}_t)\\
o_t &= \sigma(\tilde{o}_t)\\
c_t &= f_t\odot c_{t-1} + i_t\odot g_t\\
h_t &= o_t\odot \tanh(c_t)
\end{aligned}$$

where $(\tilde{i}_t,\tilde{f}_t,\tilde{g}_t,\tilde{o}_t)$ are gate pre-activations produced by either classical affine maps, qubit expectation vectors, or photonic mode expectation projections.


### Visualization helpers (`utils/`)

This project saves raw arrays for snapshots as `simulation_data_e{epoch}.npz` together with loss CSVs. The `utils/` folder exposes a few small scripts to inspect them:

#### `aggregate_plots.py`

```bash
python -m utils.aggregate_plots \
  --runs outdir/run_2025...:QLSTM outdir/run_2025...:LSTM \
  --epochs 1,15,30,100 \
  --out outdir/aggregate_grid.png \
  --width 4 --height 3
```

- Accepts any number of `RUN_DIR:LABEL` pairs; each run must contain the requested `simulation_data_e{epoch}.npz` files.
- Produces a grid where **columns are epochs** and **rows are runs**. Every panel shows the ground truth (orange dashed) and the chosen run (blue) for that epoch, with the train/test split marked by a red vertical line.
- Adjust `--width` / `--height` to control per-subplot size. The figure is written wherever `--out` points (format inferred from the suffix).

#### `compare_losses.py`

```bash
python -m utils.compare_losses \
  --runs outdir/run_2025...:QLSTM outdir/run_2025...:Photonic \
  --metric test \
  --out outdir/test_loss_compare.png
```

- Reads each run’s `losses.csv` (written automatically at the end of training) and overlays either the train or test MSE curve (`--metric {train,test}`) across runs.
- Use this to sanity-check convergence differences between classical, gate-based quantum, and photonic variants.

#### `run_all_configs.sh`

```bash
./utils/run_all_configs.sh
```

- Executes the three canonical datasets (`sine`, `damped_shm`, `airline`) for each model family (LSTM, QLSTM, photonic QLSTM).
- The helper captures every fresh `outdir/run_*` directory from the logs, then renders:
  - `results/{dataset}_aggregate.png` via `aggregate_plots.py` (predictions per epoch/run)
  - `results/{dataset}_loss_comparison.png` via `compare_losses.py` (test MSE curves)
- Override the interpreter by exporting `PYTHON=...` beforehand if you need a specific virtual environment.


## Reproducibility Notes

- Seed control via `--seed` and per-run config snapshot.
- Global dtype via the top-level `"dtype"` config key (per-model overrides still supported). This dtype is applied to the dataset tensors and every cell implementation.
- Default dtype: `torch.double` (float64) for PennyLane compatibility. If using float32, verify device support.
- Optionally freeze exact versions (`pip freeze > outdir/run_.../requirements.txt`).

## Testing

Tests are scoped to the `QLSTM/` folder:

```bash
cd QLSTM
pytest -q
```

Notes:
- Tests expect the current working directory to be `QLSTM/`.
- If `pytest` is missing: `pip install pytest`.

## BibTeX

```bibtex
@article{chen2020quantumlstm,
  title   = {Quantum Long Short-Term Memory},
  author  = {Chen, Samuel Yen-Chi and Yoo, Shinjae and Fang, Yao-Lung L.},
  journal = {arXiv preprint arXiv:2009.01783},
  year    = {2020},
  doi     = {10.48550/arXiv.2009.01783}
}
```
