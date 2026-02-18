# QRNN — Quantum Recurrent Neural Networks (Neural Networks, 2023)

This project bootstraps a reproduction of the paper on quantum recurrent neural networks (QRNN, Neural Networks, 2023). It includes both a classical RNN baseline, the paper described Gate-Base "plain QRNN" implemented with Pennylane, and an experimental photonic QRNN prototype implemented with MerLin/Perceval.

## Reference and Attribution

- Paper: Quantum Recurrent Neural Networks for sequential learning (Neural Networks, 2023)
- Authors: Yanan Li, Zhimin Wang, Rongbing Han, Shangshang Shi, Jiaxin Li, Ruimin Shang, Haiyong Zheng, Guoqiang Zhong, Yongjian Gu
- DOI/ArXiv: https://doi.org/10.1016/j.neunet.2023.07.003 (publisher page: https://www.sciencedirect.com/science/article/abs/pii/S089360802300360X)
- Original repository (if any): not referenced here
- License and attribution notes: cite the published Neural Networks article when using results derived from this code.

## Overview

This implementation (see [QRNN.md]) provides three model families for comparison:

- a **classical RNN baseline** for sequence forecasting on meteorological time-series data
- a **gate-based pQRNN implementation** (PennyLane) following the paper's QRB/pQRNN description
- an **experimental photonic QRNN prototype** using partial measurement to propagate a hidden photonic register


Datasets and evaluation focus:

- **Weather dataset (main):** we introduce a practical meteorological forecasting dataset based on the Kaggle Szeged weather data (`budincsevity/szeged-weather`) with a preprocessing step that aggregates daily statistics. The original paper does not specify the exact meteorological dataset instance in a fully reproducible way, so this serves as a close, well-known proxy.
- **Simple generators (sanity checks):** additional smoke testing is provided on lightweight synthetic generators such as `sin` and `damped_shm` (shared under `papers/shared/time_series/`).
- **Airline Passengers (baseline time-series):** we also include configs for the classic Airline Passengers dataset (`airline-passengers.csv`) as an additional reference benchmark.

### Photonic QRNN (experimental)

In addition to the classical baseline, this paper folder now includes an **experimental photonic QRNN prototype** implemented with MerLin/Perceval.

![Photonic QRNN circuit (2-level QRB block)](photonic-qrnn.svg)

Figure: photonic circuit used by the QRB block (example shown for a 2-level QRB), including encoding, trainable phases, and partial measurement.

The implementation follows the design decisions we settled on for the QRB/QRNN block:

- **Encoding:** the input at time step `t` (denoted `x_t`) is assumed to have dimension `2*kd` and is mapped **directly** to `2*kd` phase shifters on the D register (dual-rail modes). Concretely, `x_t[i]` sets the phase of parameter `in_i` through encoding conversion (available `arccos` and `identity` options).
- **Partial measurement:** after the QRB circuit, we apply **partial measurement** on the D register (the first `2*kd` modes). The post-selected computation space is controlled by `measurement_space` (dual-rail by default).
- **Readout:** we keep the full measurement outcome space for the chosen post-selection. Let `p_t` be the probability vector over these outcomes. The model predicts
	`y_t = Linear(p_t)`, i.e. a trainable `torch.nn.Linear(len(p_t), 1)`.
- **Hidden-state propagation:** the next hidden state is derived from the conditional remaining state on the unmeasured modes (H register) and projected back into a dual-rail amplitude vector.
	- The photonic QRNN propagates **all** partial-measurement branches. This yields an exact classical mixture over possible hidden states, but the number of branches can grow exponentially.

Post-selection option for the measured D register (`measurement_space`):

- `measurement_space="dual_rail"` (default): outcomes are restricted to dual-rail bitstrings; output dimension is `2**kd`.
- `measurement_space="unbunched"`: outcomes are any 0/1 occupation patterns with `kd` photons across `2*kd` modes (more general than dual-rail); output dimension becomes `binom(2*kd, kd)`.

Current prototype limitations (intentional, to keep the first version minimal and testable):

- `shots=0` only (statevector mode; no sampling)
- Batched inputs are supported, but the current implementation evaluates samples sequentially because partial measurement returns structured branch objects (expect it to be much slower than the classical baseline).
Branch explosion note: if the measured register has `N_out` possible outcomes, then running `depth=j` layers can create up to `N_out**j` branches at a single time step (and the count also multiplies across time steps). Here `N_out` is `2**kd` in dual-rail, or `binom(2*kd, kd)` in unbunched mode.

To enable the photonic model, set the model block in your config to:

```json
{
	"model": {
		"name": "photonic_qrnn",
		"params": {
			"kd": 3,
			"kh": 1,
			"depth": 2,
			"shots": 0,
			"measurement_space": "dual_rail"
		}
	}
}
```

Important: when `model.name="photonic_qrnn"`, the dataset feature dimension must satisfy `len(dataset.feature_columns) <= 2*kd`.

- If fewer than `2*kd` features are provided, the photonic model will **pad the missing features with zeros**.
- If more than `2*kd` features are provided, the run fails with a `ValueError`.

Implementation entry points:

- QRB circuit construction: `lib/qrb_circuit.py`
- Partial-measurement readout utilities: `lib/qrb_readout.py`
- Model: `lib/photonic_qrnn.py`

### Gate-based pQRNN (PennyLane)

This folder also includes a gate-based implementation of the pQRNN described in [QRNN.md].

- **Encoding:** angle encoding via `RY` on the D register (`kd` qubits)
- **Ansatz:** hardware-efficient layers on D∪H using `RX-RZ-RX` per qubit + `IsingZZ` entanglers
- **Readout:** probability $p(1)$ on the first D qubit (converted from $\langle Z\rangle$), then a trainable linear layer
- **Hidden-state propagation:** exact reduced density matrix $\rho_H$ (no branching)

Enable it with:

```json
{
	"model": {
		"name": "gate_pqrnn",
		"params": {
			"kd": 2,
			"kh": 1,
			"depth": 1,
				"entangling": "cb",
				"entangling_wrap": true
		}
	}
}
```

Example config: `configs/sin_tiny_gate_pqrnn.json`.

## How to Run

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Command-line interface

Use the repository-level runner (`implementation.py` at the repo root) with `--paper QRNN`.

```bash
# From the repo root
python implementation.py --paper QRNN --help

# From inside this folder
python ../../implementation.py --help
```

Common options (see `cli.json` for the full schema alongside the global flags injected by the shared runner):

- `--config PATH` Load an additional JSON config (merged over defaults).
- `--epochs INT` Override `training.epochs`.
- `--batch-size INT` Override `dataset.batch_size`.
- `--sequence-length INT` History length used for forecasting.
- `--hidden-dim INT` RNN hidden dimension.
- Standard global flags: `--seed`, `--dtype`, `--device`, `--log-level`, `--outdir`.

Example runs:

```bash
# Quick smoke test (no download; uses a synthetic generator from papers/shared/time_series)
python implementation.py --paper QRNN --config configs/damped_shm_default_rnn.json --epochs 1

# Same run from inside QRNN/
python ../../implementation.py --config configs/damped_shm_default_rnn.json --epochs 1

# Train on the Szeged Kaggle dataset (downloads on first run if needed)
python implementation.py --paper QRNN --outdir runs/qrnn_baseline
```

### Quick comparison sweep (paper utilities)

To compare `rnn_baseline`, `gate_pqrnn`, and `photonic_qrnn` on `sin_tiny` and `airline_small` with a small parameter sweep (default: 5 epochs), use:

```bash
# From the repo root
python papers/QRNN/utils/compare_qrnn_models.py --epochs 5

# Optional: cap runs while debugging
python papers/QRNN/utils/compare_qrnn_models.py --dry-run --epochs 5 --max-runs 5
```

The script writes a `summary.csv` under `papers/QRNN/outdir/qrnn_compare/compare_seed{seed}_e{epochs}/` (config files are exported in `sweep_configs/`).

At the end of the sweep, it also saves two overlay plots per dataset next to `summary.csv`:
- `<dataset>__metrics_overlay.png` (test curves + val dashed)
- `<dataset>__predictions_overlay.png` (prediction traces vs reference)

### Dataset setup & preprocessing

- Datasets are stored under the shared data root: by default `data/` at the repo root. For relative `dataset.path` values, QRNN first looks under `data/QRNN/` and, if the file is not found there, falls back to `data/time_series/` for shared time-series CSVs. You can override the base location with `DATA_DIR=/abs/path` or `python implementation.py --data-root /abs/path ...`.
- The default configuration uses `dataset.path="szeged-weather.csv"` and `dataset.preprocess="szeged_weather"`. If the raw file is missing, it will be downloaded from `budincsevity/szeged-weather`, then preprocessed into `szeged-weather.preprocess.csv` the first time. Subsequent runs reuse the preprocessed file.
- The Szeged-specific preprocessor consumes `Formatted Date`, `Temperature (C)`, `Humidity`, `Wind Speed (km/h)`, and `Pressure (millibars)` and outputs a daily CSV with columns: `date`, `min_temperature`, `max_temperature`, `avg_humidity`, `avg_wind_speed`, `avg_pressure`.
- The output also includes a `month` column (derived from `Formatted Date`) which is used by default as an additional feature.
- To run without downloading any CSV, use `configs/damped_shm_default_rnn.json`, which generates a synthetic time-series via the shared time-series generators (`papers/shared/time_series/`).
- Additional dataset-specific preprocessors can be registered in `lib/preprocess.py`; set `dataset.preprocess` to the corresponding key to enable them. If a `<file>.preprocess.csv` already exists, it is reused without rebuilding.
- Use `dataset.max_rows` to cap the number of rows ingested from the CSV (applied before splitting). Normalization stats are computed on the resulting training split to avoid leakage.

### Outputs

Each run writes to `<outdir>/run_YYYYMMDD-HHMMSS/` and includes:

- `config_snapshot.json` — resolved configuration used for the run
- `metrics.json` — train/validation loss history
- `predictions.csv` — reference vs model predictions for train/val/test splits
- `metadata.json` — dataset and preprocessing metadata
- `rnn_baseline.pt` — PyTorch checkpoint for the baseline model
- `done.txt` — completion marker

Plotting scripts are shared across papers under `papers/shared/time_series/`:

- `python -m papers.shared.time_series.plot_predictions <run_dir>`
- `python -m papers.shared.time_series.plot_metrics <run_dir>`

Convenience wrappers (QRNN-local):
- `python papers/QRNN/utils/plot_predictions.py <run_dir>`
- `python papers/QRNN/utils/plot_metrics.py <run_dir>`

Training on the Airline Passengers dataset: use `configs/airline_default_rnn.json` (baseline) or `configs/airline_strong_rnn.json` (stronger). Canonical path is `data/time_series/airline-passengers.csv`.

### Shared time-series plotting

The generic scripts live in `papers/shared/time_series/` and can be used across papers.

## Configuration

Key files:

- `configs/defaults.json` — baseline hyperparameters and dataset paths
- `configs/damped_shm_default_rnn.json` — synthetic dataset (damped SHM) via shared time-series generators
- `cli.json` — CLI schema consumed by the shared runner

Precision control: include `"dtype"` (e.g., `"float32"`) at the top level or under `model` to run in a specific torch dtype.

### Parameter reference (classical vs photonic)

This section documents the configuration parameters that matter most for the two supported model families:

- **Classical baseline**: `model.name="rnn_baseline"` (PyTorch RNN/GRU/LSTM).
- **Photonic prototype**: `model.name="photonic_qrnn"` (MerLin/Perceval partial measurement).

Common top-level fields

- `description` (str): free-form description required by the shared runtime.
- `dtype` (str): floating point dtype label, e.g. `"float32"` or `"float64"`.

Dataset fields (`dataset.*`)

- `dataset.name` (str): dataset identifier. For synthetic generators, use `"sin"`, `"damped_shm"`, etc.
- `dataset.path` (str|null): CSV filename or path. If null, the generator must be set.
- `dataset.preprocess` (str|null): optional preprocessing key (e.g. `"szeged_weather"`).
- `dataset.generator` (obj|null): enables synthetic generation.
	- `dataset.generator.name` (str): generator key (e.g. `"sin"`).
	- `dataset.generator.params` (obj): generator-specific parameters.
	- `dataset.generator.feature_dim` (int, optional): number of features per time step.
		- For the photonic QRNN, this must be at most `2*kd`.
		- When fewer features are produced (e.g., a 1D signal), the photonic model pads the remaining features with zeros.
- `dataset.sequence_length` (int): number of time steps fed to the model.
- `dataset.prediction_horizon` (int): how far ahead to predict (currently used as a shift in the target).
- `dataset.max_rows` (int|null): optional cap on the number of rows used (applied before splitting).
- `dataset.train_ratio` / `dataset.val_ratio` (float): split ratios; test ratio is implicit.
- `dataset.batch_size` (int): training batch size.
	- Photonic prototype note: batching is supported but evaluated sequentially (performance can be dominated by the quantum simulation).
- `dataset.shuffle` (bool): whether to shuffle samples when building the dataloaders.
- `dataset.feature_normalization` (str): feature normalization scheme applied using training-split statistics.
	- Supported: `"standard"`, `"none"`, `"minmax_0_1"`, `"minmax_-1_1"`.
	- Default for QRNN configs/CLI in this repo: `"minmax_-1_1"` (recommended for the paper-style $\arccos$ embedding).

Training fields (`training.*`)

- `training.epochs` (int)
- `training.optimizer` (str): currently supports `"adam"`.
- `training.lr` (float)
- `training.weight_decay` (float)
- `training.clip_grad_norm` (float|null): optional gradient norm clipping.
- `training.early_stopping.enabled` (bool)
- `training.early_stopping.patience` (int)
- `training.early_stopping.min_delta` (float)

Classical RNN model fields (`model.name="rnn_baseline"`, `model.params.*`)

- `cell_type` (str): `"rnn"`, `"gru"`, or `"lstm"`.
- `hidden_dim` (int): hidden state size of the RNN.
- `layers` (int): number of stacked recurrent layers.
- `dropout` (float): inter-layer dropout (applied by PyTorch when `layers>1`).
- `input_dropout` (float): dropout on the input features before the recurrent core.
- `bidirectional` (bool): whether to use a bidirectional recurrent core.

Photonic QRNN model fields (`model.name="photonic_qrnn"`, `model.params.*`)

- `kd` (int): number of logical "data" photons (D register). Input feature dimension must be `<= 2*kd` (padded with zeros if smaller).
- `kh` (int): number of logical "hidden" photons (H register). This is the recurrent memory.
- `depth` (int): number of QRB applications *within a single time step*.
	- Note: this is a prototype/generalization knob and is not a named hyperparameter of the original paper. To stay closest to the paper, set `depth=1`.
- `shots` (int): number of samples.
	- Prototype limitation: must be `0` (statevector mode).
- `measurement_space` (str): post-selection space for the measured D register.
	- `"dual_rail"`: outcome space size `2**kd`.
	- `"unbunched"`: outcome space size `binom(2*kd, kd)`.
- `input_encoding` (str): optional input-to-angle encoding before setting the `2*kd` phase shifters.
	- `"identity"` (default): uses features directly as phase angles.
	- `"arccos"`: uses $\theta=\arccos(x)$ (paper-style). Prefer `dataset.feature_normalization="minmax_-1_1"`.

Gate-based pQRNN model fields (`model.name="gate_pqrnn"`, `model.params.*`)

- `kd` (int): number of qubits in the data register D.
	- Input feature dimension must satisfy `feature_dim <= kd`.
	- If fewer than `kd` features are provided, the model pads missing features with zeros.
- `kh` (int): number of qubits in the hidden register H.
	- `kh=0` is supported (degenerate “no memory” case).
- `depth` (int): number of ansatz layers applied at each time step.
	- Each layer applies `RX-RZ-RX` on every qubit in D∪H, followed by `IsingZZ` entanglers.
- `entangling` (str): entangling connectivity for `IsingZZ` gates.
	- `"cb"` (default): Circuit-Block (CB) pattern from the paper: nearest-neighbor chain + optional wrap-around `(0, N-1)` over the full D∪H register.
	- `"nn"`: nearest-neighbor chain over the full D∪H register (no wrap-around).
	- `"all"`: all-to-all connectivity.
- `entangling_wrap` (bool): only used when `entangling="cb"`.
	- `true` (default): include the wrap-around edge `(0, N-1)` when `N>2`.
	- `false`: disable wrap-around.
- `input_encoding` (str): feature-to-angle encoding applied before `RY` on the D register.
	- `"identity"` (default): uses features directly as rotation angles.
	- `"arccos"`: uses $\theta=\arccos(x)$ (paper-style). Prefer `dataset.feature_normalization="minmax_-1_1"`.

Implementation details (gate pQRNN)

- **Backend:** PennyLane `default.mixed` in analytic mode.
- **Encoding:** `RY(theta_i)` on each D qubit (after zero-padding up to `kd`), where `theta_i` depends on `model.params.input_encoding`.
- **Readout:** probability $p(1)$ on the first D qubit (computed from $\langle Z\rangle$ via $p(1)=(1-\langle Z\rangle)/2$), then a trainable linear layer.
- **Hidden propagation:** the hidden register is propagated exactly as a reduced density matrix $\rho_H$.

Important convention note (gate pQRNN)

- The paper uses $R_{zz}(\theta)=e^{i\theta Z\otimes Z}$.
- PennyLane uses `qml.IsingZZ(phi)=exp(-i*phi/2*Z\otimes Z)`.
- The implementation therefore applies `qml.IsingZZ(-2*theta)` internally (the trainable parameter stored is $\theta$ in the paper’s convention).

Limitations / gotchas

- `shots` is intentionally **not supported** for `gate_pqrnn` (analytic mode only).
- The internal cell currently runs with `batch_size==1` at the quantum step; the regressor loops over the batch.
- Use `dtype="float64"` for stability and to match the PennyLane mixed-state backend.

## Results and Next Steps

- Planned extensions: compare different models, optimize hyper-parameters and run ablations versus the classical RNN.

## Testing

Run tests from inside `papers/QRNN/`:

```bash
cd papers/QRNN
pytest -q
```

Tests cover the CLI, config loading, and a smoke run of the training loop on the synthetic dataset.
