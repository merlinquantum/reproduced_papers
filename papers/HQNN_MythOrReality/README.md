# Reproduction of the work: "Computational Advantage in Hybrid Quantum Neural Networks: Myth or Reality?", by Kashif et al.

## Reference and Attribution

- Paper: Computational Advantage in Hybrid Quantum Neural Networks: Myth or Reality? (2024 preprint)
- Authors: Kashif et al.
- ArXiv: https://arxiv.org/abs/2412.04991
- Original repository: not provided by the authors (built from description)
- License and attribution: cite the paper above when reusing this reproduction

## Overview

The paper investigates whether hybrid quantum neural networks (HQNNs) can surpass purely classical models when both are trained on increasingly complex spiral datasets. Our reproduction rebuilds the synthetic 3-class spiral data generator, the hybrid photonic HQNN stack, and the classical multilayer perceptron baseline so we can compare accuracy and parameter efficiency as the number of features ranges from 10 to 110.

- **Reproduced scope:** spiral dataset (12 feature settings), HQNN configurations with variable qubits/modes and depths, 155+ classical MLPs, accuracy and parameter-count metrics logged per setting.
- **Deviations & assumptions:** extended the classical search space to larger hidden layers (up to 256 neurons) because the original bounds underperformed on >70 features; photonic HQNN search additionally swept 2–24 modes with optional bunching to mirror the photonic simulator available locally.
- **Hardware / software environment:** runs executed in a Python virtual environment (`hqnn-venv`) using the pinned dependencies in `requirements.txt` (PyTorch 2.8.0, scikit-learn 1.7.2, perceval-quandela 1.0.0, etc.) on a CPU-only workstation; no GPU acceleration is required for the reported sweeps.

## Project Layout

- Project location: `papers/HQNN_MythOrReality/` (dispatched by the repo-level `implementation.py`)
- Repository root `implementation.py`: shared CLI that dispatches to every paper folder via `--paper`
- `configs/`: JSON configuration files (`defaults.json`, `cli.json`)
- `models/`: HQNN and classical baseline model definitions
- `utils/`: shared helpers for data loading, training, and result persistence
- `tests/`: lightweight sanity tests for the CLI and architecture enumeration
- `scripts/`: auxiliary runners such as the classical baseline sweep
- `requirements.txt`: pinned dependencies matching the `hqnn-venv` environment

## How to Run

### Install dependencies

- Create an isolated environment (e.g. `python -m venv hqnn-venv && source hqnn-venv/bin/activate`)
- Install the pinned requirements: `pip install -r requirements.txt`

### Command-line interface

- HQNN sweep (from repo root): `python implementation.py --paper HQNN_MythOrReality --config configs/spiral_scan.json` creates a timestamped run folder under `results/` with logs, metrics, and config snapshots. Override config values with flags such as `--feature-grid 10,20`, `--accuracy-threshold 92`, `--lr 0.02`, or `--figure` (parameter-count plot generation). Combine with global switches like `--seed` or `--device mps` as needed.
- HQNN sweep (from inside this project): `python ../../implementation.py --config configs/spiral_scan.json --feature-grid 10,20 --lr 0.02` (no `--paper` flag needed when the current working directory is `papers/HQNN_MythOrReality`).
- Classical baseline: `python3 scripts/run_classical_baseline.py --features 10,20,30` sweeps MLP configurations and appends results to `results/classical_baseline.json`. Adjust `--lr`, `--batch-size`, or `--threshold` as needed.
- Every run persists its merged configuration to `<outdir>/run_<timestamp>/config_snapshot.json` for traceability.

### Output directory and generated files

- HQNN runs write `results/run_<timestamp>/` directories containing resolved configs, log files, optional figures, and per-feature accuracy summaries.
- The classical sweep accumulates results in `results/classical_baseline.json` so repeated commands append without overwriting earlier experiments.

## Configuration

- JSON files under `configs/` encode feature grids, training regimes, and HQNN search bounds; `configs/defaults.json` mirrors the paper’s setup, `cli.json` defines exposed overrides, and the shared runtime automatically calls `lib/runner.train_and_evaluate`.
- CLI overrides (`--lr`, `--feature-grid`, `--accuracy-threshold`, etc.) merge with the JSON defaults via the runtime layer and are recorded with each run.

## Dataset

The original paper states: _"We generate a spiral dataset with 1500 data points distributed across 3 classes... We utilize feature sizes ranging from 10 to 110 in increments of 10"._

Here is the dataset that we generated:
![First 2 features of the Spiral dataset](./assets/blobs_dataset.png)
And we can observe that the complexity increases with the number of features as the performance of fixed classical neural networks (NN) and Hybrid Quantum NN (HQNN) decreases:
![Performance with respect to feature size](./assets/acc_param_wrt_features.png)

## Results and Analysis

### Classical neural networks
The original paper mentions: _"We restrict the classical models to have a maximum of n = 3 layers, with the number of neurons in each layer chosen from the set m = {2, 4, 6, 8, 10} resulting in a search space of total 155 model combinations for classical models for each complexity level."_

Here, we implement the same models but the largest network cannot classify the spiral dataset with ≥90% accuracy when the feature count exceeds 70. We therefore increase the search space to allow more neurons per hidden layer (up to 256). The resulting parameter sweep is illustrated below:
![Search space for the classical NN](./assets/cl_NN_params.png)

`python3 scripts/run_classical_baseline.py` executes the sweep and logs results to `results/classical_baseline.json`.

### Hybrid Quantum Neural Networks
The original paper describes: _"During the hybrid model search only the quantum layers are varied. We use [3, 4, 5] qubits quantum layers, and for each qubit size, quantum layers of depth [1, 2, 3, ..., 10] are tested, yielding 30 model combinations per feature size."_

For the photonic implementation, we search 2–24 modes with 1 photon up to modes/2 photons, testing both Fock and unbunched configurations. The search space looks as follows:
![Search space for the HQNN](./assets/HQNN_params.png)

`python implementation.py --paper HQNN_MythOrReality` launches the HQNN experiments and stores outputs under `results/run_<timestamp>/`.

### Summary metrics
The parameter-count comparison is captured below:
![Results parameters](./assets/Results.png)

## Extensions and Next Steps

1. Explore additional dataset complexities (noise schedules, new class counts) to probe HQNN robustness.
2. Replace the photonic simulator with different quantum backends or hardware access.
3. Add Bayesian search or pruning heuristics to reduce the number of HQNN/MLP experiments while maintaining coverage.

## Reproducibility Notes

- Random seeds can be fixed via CLI flags (see `implementation.py --help`); each run stores the resolved configuration alongside outputs.
- Dataset generation relies on deterministic numpy/torch calls; enabling `torch.backends.cudnn.deterministic = True` keeps GPU executions reproducible if GPUs are later used.
- Export dependency versions via `pip freeze > results/requirements.txt` when publishing artifacts.

## Testing

- Run `cd papers/HQNN_MythOrReality && pytest tests -q` to execute smoke tests that validate CLI wiring and architecture enumeration.
- Ensure dependencies from `requirements.txt` are installed; install `pytest` separately if omitted from your environment.
