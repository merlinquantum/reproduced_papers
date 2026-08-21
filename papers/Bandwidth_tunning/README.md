# Bandwidth Tunning

Reproduction code for the paper _On the similarity of bandwidth-tuned quantum kernels and classical kernels_ by Roberto Florez-Ablan, Marco Roth, and Jan Schnabel.

This project runs bandwidth sweeps for several datasets, builds quantum and classical kernels, and writes plots and per-seed metric tables for each experiment.

## Project Layout

- `implementation.py`: local CLI entry point.
- `configs/`: experiment definitions used to reproduce the paper figures.
- `lib/imports.py`: dataset loading.
- `lib/runner.py`: experiment orchestration, PCA subsetting, training, and artifact generation.
- `lib/ploting.py`: plot generation.
- `outdir/`: timestamped run folders created by the CLI.
- `results/`: reserved for curated outputs.

## Environment Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Main runtime dependencies are:

- `scikit-learn`
- `matplotlib`
- `torch`
- `torchvision`
- `merlinquantum`
- `perceval-quandela`

## Running Experiments

Run the default configuration:

```powershell
.\env\Scripts\python.exe implementation.py
```

Run a specific configuration:

```powershell
.\env\Scripts\python.exe implementation.py --config configs/fig3.2-hidden_manifold.json
```

Override the seed from the command line:

```powershell
.\env\Scripts\python.exe implementation.py --config configs/defaults.json --seed 123
```

Each run creates a new folder under `outdir/run_YYYYMMDD-HHMMSS/`.

## Output Artifacts

For each run, the code writes:

- a PNG summary figure named after `graph_name`
- per-experiment CSV files under `raw data/`

The per-seed CSV files contain the following columns:

- `x`
- `y_g`
- `y_FQK`
- `y_RBF`
- `y_F`
- `y_eta_max_Q`
- `y_eta_max_C`
- `y_ROC_AUC`

## Datasets

The current dataset loader supports:

- `fashion_mnist`
- `kmnist28`
- `hidden_manifold`
- `plasticc`

Notes:

- `fashion_mnist` and `kmnist28` are filtered to classes `2` and `8`.
- `hidden_manifold` is a synthetic binary dataset.
- `plasticc` currently resolves to the OpenML fallback with `data_id=40900`, which is a binary anomaly dataset (`Normal` vs `Anomaly`), not the full multiclass PLAsTiCC challenge dataset.
- PLAsTiCC subset sampling in `lib/runner.py` is stratified so the sampled train and test subsets preserve both classes.

## Configuration

Configurations are JSON files in `configs/`. The main fields are:

- `seed`: base random seed
- `outdir`: output directory for timestamped runs
- `dataset.name`: dataset identifier
- `graph_name`: figure title and output PNG name
- `graphs.min`, `graphs.max`, `graphs.number_of_points`: bandwidth sweep definition
- `figs`: list of plots to generate
- `experiments`: experiment list with sample sizes, PCA dimension, projection mode, and seed count

Example experiment entry:

```json
{
  "projected": true,
  "train_sample": 320,
  "test_sample": 80,
  "description": "8 plasticc 320-80",
  "dimension": 8,
  "nb_seeds": 5
}
```

## Tests

This repository contains `test_cli.py` and `test_smoke.py`.

In the current environment, `pytest` is not installed by `requirements.txt`, so the tests will not run until it is added manually:

```powershell
pip install pytest
.\env\Scripts\python.exe -m pytest test_cli.py test_smoke.py -q
```

## Known Limitations

- The PLAsTiCC loader uses the OpenML anomaly dataset fallback rather than a multiclass PLAsTiCC source.
- Dataset downloads can fail on some Windows SSL setups. The KMNIST loader already includes a fallback path; OpenML-backed datasets still depend on the local SSL configuration.
- The helper script `run.sh` reflects a parent-project workflow and may require adjustment if used directly from this folder.