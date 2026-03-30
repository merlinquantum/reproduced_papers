# VQC Fourier Series — Expressivity Reproduction

Template-aligned reproduction of the Fourier-series fitting experiment from the parent paper (see the root README for the citation). Implementation reconstructed from the manuscript.

## Overview
- Recreates the Fourier-series regression task used to validate that adding photons to a photonic VQC increases expressivity.
- Implements the circuit with Perceval + MerLin so that it can be called as a PyTorch module.
- Sweeps three initial Fock states `[[1,0,0],[1,1,0],[1,1,1]]` with multiple random restarts to gather statistics.
- Produces plots (loss envelopes + learned functions) and summary tables per run directory.

## How to Run

### Command-line interface
Main entry point: `implementation.py` (run from repo root)
```bash
python implementation.py --help
```

Key options:
- `--config PATH` JSON config (defaults to `configs/defaults.json`).
- `--seed INT` Override RNG seed.
- `--outdir DIR` Base folder for timestamped runs (default: `results/`).
- `--device STR` Torch device string (default `cpu` — GPU support depends on Perceval/MerLin build).
- `--show-plots` Display matplotlib figures when running visualization utilities.

Example runs:
```bash
# Default reproduction (3 runs per photon configuration)
python implementation.py --paper fock_state_expressivity/VQC_fourier_series

# Custom seed and output directory
python implementation.py --paper fock_state_expressivity/VQC_fourier_series \
  --seed 2024 --outdir results/vqc_fourier
```

Each run creates `results/run_YYYYMMDD-HHMMSS/` (or `<outdir>/...` if overridden) containing:
```
summary.txt                 # Textual statistics per photon config
metrics.json                # JSON dump of all recorded losses/MSEs
config_snapshot.json        # Resolved config (after CLI overrides)
learned_function/predictions.json # Best-model predictions for visualization
```

### Visualizations
Plots are generated via utilities that can reuse a previous run or re-run the experiment when the flag `--previous-run` is not used:
```bash
python papers/fock_state_expressivity/VQC_fourier_series/utils/visu_training_curves.py \
  --previous-run results/run_YYYYMMDD-HHMMSS

python papers/fock_state_expressivity/VQC_fourier_series/utils/visu_learned_functions.py \
  --previous-run results/run_YYYYMMDD-HHMMSS
```
Figures are saved under `<run_dir>/figures/`.

## Configuration
Place JSON configs inside `configs/`.

- `defaults.json` reproduces the paper setup: degree-3 Fourier series coefficients, 120 epochs, batch size 32, Adam lr 0.02, and the three photon settings.
- Override individual fields via CLI or by creating new config files (e.g., to change the domain sampling, optimizer, or list of photon states).

Relevant keys:
- `data`: domain bounds, sampling step, Fourier coefficients (positive orders, negatives inferred via conjugation).
- `model`: number of modes, scale layer type, Perceval/MerLin layer options.
- `training`: optimizer hyperparameters, number of restarts.
- `plotting`: color palette per photon configuration.

## Results and Analysis
- Increasing the photon number increases the expressivity of the photonic circuit which monotonically lowers the training MSE, matching Fig. 3 of the paper:

Paper:

![VQC_fourier_series_to_reproduce](./results/Fitting_example_and_DOF.png)

Ours:

![VQC_fourier_series_reproduced](./results/expressive_power_of_the_VQC.png)

- The `summary.txt` file reports average/min/max final MSE over the independent restarts for each photon configuration.

## Reproducibility Notes
- RNG seed controls `random`, `numpy`, and `torch` generators; set via config or `--seed`.
- Perceval/MerLin layers currently operate on CPU; GPU acceleration requires their CUDA builds.
- Determinism beyond same-seed reproducibility is not guaranteed because MerLin layers rely on differentiable photonic simulations.

## Testing
From this folder:
```bash
pytest -q
```
Current tests cover Fourier data generation sanity checks; extend with circuit-level or regression tests as needed.

## Additional Material
- `notebook.ipynb` retains the exploratory notebook used before the scripted refactor (kept for visualization/experimentation).
- `results/` stores static figures reproduced for comparison with the paper.
