# Quantum Random Kitchen Sinks

Template-aligned reproduction of Algorithm 3 (“Quantum-enhanced random kitchen sinks”) from the parent paper (see the root README for citation details). Reproduction built from the manuscript.

## Overview
- Benchmarks classical random kitchen sinks (RKS) and their photonic quantum counterpart on the moons dataset.
- Uses Perceval+MerLin to build photonic random feature maps and compares against cosine random Fourier features for several `R` (feature dimensionality) and `γ` (kernel bandwidth) values.
- Produces dataset visualizations, kernel heatmaps, accuracy heatmaps, and decision boundaries for both classical and quantum methods via post-run utilities. This reproduces the Figure 6 from the reference paper.
- Everything is orchestrated through `implementation.py` with JSON-configured sweeps.

The experiments demonstrate the behavior of quantum-enhanced random kitchen sinks:
- **Parameter-dependent performance**: The quantum method tends to outperform the classical method when using low gamma values (high standard deviation for Gaussian kernels) while the opposite happens with high gamma values
- **Photon number effects**: Using more photons leads to more extreme decision boundaries with more variance
- **Architecture sensitivity**: Different quantum circuits show varying performance characteristics
- **Parameter optimization**: Proper hyperparameter tuning is crucial for optimal performance

## How to Run

### CLI entrypoint
Run from the repo root with `--paper fock_state_expressivity/q_random_kitchen_sinks`:
```bash
python implementation.py --help
```

Key options:
- `--config PATH` JSON config (default `configs/defaults.json`).
- `--seed INT` override RNG seed (controls dataset split + random features).
- `--outdir DIR` choose run directory root (default `results/`).

Example sweep (default r∈{1,10,100}, γ∈{1…5}):
```bash
python implementation.py --paper fock_state_expressivity/q_random_kitchen_sinks
```

Each run produces `results/run_YYYYMMDD-HHMMSS/` (or `<outdir>/...` if overridden) with:
```
summary.txt                 # Accuracy table per (method, R, γ)
metrics.json                # Scalar accuracy entries for every repeat
config_snapshot.json        # Resolved config after CLI overrides
visualization_data/
  dataset.json
  combined_decisions_quantum.json
  combined_decisions_classical.json
  kernel_quantum.json       # only when the sweep contains a single (R, γ)
  kernel_classical.json     # only when the sweep contains a single (R, γ)
```

### Visualizations
Generate figures from a previous run or re-run when `--previous-run` is omitted:
```bash
python papers/fock_state_expressivity/q_random_kitchen_sinks/utils/visu_dataset.py \
  --previous-run results/run_YYYYMMDD-HHMMSS

python papers/fock_state_expressivity/q_random_kitchen_sinks/utils/visu_accuracies.py \
  --previous-run results/run_YYYYMMDD-HHMMSS

python papers/fock_state_expressivity/q_random_kitchen_sinks/utils/visu_decisions.py \
  --previous-run results/run_YYYYMMDD-HHMMSS

python papers/fock_state_expressivity/q_random_kitchen_sinks/utils/visu_kernel_heatmap.py \
  --previous-run results/run_YYYYMMDD-HHMMSS
```
Figures are saved under `results/run_YYYYMMDD-HHMMSS/figures/`:
```
figures/
  moons_dataset.png
  accuracy_bar.png           # single (R, γ) case
  accuracy_quantum.png       # sweep case
  accuracy_classical.png     # sweep case
  decision_boundary_quantum.png      # single (R, γ) case
  decision_boundary_classical.png    # single (R, γ) case
  combined_decisions_quantum.png     # sweep case
  combined_decisions_classical.png   # sweep case
  kernel_quantum.png        # only when the sweep contains a single (R, γ)
  kernel_classical.png      # only when the sweep contains a single (R, γ)
```

## Configuration
Configs live under `configs/` (the default config trains the hybrid MerLin layer on the cosine target before fitting the SVM):
- `defaults.json` — hybrid training ON, photonic layer fits gaussian kernel function (`hybrid_model_data="Default"`).
- `no_hybrid_training.json` — disables the hybrid optimization phase to reproduce the “frozen layer” ablation.
- `hybrid_generated.json` — hybrid training ON using synthetic inputs (`hybrid_model_data="Generated"`) for a denser fitting set.
- `single_combo_heatmap.json` — fixes the sweep to a single `(R, γ)` pair so the quantum/classical kernel heatmaps are emitted.

Common config blocks:
- `data`: moons dataset generation, scaling, split.
- `model`: photonic circuit choices (MZI/general), photon count.
- `training`: optimizer, epochs, hybrid-model toggles (`train_hybrid_model`, `pre_encoding_scaling`, `z_q_matrix_scaling`, `hybrid_model_data`).
- `classifier`: downstream SVM hyperparameters (`C`).
- `sweep`: lists of `r_values`, `gamma_values`, and number of repeats per combination.

Adjust or duplicate `configs/defaults.json` to experiment with custom sweeps or training regimes.

## Results and Analysis
Visualization utilities generate comprehensive analysis outputs:
- **Kernel approximation plots**: Visual comparison of quantum vs classical kernel approximations
- **Performance metrics**: Accuracy comparison across different configurations
- **Parameter sweeps**: Analysis of hyperparameter effects on performance

The paper's obtained decision boundaries are the following:

![q_gauss_kernel_to_reproduce](./results/Classification-RKS.png)

Ours:

![q_gauss_kernel_reproduced](./results/q_rand_kitchen_sinks_overall_example.png)

Key findings include:
- Quantum circuits can approximate Gaussian kernels with competitive or superior accuracy
- The quantum method tends to outperform the classical method with low gamma values while classical methods excel with high gamma values
- Different photon numbers and circuit architectures provide trade-offs between expressivity and optimization difficulty

## Hyperparameters

### Data Parameters
- **`n_samples`** (200): Number of training samples
- **`noise`** (0.2): Amount of noise added to data
- **`random_state`** (42): Random seed for reproducibility
- **`scaling`** ("MinMax", "Standard"): Data normalization method
- **`test_prop`** (0.4): Fraction of data used for testing

### Training Parameters
- **`batch_size`** (30): Number of samples processed together in each training step
- **`optimizer`** ("adam", "sgd", "adagrad"): Optimization algorithm choice
- **`learning_rate`** (0.01): Learning rate - controls optimization step size
- **`betas`** ((0.99, 0.9999)): Adam optimizer momentum parameters
- **`weight_decay`** (0.0002): L2 regularization strength
- **`num_epochs`** (200): Number of training iterations

### Quantum Circuit Parameters
- **`num_photon`** (10): Number of photons in the quantum system
- **`no_bunching`** (False): Whether to prevent multiple photons in the same mode
- **`circuit`** ("general", "mzi"): Quantum circuit architecture
  - "mzi": Mach-Zehnder interferometer
  - "general": General linear optical circuit

### Algorithm-Specific Parameters
- **`C`** (5): SVM regularization parameter (for classification phase)
- **`r`** (1, tested: [1,10,100]): Dimensionality of random Fourier features
- **`gamma`** (1, tested: [1-10]): Kernel bandwidth parameter (σ = 1/γ)
- **`train_hybrid_model`** (True): **Controls whether to train the hybrid quantum-classical model on the function fitting task.** This is separate from SVM classification and determines if quantum circuit parameters get optimized to approximate target functions.
- **`pre_encoding_scaling`** (1.0): Input scaling factor applied before quantum encoding
- **`z_q_matrix_scaling`** (10): Output matrix scaling factor
- **`hybrid_model_data`** ("Generated"): **Data source for training the hybrid model on fitting tasks** (separate from SVM classification)
  - "Generated": Uses synthetically generated data for fitting
  - "Default": Uses data from the moon dataset for fitting

## Reproducibility Notes
- Seeds affect dataset generation and the random Fourier parameters `(w,b)` to ensure fair classical-vs-quantum comparisons.
- All artifacts (figures/kernels) are deterministic per `(seed, config)` pair.
- Hybrid quantum-layer training is optional; disable it in `training.train_hybrid_model` to reproduce the “inference-only” regime from the manuscript.

## Testing
```bash
pytest -q papers/fock_state_expressivity/q_random_kitchen_sinks/tests
```
Current tests cover dataset preparation; extend with regression tests (e.g., accuracy thresholds) as needed.

## Additional Material
- `notebook.ipynb` keeps an interactive exploratory workflow.
- `results/` stores the curated figures reproduced from the paper for documentation.
