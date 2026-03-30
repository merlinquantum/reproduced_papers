# Quantum Gaussian Kernel

Reproduction of Algorithm 2 (“Linear quantum photonic circuits as Gaussian kernel samplers”) from the parent paper (see the root README for citation details). Implementation recreated from the manuscript.

## Overview
- Trains photonic quantum circuits to approximate Gaussian kernels of varying standard deviations by minimizing MSE between learned kernels and analytical targets. This reproduces Figure 5 from the reference paper.
- Exports the best-performing kernel models (for photon counts `{2,4,6,8,10}`) and reuses them to build precomputed kernels for SVM classification on circular, moon, and blob datasets.
- Compares quantum kernel SVMs against classical RBF-SVM baselines.
- CLI task system (`implementation.py --task {sampler,classify}`) orchestrates both phases with JSON-configured hyperparameters.

The experiments demonstrate that quantum circuits can learn to approximate Gaussian kernels, with performance varying based on:
- **Circuit architecture**: Different interferometer designs show varying approximation capabilities
- **Photon number**: Circuits with more photons have a tendency for better fits on Gaussians with smaller standard deviation
- **Training dynamics**: Quantum kernel learning presents unique optimization challenges compared to classical methods
- **Approximation accuracy**: Our Gaussian kernel fits are less accurate than those presented in the reference paper

## How to Run

### CLI workflow
Run from the repo root with `--paper fock_state_expressivity/q_gaussian_kernel`.

Sampler task (creates a new run folder and writes sampler outputs + checkpoints):
```bash
python implementation.py --paper fock_state_expressivity/q_gaussian_kernel --task sampler
```

Classify task with prior sampler outputs (creates a new run folder and writes classify outputs there):
```bash
python implementation.py --paper fock_state_expressivity/q_gaussian_kernel --task classify \
  --previous-run results/run_YYYYMMDD-HHMMSS
```

Classify task without `--previous-run` (runs the sampler first using `configs/defaults.json`,
then classifies using the newly created run’s `model/manifest.json`):
```bash
python implementation.py --paper fock_state_expressivity/q_gaussian_kernel --task classify
```

Common flags:
- `--config PATH` select another JSON config (default `configs/defaults.json`).
- `--seed INT`, `--outdir DIR`, `--device STR` override reproducibility and device (default outdir `results/`).
- `--num-runs INT` override the sampler’s `training.num_runs` value (number of restarts per `(σ, n)` pair).
- `--previous-run PATH` reuse model outputs from a prior sampler run for classification.
- `--manifest PATH` manually point to a checkpoint manifest for classification runs.
  If `--previous-run` is not set, `--manifest` is ignored.

Each execution creates `results/run_YYYYMMDD-HHMMSS/` (or `<outdir>/...` if overridden) containing:
```
config_snapshot.json      # Resolved config (after CLI overrides)
model/                    # Sampler checkpoints + manifest
sampler/                  # Sampler outputs
classify/                 # Classification outputs
```
Sampler checkpoints are stored under `results/<run>/model/`.
Sampler and classification outputs:
```
sampler/
  summary.txt
  metrics.json
  visualization_data/learned_functions.json
classify/
  summary.txt
  quantum_metrics.json
  classical_metrics.json
  visualization_data/classification_datasets.json
```

### Visualizations
Each visualization script can either reuse a prior run (`--previous-run`) or launch the
task it depends on if `--previous-run` is omitted (except `visu_dataset_examples.py`,
which generates datasets from `configs/defaults.json` without running training):
```bash
python papers/fock_state_expressivity/q_gaussian_kernel/utils/visu_learned_functions.py \
  --previous-run results/run_YYYYMMDD-HHMMSS

python papers/fock_state_expressivity/q_gaussian_kernel/utils/visu_dataset_examples.py \
  --previous-run results/run_YYYYMMDD-HHMMSS

python papers/fock_state_expressivity/q_gaussian_kernel/utils/visu_accuracy_bars.py \
  --previous-run results/run_YYYYMMDD-HHMMSS
```
Notes:
- `visu_learned_functions.py` requires sampler outputs (`sampler/visualization_data/`).
- `visu_dataset_examples.py` uses `classify/visualization_data/` when `--previous-run`
  is provided; otherwise it generates datasets from defaults.
- `visu_accuracy_bars.py` requires classify outputs (`classify/visualization_data/` and
  `classify/*_metrics.json`).
If required data is missing, the script warns and exits.
Figures are saved under `results/run_YYYYMMDD-HHMMSS/<task>/figures/` (or
`results/figures/` for `visu_dataset_examples.py` without `--previous-run`):
```
sampler/figures/
  learned_vs_target.png
classify/figures/
  datasets/classification_datasets.png
  svm_accuracy.png
figures/
  classification_datasets.png
```

## Configuration
Configs follow the template structure (`configs/`):
- `experiment.task`: `"sampler"` or `"classify"` (can be overridden via CLI).
- `model`: circuit architecture (`general`, `mzi`, etc.), scaling layer type, trainable interferometer flag, bunching toggle.
- `training`: optimizer settings, number of restarts, epochs, batch size, shuffle toggle.
- `sampler`: photon counts, Gaussian grid definition (span/step/σ list).
- `classification`: dataset generation parameters (noise, scaling) and manifest path reused for evaluation.

Modify/duplicate `configs/defaults.json` (sampler) or `configs/classify.json` (classification) to explore other circuits, photon sets, or datasets.

## Results and Analysis
- **Sampler task**: `sampler/summary.txt` reports mean/min/max training MSE for each `(σ, n)` pair; use `visu_learned_functions.py` to render learned kernels vs analytical Gaussians.
- **Classification task**: `visu_accuracy_bars.py` compares quantum precomputed kernels against RBF-SVMs. For small σ, high-photon circuits close the gap with classical baselines but optimization becomes harder, mirroring the paper’s finding.

The visualization utilities generate:
- **Kernel approximation plots**: Visual comparison of learned vs target kernels
- **SVM performance**: Classification accuracy using quantum vs classical kernels

The paper's results are the follow:

![q_gauss_kernel_to_reproduce](./results/Gaussian_kernel-kernel_methods.png)

Ours:

![q_gauss_kernel_reproduced](./results/learned_gauss_kernels_best_hps.png)

Key findings include:
- Quantum circuits can successfully approximate Gaussian kernel functions
- Circuits with more photons have a tendency for better fits on Gaussians with smaller standard deviation
- Our implementation could not fit the Gaussians as accurately as seen in the reference paper, requiring further investigation

## Hyperparameters

### Training Parameters
- **`num_runs`** (5): Number of experimental repetitions for statistical reliability
- **`num_epochs`** (200): Number of training iterations
- **`batch_size`** (32): Number of samples processed together in each training step
- **`lr`** (0.02, range: 0.002-0.2): Learning rate - controls optimization step size

### Optimizer Settings
- **`betas`** ([0.7, 0.9], options: [0.7,0.9], [0.9,0.999], [0.95,0.9999]): Adam optimizer momentum parameters
- **`weight_decay`** (0.0, options: 0.0, 0.02): L2 regularization strength
- **`optimizer`** ("adam", "adagrad"): Optimization algorithm choice
- **`shuffle_train`** (True): Whether to randomize training data order

### Quantum Circuit Configuration
- **`num_photons`** ([2,4,6,8,10]): Different photon numbers to test circuit expressivity
- **`train_circuit`** (True): Whether to optimize quantum circuit parameters during training
- **`scale_type`** ("learned"): Parameter scaling method
  - "learned": Parameters optimized during training
  - "1", "pi", "2pi", "/pi", "/2pi", "0.1": Fixed scaling factors
- **`circuit`** ("general"): Quantum circuit architecture
  - "mzi": Mach-Zehnder interferometer
  - "general": General linear optical circuit
  - "spiral": Spiral circuit architecture
  - "general_all_angles": General circuit with all angle parameters
  - "ps_based": Phase shifter-based circuit
  - "bs_based": Beam splitter-based circuit
- **`no_bunching`** (False): Whether to prevent multiple photons in the same mode

## Reproducibility Notes
- Seeds propagate to `random`, `numpy`, and `torch`; dataset caches under `data/cache/` ensure deterministic reuse unless `force_regenerate` is set.
- Checkpoints are saved inside `results/<run>/model/`.
- GPU execution depends on Perceval/MerLin CUDA builds; defaults run on CPU.

## Testing
```bash
pytest -q papers/fock_state_expressivity/q_gaussian_kernel/tests
```
Current tests cover Gaussian grid and dataset preparation. Extend with regression tests (e.g., MSE/accuracy thresholds) as needed.

## Additional Material
- `notebook.ipynb` preserves the exploratory workflow prior to the scripted refactor.
- `results/` houses the static figures reproduced from the paper for documentation.
