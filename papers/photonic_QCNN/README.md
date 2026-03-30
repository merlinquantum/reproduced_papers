# Photonic QCNN with Adaptive State Injection

## Reference and Attribution

- Paper: Photonic Quantum Convolutional Neural Networks with Adaptive State Injection (arXiv, 2025)
- Authors: Léo Monbroussou, Beatrice Polacchi, Verena Yacoub, Eugenio Caruccio, Giovanni Rodari, Francesco Hoch, Gonzalo Carvacho, Nicolò Spagnolo, Taira Giordani, Mattia Bossi, Abhiram Rajan, Niki Di Giano, Riccardo Albiero, Francesco Ceccarelli, Roberto Osellame, Elham Kashefi, Fabio Sciarrino
- DOI/ArXiv: https://arxiv.org/abs/2504.20989
- Original repository: https://github.com/ptitbroussou/Photonic_Subspace_QML_Toolkit
- License and attribution notes: The original implementation is used verbatim (with light wrappers) under the authors' repository license for comparison runs. Please cite both the paper and the source repository when using this work.

## Overview

This reproduction implements the proposed photonic QCNN with adaptive state injection using two complementary stacks:

- **MerLin + Perceval re-implementation** (`lib/src`, training loops in `lib/training/`) for flexible experimentation. The frameworks used for our reproduction were [Perceval](https://perceval.quandela.net) and [MerLin](https://merlinquantum.ai).

Perceval is a Python API that provides all the tools for creating and manipulating circuits from linear optical components. That makes Perceval an ideal companion for developing photonic circuits, running simulations and even accessing quantum hardwares.

MerLin is a framework which allows to define derivable quantum layers to use and optimize in the exact same manner as any classical layer with PyTorch. This integration of
quantum components into machine learning is very intuitive and it was optimized for GPU simulations on the cuda level too.
- **Paper code integration** (`lib/run_*_paper.py`) to reproduce the authors' exact setup.

Scope and components:

- Datasets: 4x4 Bars and Stripes (BAS), 4x4 Custom BAS (noise variants), and 8x8 MNIST 0 vs 1.
- Architecture:
1. Data loading on the circuit
2. Convolutions
3. Pooling
4. Dense network
5. Measurement
- Metrics: train/test accuracy, detailed training curves, readout analysis figures (Figure 4 replication).
- Hardware/Software: Python 3.10+, PyTorch, MerLin, Perceval; CPU by default, optional CUDA/MPS via `--device`.

Key deviations/assumptions:

- This repository runs simulations only.
- Configuration-driven CLI (`implementation.py`) with JSON configs under `configs/` (run from repo root with `--paper photonic_QCNN`).
- All reusable code lives in `lib/` (per template).
- Readout sweeps and visualizations reside in `lib/run_*readout.py` and `results/`.

## How to Run

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Command-line interface

All `implementation.py` commands must be run from the repository root and include
`--paper photonic_QCNN`.

Main entry point: `implementation.py`.

```bash
python implementation.py --paper photonic_QCNN --help
```

Supported options (can also be set via JSON config):

- `--merlin` forces the MerLin implementation (paper implementation is selected via `configs/paper.json`).
- `--config PATH` load/merge JSON configuration (`configs/defaults.json` and `configs/example.json` available).
- `--datasets BAS Custom BAS` limit runs to listed datasets.
- `--seed INT` global RNG seed (propagated to dataset configs).
- `--outdir DIR` base directory for new run folders (timestamped subfolders are created automatically).
- `--device STR` `cpu`, `cuda:0`, `mps`, etc. MerLin runs move models/tensors accordingly.
- `--n-runs INT`, `--batch-size INT`, `--epochs INT`, `--lr FLOAT`, `--weight-decay FLOAT`, `--gamma FLOAT` override per-dataset restarts, loaders, and training hyperparameters.

Example runs (from repo root):

```bash
# Default MerLin pipeline on all datasets
python implementation.py --paper photonic_QCNN --merlin

# Paper implementation on BAS + MNIST only
python implementation.py --paper photonic_QCNN --config configs/paper.json --datasets BAS MNIST

# Custom experiment from JSON, overriding epochs/lr inline
python implementation.py --paper photonic_QCNN --config configs/example.json --epochs 30 --lr 5e-2
```

Each MerLin call creates `results/run_YYYYMMDD-HHMMSS/<dataset>/` containing configs, metrics, plots, and summaries. Paper run metrics land under `results/run_YYYYMMDD-HHMMSS/`, and model checkpoints are stored under `models/run_YYYYMMDD-HHMMSS/`.

### Data location

- Default data root resolves via the shared helper `paper_data_dir("photonic_QCNN")` (repository `data/photonic_QCNN` by default).
- The bundled `FULL_DATASET_600_samples.bin` (Custom BAS paper split) is automatically copied from this paper folder into the shared data root on first use; place it there yourself if it is missing.
- Relative dataset paths are resolved against the shared data root; absolute paths are honored.
- Override the base directory with `DATA_DIR=/abs/path` or `--data-root /abs/path` when invoking `implementation.py`.

### Output directory and generated files

A typical MerLin run folder contains:

```
results/run_2025-05-13-143210/BAS/
├── args.txt                 # resolved dataset hyperparameters
├── detailed_results.json    # per-run metrics & histories
├── summary.json             # mean/std accuracy statistics
└── BAS_training_plots.png   # loss/train/test curves
```

Override the base directory via `--outdir` or the `outdir` key in the config. Paper run metrics land under `results/run_YYYYMMDD-HHMMSS/`, and model checkpoints are stored under `models/run_YYYYMMDD-HHMMSS/`.

Paper training metric plots can be regenerated with:

```bash
python utils/BAS_paper_plots.py --input-dir results/run_YYYYMMDD-HHMMSS
python utils/custom_BAS_paper_plots.py --input-dir results/run_YYYYMMDD-HHMMSS
python utils/MNIST_paper_plots.py --input-dir results/run_YYYYMMDD-HHMMSS
```

### Figure 12 simulation plots

Replicate the paper's Figure 12 (loss/accuracy trajectories per dataset) with:

```bash
python utils/figure12.py
```

This launches fresh MerLin runs for every dataset in the default config and automatically renders the plots via `utils/simulation_visu.py`. Outputs are stored under `results/figure12/<timestamp>/`, with run artifacts in `runs/<dataset>/` and figures in `figures/` (override the base directory with `--outdir`).

Manual route: run your desired MerLin experiment, then call:

```bash
python utils/simulation_visu.py --detailed-results path/to/detailed_results.json
```

Generated files (one per dataset) follow the pattern:

- `results/figure12/<timestamp>/figures/bas_simulation_results.png`
- `results/figure12/<timestamp>/figures/custom_bas_simulation_results.png`
- `results/figure12/<timestamp>/figures/mnist_simulation_results.png`

### Readout strategy plots

The full Figure 4 reproduction can be triggered with:

```bash
python utils/figure4.py
# optional: python utils/figure4.py --max-iter 25 --n-runs 3  # stop after 25 label assignments for the first readout strategy and repeat second readout strategy only 3 times to shorten running time
```

This runs both readout sweeps (two-fold with k∈{7,8} and the modes-pair variant), then generates all visualizations via `utils/readout_visu.py`. Artifacts are collected under `results/figure4/<timestamp>/` by default (append `--outdir my_dir` to choose a different base directory).

Manual route (still available): run `lib/run_two_fold_readout.py` for k=7 and k=8, `lib/run_modes_pair_readout.py`, then `utils/readout_visu.py`.

> **Note:** `--max_iter` short-circuits the two-fold strategy to the requested number of label assignments and prints a warning that results are incomplete.

Generated outputs include:

- `results/figure4/<timestamp>/figures/first_readout_tain_vs_test_accs.png`
- `results/figure4/<timestamp>/two_fold/k_7/first_readout_detailed_confusion_matrix_k_7.png`
- `results/figure4/<timestamp>/figures/second_readout_accs_vs_modes.png`
- `results/figure4/<timestamp>/modes_pair/second_readout_detailed_confusion_matrix.png`

## Configuration

All configs live in `configs/`:

- `defaults.json` — Short training of the MerLin photonic QCNN on MNIST (0 vs 1).
- `merlin.json` — MerLin reproduction setup (20 epochs, all datasets).
- `paper.json` — paper reproduction setup (20 epochs, all datasets) for the reference implementation.
- `cli.json` — CLI schema consumed by `implementation.py` (not an experiment config).

Config structure:

```json
{
  "implementation": "merlin",
  "datasets": ["BAS", "Custom BAS", "MNIST"],
  "seed": 42,
  "device": "cpu",
  "outdir": "results",
  "training": {
    "lr": 0.1,
    "weight_decay": 0.001,
    "gamma": 0.9,
    "epochs": 20
  },
  "runs": {
    "BAS": { ... dataset-specific overrides ... }
  }
}
```

Keys under `runs.{DATASET}` feed the corresponding runner (`lib/run_BAS.py`, etc.) and accept:

- `n_runs`, `random_states` – number of repetitions and RNG seeds (auto-extended if shorter than `n_runs`).
- `data_source` – `paper` (original splits) or `scratch` (local generation).
- `conv_circuit`, `dense_circuit`, `measure_subset`, `dense_added_modes` – circuit topology knobs.
- `output_proba_type` (`state` / `mode`) and `output_formatting` (`Train_linear`, `Lex_grouping`, `Mod_grouping`).
- `batch_size`, `results_subdir`, `outdir`, `device` – loader and filesystem overrides.

Global CLI overrides (`--lr`, `--batch-size`, etc.) transparently update these entries at runtime.

## Results and Analysis

### Reported vs reproduced accuracies
The paper discusses the great scaling of their proposed architectures regarding the required resources. It also reports 
the number of parameters of their model on the three different datasets as well as their attained training and test 
accuracies:

| Dataset        | Input Size | # Parameters | Train Acc    | Test Acc     |
|----------------|------------|--------------|--------------|--------------|
| BAS            | 4x4        | 10           | 93.7 ± 1.6 % | 93.0 ± 1.2 % |
| Custom BAS     | 4x4        | 10           | 91.3 ± 2.6 % | 92.7 ± 2.1 % |
| MNIST (0 vs 1) | 8x8        | 30           | 95.1 ± 2.9 % | 93.1 ± 3.6 % |

Through additional hyperparameter optimization, we have been able to improve the accuracy of their model (using their code) on Custom BAS and MNIST:

| Dataset        | Input Size | # Parameters | Train Acc     | Test Acc     |
|----------------|------------|--------------|---------------|--------------|
| Custom BAS     | 4x4        | 10           | 97.3 ± 1.6 %  | 98.2 ± 2.0 % |
| MNIST (0 vs 1) | 8x8        | 30           | 100.0 ± 0.0 % | 97.2 ± 1.3 % |

Additionally, our own implementation of their model using MerLin has reached equivalent accuracies:

| Dataset        | Input Size | # Parameters | Train Acc    | Test Acc     |
|----------------|------------|--------------|--------------|--------------|
| BAS            | 4x4        | 10           | 94.7 ± 1.0 % | 93.0 ± 1.1 % |
| Custom BAS     | 4x4        | 10           | 98.4 ± 1.9 % | 98.2 ± 2.2 % |
| MNIST (0 vs 1) | 8x8        | 30           | 99.7 ± 0.5 % | 98.8 ± 1.0 % |

Moreover, they reported their simulation results on BAS, Custom BAS and MNIST through the following figure at a), b) and c) respectively:

![paper_simulation_results](./results/Fig_supplementary_11_15-1.png)

Comparatively, here are our respective simulation results:

![bas_simulation_results](./results/bas_simulation_results.png)
![custom_bas_simulation_results](./results/custom_bas_simulation_results.png)
![mnist_simulation_results](./results/mnist_simulation_results.png)

Then, the first approach considered for the readout (measurement) layer was the clustering of all possible two-photons measurements in two groups for the two labels. This approach was tested on the Custom BAS dataset and led to the results at section b) of the following figure. The second approach, to associate a single pair of modes to label 0, led to the results at section c) of the same figure.

![paper_first_approach](./results/Fig_4_NEW_3-1.png)

We have also implemented these two approaches and the results are the following:

First approach:

![first_approach](./results/first_readout_tain_vs_test_accs.png)

![first_approach_matrix](./results/first_matrix.png)

Second approach:

![second_approach](./results/second_readout_accs_vs_modes.png)

![second_approach_matrix](./results/second_readout_detailed_confusion_matrix.png)

## Hyperparameters

The model uses the following key hyperparameters across different experiments:

### Training Parameters
- **Learning Rate**: 0.1 (with Adam optimizer)
- **Weight Decay**: 0.001
- **Scheduler**: Exponential decay with γ=0.9
- **Epochs**: 20
- **Batch Size**: 6
- **Loss Function**: CrossEntropyLoss

### Model Architecture Parameters
- **Kernel Size**: 2×2 for convolution and pooling
- **Stride**: 2 (same as kernel size)
- **Circuit Types**:
  - Convolution: 'MZI' (Mach-Zehnder Interferometer) or 'BS' (Beam Splitter)
  - Dense: 'MZI', 'BS', or 'paper_6modes' (paper-specific circuit)
- **Additional Modes**: 2 modes inserted in dense layer for improved expressivity
- **Measurement Subset**: 2 modes measured (partial measurement)

### Output Formatting Options
- **'Train_linear'**: Trainable linear layer for classical post-processing
- **'No_train_linear'**: Fixed linear layer (no gradient updates)
- **'Lex_grouping'**: Lexicographic grouping of quantum measurement outcomes
- **'Mod_grouping'**: Modular grouping of quantum measurement outcomes

### Probability Output Types
- **'state'**: Use full Fock state probability distributions
- **'mode'**: Use marginalized per-mode occupation probabilities

## Extensions and Next Steps

1. Benchmark alternative circuit templates (different convolution/pooling primitives) under the same config-driven runner.
2. Port the implementation to hardware backends to validate photonic noise assumptions (aligns with Figure 3 discussion in the paper).
3. Add multi-channel input support (prototype exists internally) and extend datasets beyond binary BAS/MNIST.

## Reproducibility Notes

- Determinism: `implementation.py --paper photonic_QCNN` exposes `--seed` (default 42) and propagates it to dataset loaders + PyTorch/MerLin layers.
- Training determinism depends on backend support (set `torch.backends.cudnn.deterministic = True` externally if needed).
- Captured environment: `results/requirements.txt` contains all the specific environment libraries.
- Saved artifacts: use `models/` for checkpoints, `results/` for plots/metrics, `lib/` for reusable code per template guidelines.

## Testing

Run tests from the repository root:

```bash
pytest -q papers/photonic_QCNN
```

Tests cover the QCNN building blocks (`tests/test_qcnn.py`). Add additional tests under `tests/` as new functionality lands.
