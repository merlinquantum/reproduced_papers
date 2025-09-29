# Quantum Self-Supervised Learning (qSSL)

Reproduction of: “Quantum Self-Supervised Learning” (Jaderberg et al.), arXiv:2103.14653 — https://arxiv.org/abs/2103.14653

In this folder, you will find an implementation and evaluation of the core ideas from the paper. It supports three representation networks under the same SSL pipeline: a photonic (MerLin/Perceval) model, a gate-model (Qiskit) model, and a classical MLP baseline.

The default backend in this repo is MerLin (photonic).

## What is reproduced
- Dataset and task: CIFAR-10, restricted to the first k labels (e.g., k=5).
- Training: Self-supervised pretraining with InfoNCE on two augmented views (SimCLR-style), followed by linear evaluation with a frozen encoder.
- Models (representation network):
  - MerLin photonic circuit (Perceval + Merlin quantum layer)
  - Qiskit parameterized circuit (via `qSSL/qnn`)
  - Classical MLP baseline
- Metrics: SSL losses over epochs and linear-probing accuracy curves; checkpoints and run metadata saved per experiment.

## Results reproduced

Pretrained models in `./results` give the following results:

| Number of epochs | Number of classes (CIFAR10) | Qiskit based | Classical SSL | Quantum SSL (`no_bunching=False`) | Quantum SSL (`no_bunching=True`) |
|------------------|-----------------------------|--------------|---------------|----------------------------------|---------------------------------|
| 2                | 5                           | 48.37 <br> ✅ OK <br> #32 <br> x0.08/x0.008 | 48.08 <br> 🚫 <br> #144 <br> x1/x1 | 8 modes: **49.22** <br> #184 <br> x0.97/x0.95 <br><br> 10 modes: 47.28 <br> #320 <br> x0.89/x0.88 <br><br> 12 modes: 46.46 <br> #488 <br> x0.83/x0.65 | 8 modes: 45.58 <br> #184 <br> x0.97/x0.97 <br><br> 10 modes: 45.58 <br> #320 <br> x0.97/x0.93 <br><br> 12 modes: 45.76 <br> #488 <br> x0.94/x0.82 |
| 5                | 5                           | 47.88  | 49.04 | 8 modes: 49.9 <br><br> 10 modes: **51.12** <br><br> 12 modes: 50.64 | 8 modes: 49.3 <br><br> 10 modes: 48.86 <br><br> 12 modes: **51.74** |

Legend:
- #number of parameters
- x ... speed-up (relative to classical baseline)

Overall, we reproduced the results highlighted in the paper and we have a photonic implementantion of it, using MerLin, that is faster and more accurate (but has more trainable parameters).

## Project structure
- `implementation.py` — main entry point (replaces the old `main.py`)
- `lib/` — core library modules used by scripts
  - `data_utils.py` — datasets, transforms (SSL and linear eval)
  - `model.py` — backbone, representation networks (MerLin/Qiskit/Classical), projection head
  - `training_utils.py` — InfoNCE, training loops, metrics and results I/O
  - `config.py` — JSON config loading and defaults
- `configs/` — example JSON configs (default uses MerLin)
  - `qssl_default.json`
- Other
  - `linear_probing.py` — evaluate frozen features with a linear head. Pretrained models can be found in `results/`
  - `requirements.txt` — Python dependencies
  - `utils/`, `tests/` — placeholders following the template

## Install
```bash
python -m venv ssl-venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start
- Run with default MerLin settings (from JSON config):
```bash
python implementation.py --config configs/qssl_default.json
```
- CLI alternatives (override or skip configs):
```bash
# MerLin (photonic)
python implementation.py --merlin --classes 5 --modes 10 --epochs 2 --batch_size 256 --ckpt-step 1

# Qiskit (gate-model)
python implementation.py --qiskit --classes 5 --epochs 2 --batch_size 256 --ckpt-step 1

# Classical baseline
python implementation.py --classes 5 --epochs 2 --batch_size 256 --ckpt-step 1
```

## Configuration (JSON)
See `configs/qssl_default.json`. Key fields:
- `dataset`: `root`, `classes`, `batch_size`
- `model`: `backend` (`merlin` | `qiskit` | `classical`), `width`, `loss_dim`, `batch_norm`, `temperature`
- Qiskit-specific: `layers`, `encoding`, `q_ansatz`, `q_sweeps`, `activation`, `shots`, `q_backend`
- MerLin-specific: `modes`, `no_bunching`
- `training`: `epochs`, `ckpt_step`, `le_epochs`

You can combine `--config` with CLI overrides. The runner resolves the final configuration and saves it to the results directory (`args.json`).

## Training pipeline (pedagogical overview)


![SSL Model](SSL_model.png)

1) SSL pretraining
- Input: for each image, generate two strong augmentations (query/key) using `TwoCropsTransform`.
- Backbone: ResNet18 (final FC replaced by Identity).
- Compression: Linear layer to `width` (quantum-friendly size).
- Representation network (choose one): MerLin, Qiskit, or Classical MLP.
- Projection head: MLP to `loss_dim` with BN + ReLU.
- Loss: InfoNCE (temperature τ) on the two views.

2) Linear evaluation
- Freeze backbone + compression + representation.
- Train a linear classifier on top using lightly augmented train data and minimal val transforms.
- Report accuracy curves and final/best validation accuracy.



## Models explained

- MerLin (default)
  - Photonic circuit built with Perceval: two trainable interferometers around a phase-encoding layer.
  - Features are Sigmoid-normalized and scaled by 1/π to map into phase parameters.
  - Parameters: `modes` (number of photonic modes), `no_bunching` (photon statistics), `width` (input feature size to the circuit), plus trainable circuit phases.

- Qiskit (gate-model)
  - Representation network `QNet` with `n_qubits = width`.
  - Configurable `encoding`, `q_ansatz`, `layers`, `q_sweeps`, `activation`, `shots`, and `q_backend` (e.g., `qasm_simulator`).

- Classical baseline
  - Simple MLP with `args.layers` repetitions of Linear(width, width) + LeakyReLU.

## Outputs and checkpoints
Results are saved under `results/<backend>/<timestamp>/`:
- `args.json` — resolved arguments used for the run
- `training_metrics.json` — SSL and linear-eval losses/accuracies over epochs
- `experiment_summary.json` — consolidated summary with final and best val accuracy
- `model-cl-<classes>-epoch-<n>.pth` — checkpoints saved every `ckpt_step` epochs

## Linear probing only
Evaluate pretrained encoders with a frozen representation and train a linear head:
```bash
# Single checkpoint file
python linear_probing.py --pretrained ./results/merlin/<timestamp>/model-cl-5-epoch-5.pth

# Or point to a results directory (auto-discovers .pth files)
python linear_probing.py --pretrained ./results/merlin/<timestamp>/
```

## Acknowledgments
- Original paper: Quantum Self-Supervised Learning — https://arxiv.org/abs/2103.14653
- Portions of the Qiskit pipeline and general approach are inspired by the original authors’ resources where relevant.

## Troubleshooting
- For Qiskit, ensure `qiskit-aer` is installed and the selected backend (e.g., `qasm_simulator`) is available.

## Tests:
Tests are in the ./tests folder and contain tests to validate one forward pass in the classical, MerLin and Qiskit models as well as a test on the InfoNCE loss. Once the environment is installed, you can run them
```
python3 -m pytest tests/
```