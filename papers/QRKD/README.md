# Quantum Relational Knowledge Distillation (QRKD)

## Reference and Attribution

- Title: Quantum Relational Knowledge Distillation
- arXiv: https://arxiv.org/abs/2508.13054
- License/attribution: cite the arXiv preprint when using this reproduction

## Overview

This folder reproduces MNIST and CIFAR-10 baselines for KD, RKD, and QRKD (the original paper also reports text experiments; this repo focuses on the vision side):
- Teacher CNN, KD student, RKD student, QRKD student (KD + RKD + fidelity kernel), and scratch student.
- Datasets: MNIST (28x28 grayscale) and CIFAR-10 (32x32 RGB) via `--dataset` and matching configs.
- QRKD fidelity kernel options:
  - `simple`: classical |⟨ψ_i|ψ_j⟩|² over L2-normalized features (no quantum backend; serves as a baseline fidelity term).
  - `merlin`: `merlin.algorithms.kernels.FidelityKernel.simple` over features (quantum-inspired kernel library).
  - `qiskit`: `qiskit_machine_learning.kernels.FidelityQuantumKernel` with a ZZFeatureMap.
- Architectures (mirroring the paper’s compact CNNs): Teacher is a 3-layer conv stack (1→18 channels, 3×3) with max-pool, adaptive 4×4 downsample, 1×1 conv to 12 channels, then MLP (12·4·4 → 31 → 10); ~6.7K params on MNIST. Student keeps the topology but is narrower (19→4 channels, hidden 4); ~1.7K params. CIFAR-10 variants switch to 3 input channels, same topology.

Loss composition (student):
`L = L_task + kd * L_KD + dr * L_RKD-distance + ar * L_RKD-angle + qk * L_fidelity`
- KD: KL on softened logits (temperature, alpha).
- RKD-distance/angle: pairwise relational alignment of embeddings.
- Fidelity: kernel alignment between student/teacher normalized features (backend as above).

## How to Run

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Command-line interface

Main entry point (from repo root): `python implementation.py --paper QRKD [--config ...]`. All flags are defined in `cli.json`.

```bash
python implementation.py --paper QRKD --help
```

Example overrides (see `cli.json` for the authoritative list):

- `--config PATH` Load an additional JSON config (merged over `defaults.json`), e.g. `--config QRKD/configs/suite_simple_mnist_10epochs.json`.
- `--seed INT` Random seed applied via the generic runtime.
- `--dtype STR` Force a global torch/NumPy dtype (e.g., `float32`, `float64`).
- `--device STR` Device override (`cpu`, `cuda:0`, `mps`, etc.).
- `--log-level LEVEL` Logging verbosity (`DEBUG`, `INFO`, ...).
- `--outdir DIR` Base output directory (default in `defaults.json`).
- `--epochs INT` Override `training.epochs`.
- `--batch-size INT` Override `dataset.batch_size`.
- `--lr FLOAT` Override `training.lr`.
- `--qkernel-weight FLOAT` Set the fidelity kernel (QRKD) loss weight.
- `--qkernel-backend {simple,merlin,qiskit}` Choose backend for the fidelity kernel (simple = |⟨ψ_i|ψ_j⟩|² over normalized features; merlin uses `FidelityKernel.simple`; qiskit uses `qiskit_machine_learning.kernels.FidelityQuantumKernel` with a ZZFeatureMap).

Example runs:

```bash
# Quick sanity (defaults: 1 epoch, teacher+KD+RKD+QRKD+scratch)
python implementation.py --paper QRKD

# Longer run (teacher + KD/RKD/QRKD + scratch) for MNIST/10 epochs
python implementation.py --paper QRKD --config QRKD/configs/suite_simple_mnist_10epochs.json

# CIFAR-10 teacher only (10 epochs)
python implementation.py --paper QRKD --config QRKD/configs/teacher_cifar10_10epochs.json

# CIFAR-10 suite (simple backend, override to 5 epochs)
python implementation.py --paper QRKD --config QRKD/configs/suite_simple_cifar10_10epochs.json --epochs 5

# Full suite runner (from repo root), override dataset/epochs if needed
bash QRKD/utils/run_full_suite.sh --dataset mnist --epochs 10

# Students only (requires a teacher checkpoint path)
python implementation.py --paper QRKD --config QRKD/configs/student_kd_mnist_10epochs.json --training.teacher_path /path/to/teacher.pt
```

The script saves a snapshot of the resolved config alongside results and logs.

### Output directory and generated files

Each run writes to `<outdir>/run_YYYYMMDD-HHMMSS/` with:
- `config_snapshot.json` resolved config
- `run.log` full logs
- `history_*.json` per-variant metrics (loss/accuracy per epoch)
- `metrics_*.json` final test accuracy per student
- `*.pt` checkpoints (teacher and students)

Notes:
- Change the base output directory with `--outdir` or by editing `configs/defaults.json` (key `outdir`).
- Placeholder guard: if any config value contains `<<...>>` (e.g. `"<<TEACHER_PATH>>"`), the runner aborts until you provide a real path (or pass `--teacher-path`).
- Data root: downloads land in `<DATA_DIR>/QRKD` by default (or `<repo>/data/QRKD` if `DATA_DIR` is unset). Override the base root with `--data-root`/`DATA_DIR`; the paper subfolder is appended automatically. Leave `dataset.root` as `null` to use this shared resolver.

## Configuration

Configs live in `configs/`:
- `defaults.json` base settings (MNIST 1 epoch by default).
- `cli.json` CLI schema.
- Task configs: `teacher_{dataset}_10epochs.json`, `student_{scratch|kd|rkd}_...`, `student_qrkd_{simple|merlin|qiskit}_...`, and suite configs `suite_{simple|merlin|qiskit}_{dataset}_10epochs.json`.
- Placeholders `"<<TEACHER_PATH>>"` must be replaced or overridden via `--teacher-path`; the runner fails fast if unresolved.

## Visualization and reports

- Aggregate table: `python QRKD/utils/report.py --teacher RUN --scratch RUN --kd RUN --rkd RUN --qrkd-simple RUN --qrkd-merlin RUN [--qrkd-qiskit RUN]`
- Plot histories: `python QRKD/utils/plot_history.py --teacher RUN --scratch RUN --kd RUN --rkd RUN --qrkd-simple RUN --qrkd-merlin RUN [--qrkd-qiskit RUN] --out-json results/history_combined_<dataset>_<epochs>epochs.json --out-plot results/accuracy_plot_<dataset>_<epochs>epochs.png`
- Notebook demo: `QRKD/notebook.ipynb` trains on a 10k MNIST subset for 3 epochs and visualizes KD vs RKD vs QRKD.

## End-to-end suite script

`bash QRKD/utils/run_full_suite.sh --dataset mnist --epochs 10` runs teacher, scratch, KD, RKD, QRKD (simple + merlin) and writes report/plots named with `<dataset>_<epochs>epochs` in `results/`. Pass `--dataset cifar10` for CIFAR-10; qiskit backend is optional and currently commented out in the script because it is slow.
