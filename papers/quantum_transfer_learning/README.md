# Transfer Learning in Hybrid Classical-Quantum Neural Networks — MerLin Reproduction

## Paper Reference

**Title**: Transfer Learning in Hybrid Classical-Quantum Neural Networks  
**Authors**: Andrea Mari, Thomas R. Bromley, Josh Izaac, Maria Schuld, Nathan Killoran  
**Affiliation**: Xanadu, Toronto, Canada  
**Published**: Quantum 4, 340 (2020)  
**arXiv**: [1912.08278v2](https://arxiv.org/abs/1912.08278)  
**DOI**: [10.22331/q-2020-10-09-340](https://doi.org/10.22331/q-2020-10-09-340)

## Overview

This repository reproduces key experiments from the quantum transfer learning paper, which extends classical transfer learning to hybrid classical-quantum neural networks. The paper introduces four transfer learning paradigms:

| Scheme | From → To | Description |
|--------|-----------|-------------|
| **CC** | Classical → Classical | Standard transfer learning |
| **CQ** | Classical → Quantum | Pre-trained CNN + variational quantum circuit |
| **QC** | Quantum → Classical | Pre-trained quantum feature extractor + classical network |
| **QQ** | Quantum → Quantum | Quantum-to-quantum transfer |

This reproduction focuses on **Examples 1-3** using two backends:

1. **MerLin (photonic)** - Primary implementation using `QuantumLayer`
2. **PennyLane (qubit)** - Reference implementation for comparison

### Backend Comparison

| Feature | MerLin (Photonic) | PennyLane (Qubit) |
|---------|-------------------|-------------------|
| **Architecture** | Beam splitter meshes | RY gates + CNOTs |
| **Encoding** | Phase shifters (angle encoding) | Hadamard + RY rotations |
| **Measurement** | Fock state probabilities | Pauli-Z expectations |
| **Hardware** | Photonic quantum computers | Superconducting/ion trap |

## Key Concepts

### Dressed Quantum Circuits

A "dressed" quantum circuit augments a bare variational quantum circuit with classical pre/post-processing layers:

```
Q̃ = L_{n_q→n_out} ∘ Q ∘ L_{n_in→n_q}
```

For **MerLin**, the quantum circuit Q uses:
- Beam splitter meshes (trainable interferometers)
- Phase shifters for angle encoding
- Fock state measurements

### CQ Transfer Learning Pipeline

```
[High-res Image] → [Pre-trained ResNet18] → [512 features] 
    → [Classical Layer] → [Photonic VQC / Qubit VQC] → [Classical Layer] → [2 classes]
```

## Reproduced Experiments

### Example 1: 2D Spiral Classification

| Model | Test Accuracy | Paper Reference |
|-------|---------------|-----------------|
| Classical (4-layer) | ~85% | Fig. 2 (left) |
| Dressed Quantum (4-qubit, depth-5) | ~97% | Fig. 2 (right) |

### Example 2: Ants vs Bees (Hymenoptera)

| Backend | Test Accuracy | Paper Reference |
|---------|---------------|-----------------|
| Simulator | 96.7% | Table 2 |
| MerLin | ~96% | — |

### Example 3: CIFAR-10 Binary

| Dataset | Test Accuracy | Paper Reference |
|---------|---------------|-----------------|
| Dogs vs Cats | 82.7% | Table 3 |
| Planes vs Cars | 96.1% | Table 3 |

## Installation

```bash
cd quantum_transfer_learning
pip install -r requirements.txt
```

## Usage

### Run via shared CLI (from repo root)

```bash
# Example 1: 2D Spiral classification
python implementation.py --project quantum_transfer_learning --config quantum_transfer_learning/configs/spiral.json

# Example 2: Ants/Bees classification
python implementation.py --project quantum_transfer_learning --config quantum_transfer_learning/configs/hymenoptera.json

# Example 3: CIFAR-10 binary
python implementation.py --project quantum_transfer_learning --config quantum_transfer_learning/configs/cifar_dogs_cats.json
python implementation.py --project quantum_transfer_learning --config quantum_transfer_learning/configs/cifar_planes_cars.json

# Run all experiments
python implementation.py --project quantum_transfer_learning --config quantum_transfer_learning/configs/all_experiments.json
```

### Run from project directory

```bash
cd quantum_transfer_learning
python ../implementation.py --project . --config configs/spiral.json
```

### CLI help

```bash
# View available CLI options for this reproduction
python implementation.py --project quantum_transfer_learning --help
```

### Quick smoke test

```bash
cd quantum_transfer_learning
pytest -q
```


# Example 1: 2D Spiral classification
python implementation.py --project quantum_transfer_learning --config quantum_transfer_learning/configs/spiral.json

# Example 2: Ants/Bees classification
python implementation.py --project quantum_transfer_learning --config quantum_transfer_learning/configs/hymenoptera.json

# Example 3: CIFAR-10 binary
python implementation.py --project quantum_transfer_learning --config quantum_transfer_learning/configs/cifar_dogs_cats.json
python implementation.py --project quantum_transfer_learning --config quantum_transfer_learning/configs/cifar_planes_cars.json

# Run all experiments
python implementation.py --project quantum_transfer_learning --config quantum_transfer_learning/configs/all_experiments.json
```

### Run from project directory

```bash
cd quantum_transfer_learning
python ../implementation.py --project . --config configs/spiral.json
```

### CLI help

```bash
# View available CLI options for this reproduction
python implementation.py --project quantum_transfer_learning --help
```

### Quick smoke test

```bash
cd quantum_transfer_learning
pytest -q
```


# Example 1: 2D Spiral classification
python implementation.py --project quantum_transfer_learning --config quantum_transfer_learning/configs/spiral.json

# Example 2: Ants/Bees classification
python implementation.py --project quantum_transfer_learning --config quantum_transfer_learning/configs/hymenoptera.json

# Example 3: CIFAR-10 binary
python implementation.py --project quantum_transfer_learning --config quantum_transfer_learning/configs/cifar_dogs_cats.json
python implementation.py --project quantum_transfer_learning --config quantum_transfer_learning/configs/cifar_planes_cars.json

# Run all experiments
python implementation.py --project quantum_transfer_learning --config quantum_transfer_learning/configs/all_experiments.json
```

### Run from project directory

```bash
cd quantum_transfer_learning
python ../implementation.py --project . --config configs/spiral.json
```

### Quick smoke test

```bash
cd quantum_transfer_learning
pytest -q
```

## Configuration Options

### Common parameters (all configs)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `seed` | Random seed for reproducibility | 42 |
| `device` | Torch device (`cpu`, `cuda:0`, `mps`) | `cpu` |
| `dtype` | Tensor dtype | `float32` |
| `outdir` | Output directory | `outdir` |

### MerLin-specific parameters

| Parameter | Description | Values |
|-----------|-------------|--------|
| `computation_space` | Photonic computation subspace | `fock`, `unbunched`, `dual_rail` |
| `n_photons` | Number of photons in input state | 2 (default) |
| `n_modes` | Number of optical modes (maps to n_qubits) | 4 (default) |

**Computation Space Options:**
- `fock`: Full Fock space with PNR detectors
- `unbunched`: At most one photon per mode (threshold detectors) - **default**
- `dual_rail`: Logical qubit encoding (one photon per mode pair)

### Experiment-specific parameters

#### Spiral (`configs/spiral.json`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_samples` | Total data points | 2200 |
| `n_qubits` | Number of qubits | 4 |
| `q_depth` | Quantum circuit depth | 5 |
| `epochs` | Training epochs | 50 |

#### Transfer Learning (`configs/hymenoptera.json`, `configs/cifar_*.json`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pretrained_model` | Feature extractor backbone | `resnet18` |
| `n_qubits` | Number of qubits | 4 |
| `q_depth` | Quantum circuit depth | 6 |
| `epochs` | Training epochs | 30 |
| `batch_size` | Batch size | 4 |
| `learning_rate` | Initial learning rate | 0.0004 |

## Project Structure

```
quantum_transfer_learning/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── notebook.ipynb            # Interactive exploration
├── configs/
│   ├── cli.json              # CLI parameter definitions
│   ├── runtime.json          # Runner configuration
│   ├── example.json          # Quick test config
│   ├── spiral.json           # Example 1 config
│   ├── hymenoptera.json      # Example 2 config
│   ├── cifar_dogs_cats.json  # Example 3a config
│   ├── cifar_planes_cars.json# Example 3b config
│   └── all_experiments.json  # Full reproduction
├── lib/
│   ├── __init__.py
│   ├── runner.py             # Main entry point for MerLin CLI
│   ├── circuits.py           # Quantum circuit definitions
│   ├── models.py             # Hybrid classical-quantum models
│   ├── datasets.py           # Data loading utilities
│   ├── training.py           # Training loops
│   └── visualization.py      # Plotting utilities
├── data/                     # Downloaded datasets (auto-populated)
├── models/                   # Saved model checkpoints
├── results/                  # Generated figures and metrics
│   ├── fig2_spiral.png
│   ├── fig3_hymenoptera.png
│   └── summary.json
├── tests/
│   └── test_smoke.py         # Validation tests
└── utils/                    # Additional utilities
```

## Results

Results are saved to `outdir/run_YYYYMMDD-HHMMSS/` including:

- `config_snapshot.json` — Configuration used
- `summary_results.json` — Accuracy metrics
- `training_curves.png` — Loss/accuracy plots
- `*.pt` — Model checkpoints

## Citation

```bibtex
@article{mari2020transfer,
  title={Transfer learning in hybrid classical-quantum neural networks},
  author={Mari, Andrea and Bromley, Thomas R and Izaac, Josh and Schuld, Maria and Killoran, Nathan},
  journal={Quantum},
  volume={4},
  pages={340},
  year={2020},
  publisher={Verein zur F{\"o}rderung des Open Access Publizierens in den Quantenwissenschaften},
  doi={10.22331/q-2020-10-09-340}
}
```

## License

This reproduction is provided under the same CC-BY 4.0 license as the original paper.
