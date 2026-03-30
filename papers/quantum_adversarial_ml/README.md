# Quantum Adversarial Machine Learning

Reproduction of "Quantum Adversarial Machine Learning" by Lu et al. (2020).

**Paper**: [arXiv:2001.00030](https://arxiv.org/abs/2001.00030)

**Authors**: Sirui Lu, Lu-Ming Duan, Dong-Ling Deng

## Overview

This implementation demonstrates the vulnerability of quantum classifiers to adversarial perturbations using MerLin's photonic quantum computing framework. Similar to classical neural networks, quantum classifiers can be deceived by carefully-crafted adversarial examples that are imperceptible to humans.

### Key Findings from the Paper

1. **Vulnerability**: Quantum classifiers are vulnerable to adversarial examples, regardless of whether the input data is classical (images) or quantum (ground states).

2. **Attack Methods**: Standard gradient-based attacks (FGSM, BIM, PGD, MIM) effectively generate adversarial perturbations for quantum classifiers.

3. **Transferability**: Adversarial examples generated for classical classifiers can transfer to quantum classifiers.

4. **Defense**: Adversarial training can significantly improve robustness against specific types of attacks.

## Paper Coverage

| Section | Content | Status |
|---------|---------|--------|
| III.A | Quantum classifier architecture, training | ✅ |
| III.B.1 | White-box untargeted attacks (FGSM, BIM, PGD, MIM) | ✅ |
| III.B.1 | Functional attacks (local unitaries) | ✅ |
| III.B.2 | White-box targeted attacks | ✅ |
| III.B.3 | Black-box transfer attacks (Table III) | ✅ |
| III.B | Noise vs adversarial comparison (Fig 11) | ✅ |
| III.C | Topological phase classification (QAH) | ✅ |
| III.D | Ising model phase classification | ✅ |
| IV | Adversarial training defense | ✅ |

## Implementation

This reproduction provides **two quantum backends** for comparison:

### 1. Photonic (MerLin)
- **Beam splitter meshes** for variational/entangling layers
- **Phase shifters** for angle encoding of input data
- **Fock state measurements** for classification output

### 2. Gate-Based (PennyLane) 
- **Euler rotations** (Z-X-Z) as in the original paper
- **CNOT ladder** for entanglement
- **Pauli-Z measurements** for output

This allows direct comparison of adversarial vulnerability between photonic and qubit-based quantum classifiers.

### Experiments

1. **MNIST Classification** (Section III.B)
   - Binary classification: digits 1 vs 9
   - Multi-class: digits 1, 3, 7, 9

2. **Adversarial Attacks** (Section III.A-B)
   - Additive attacks: FGSM, BIM, PGD, MIM
   - Functional attacks: Local phase shifter perturbations
   - Targeted and untargeted variants

3. **Black-box Transfer Attacks** (Section III.B.3)
   - CNN surrogate → Quantum target
   - FNN surrogate → Quantum target

4. **Topological Phase Classification** (Section III.C)
   - Quantum Anomalous Hall (QAH) effect
   - Time-of-flight image classification

5. **Ising Model** (Section III.D)
   - Phase classification (ferromagnetic vs paramagnetic)

6. **Adversarial Defense** (Section IV)
   - Adversarial training via robust optimization

7. **Photonic vs Gate-Based Comparison** (New)
   - Train both MerLin and PennyLane classifiers
   - Compare adversarial vulnerability

## Usage

### Jupyter Notebook (Recommended for Learning)

The easiest way to understand the paper and code is through the interactive notebook:

```bash
cd notebooks/
jupyter notebook quantum_adversarial_ml.ipynb
```

The notebook covers:
- Paper overview and key findings
- Code structure walkthrough
- Training quantum classifiers step-by-step
- Running adversarial attacks with visualizations
- Reproducing all paper figures
- Extended amplitude vs angle encoding comparison

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run example experiment (from repo root)
python implementation.py --paper quantum_adversarial_ml --config configs/example.json

# Or from within the paper directory
cd papers/quantum_adversarial_ml
python ../../implementation.py --config configs/example.json
```

### Reproduce All Paper Results

```bash
# Run all experiments (takes several hours)
bash scripts/reproduce_all.sh
```

### Individual Figures/Tables

Each figure/table is reproduced by a shell script that orchestrates multiple single-run configs:

```bash
# Figure 7: Robustness sweep (accuracy vs epsilon)
bash scripts/figure7_robustness.sh

# Figure 11: Noise vs adversarial comparison
bash scripts/figure11_noise_comparison.sh

# Figure 16: Adversarial training defense
bash scripts/figure16_defense.sh

# Table III: Black-box transfer attacks
bash scripts/table3_transfer.sh

# Section III.C: Topological phase classification
bash scripts/topological_phases.sh
```

### Single-Run Configs

Each config represents ONE experiment run. For parameter sweeps, use the shell scripts or run multiple configs:

```bash
# Train quantum classifier
python implementation.py --paper quantum_adversarial_ml \
    --config configs/train_quantum.json \
    --outdir outdir/train_quantum

# Attack at specific epsilon (requires trained model)
python implementation.py --paper quantum_adversarial_ml \
    --config configs/defaults.json \
    experiment=attack \
    attack.epsilon=0.1 \
    options.load_model=outdir/train_quantum/model.pt \
    --outdir outdir/attack_eps01
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `experiment` | Experiment type: mnist, attack, defense, ising, topological, transfer, noise | mnist |
| `dataset.digits` | MNIST digit classes | [1, 9] |
| `dataset.image_size` | Image resize dimension | 16 |
| `dataset.train_samples` | Subsample training data (for fast tests) | null (use all) |
| `dataset.test_samples` | Subsample test data (for fast tests) | null (use all) |
| `model.type` | Model type (see below) | hybrid_quantum |
| `model.n_modes` | Number of optical modes (photonic) | 8 |
| `model.n_layers` | Circuit depth (p) | 2 |
| `attack.method` | Attack method: fgsm, bim, pgd, mim, functional | bim |
| `attack.epsilon` | Perturbation magnitude | 0.1 |

### Config Files

| Config | Purpose | Runtime |
|--------|---------|---------|
| `defaults.json` | **Smoke test** - minimal settings for CI | ~30s |
| `example.json` | Quick demo with small dataset | ~10s |
| `full_experiment.json` | Paper-faithful reproduction | ~hours |
| `train_quantum.json` | Train quantum classifier only | ~30min |
| `attack.num_iter` | Attack iterations | 2 |
| `defense.epochs` | Adversarial training epochs | 100 |

### Model Types

| Type | Backend | Description |
|------|---------|-------------|
| `hybrid_quantum` | MerLin | Photonic hybrid classifier (default) |
| `quantum` | MerLin | Direct photonic classifier |
| `hybrid_gate` | PennyLane | Gate-based hybrid classifier |
| `gate_quantum` | PennyLane | Direct gate-based classifier |
| `simple_gate` | PennyLane | Simple angle-encoding gate classifier |
| `classical` | PyTorch | MLP baseline |
| `cnn` | PyTorch | CNN baseline |
| `fnn` | PyTorch | FNN baseline |

## Project Structure

```
quantum_adversarial_ml/
├── notebooks/
│   └── quantum_adversarial_ml.ipynb  # Interactive tutorial
├── configs/
│   ├── defaults.json           # Default configuration
│   ├── cli.json                # CLI parameter definitions  
│   ├── train_quantum.json      # Train amplitude-encoded classifier
│   ├── train_angle.json        # Train angle-encoded classifier
│   ├── attack_bim_eps*.json    # BIM attacks at various epsilon
│   ├── noise_comparison.json   # Noise vs adversarial (Fig 11)
│   ├── adversarial_training.json  # Defense (Fig 16)
│   ├── topological.json        # QAH phases (Sec III.C)
│   ├── ising.json              # Ising model (Sec III.D)
│   └── example.json            # Quick smoke test
├── scripts/
│   ├── reproduce_all.sh        # Run all experiments
│   ├── train_classifiers.sh    # Train MNIST classifiers
│   ├── figure7_robustness.sh   # Accuracy vs epsilon sweep
│   ├── figure11_noise_comparison.sh  # Noise vs adversarial
│   ├── figure16_defense.sh     # Adversarial training
│   ├── table3_transfer.sh      # Black-box transfer attacks
│   ├── topological_phases.sh   # QAH classification
│   └── encoding_comparison.sh  # Amplitude vs angle encoding
├── lib/
│   ├── __init__.py             # Library exports
│   ├── circuits.py             # MerLin quantum circuits
│   ├── models.py               # Quantum + Classical models
│   ├── datasets.py             # MNIST, Ising, QAH datasets
│   ├── attacks.py              # FGSM, BIM, PGD, MIM, functional, transfer
│   ├── defense.py              # Adversarial training
│   ├── training.py             # Training utilities
│   ├── visualization.py        # Plotting functions
│   └── runner.py               # Main entry point
├── tests/
│   └── test_smoke.py           # Validation tests
├── README.md
└── requirements.txt
```

## Key Implementation Details

### Amplitude Encoding (Paper's Approach)

The paper uses amplitude encoding where classical data is directly encoded into quantum state amplitudes:

|ψ_x⟩ = Σᵢ xᵢ |i⟩ (normalized to unit L2 norm)

For the MerLin photonic implementation:
- Fock states |n₁, n₂, ..., nₘ⟩ with n₁ + ... + nₘ = n_photons
- **Unbunched** (default): Dimension = C(n_modes, n_photons) (at most 1 photon per mode)
- **Fock**: Dimension = C(n_modes + n_photons - 1, n_photons) (bunched photons allowed)
- Uses `amplitude_encoding=True` with `MeasurementStrategy.PROBABILITIES`

**Important**: For faithful paper reproduction (no classical compression), choose n_modes and n_photons so that:
```
C(n_modes, n_photons) >= input_dim
```

**Recommended configurations:**

| Dataset | Input Dim | n_modes | n_photons | Fock Dim |
|---------|-----------|---------|-----------|----------|
| MNIST (16×16) | 256 | 13 | 3 | C(13,3)=286 |
| Topological (20×20) | 400 | 15 | 3 | C(15,3)=455 |
| Ising (8 spins) | 256 | 13 | 3 | C(13,3)=286 |

```python
from lib.circuits import MerLinAmplitudeClassifier

# MNIST: 256 pixels → 286 Fock states (no compression!)
model = MerLinAmplitudeClassifier(
    input_dim=256,        # MNIST 16x16
    n_outputs=2,          # binary classification
    n_modes=13,           # optical modes
    n_photons=3,          # photon number → C(13,3)=286 states
    n_layers=2,          # circuit depth
    computation_space="unbunched"
)
```

Use `model_type="amplitude_quantum"` in configs for amplitude encoding.

### Angle Encoding (Alternative Approach)

Angle encoding represents classical data in the rotation angles of phase shifters:

x → PS(φ = x * scale)  (phase shifter with angle proportional to input)

Architecture (sandwich pattern from VQC literature):
```
[Trainable BS Mesh] → [Angle Encoding via PS] → [Trainable BS Mesh]
```

This is different from amplitude encoding:
- **Amplitude**: 256 values → 286 Fock states (direct, no compression)
- **Angle**: 256 values → 13 phase shifters (requires classical compression)

```python
# Hybrid with classical encoder for high-dimensional input
from lib.models import HybridQuantumClassifier

model = HybridQuantumClassifier(
    input_dim=256,        # MNIST 16x16
    n_outputs=2,          # binary classification
    hidden_dims=[128, 64], # Classical compression: 256→128→64→13
    n_modes=13,           # Same as amplitude for fair comparison
    n_photons=3,
    n_layers=3            # Sandwich layers
)
```

Use `model_type="hybrid_angle"` or `model_type="angle_quantum"` in configs.

### Encoding Comparison Experiment

Compare three encoding strategies for adversarial robustness:

```bash
./scripts/encoding_comparison.sh
```

This compares:
1. **AMPLITUDE (direct)**: 256 pixels → 286 Fock states (no compression)
2. **AMPLITUDE+COMPRESSION**: 256 pixels → 64 → 66 Fock states  
3. **ANGLE**: 256 pixels → 13 phase shifters

**Key Finding**: Classical compression significantly improves robustness!

| Encoding | Clean Acc | Adv Acc (ε=0.10) | Robustness |
|----------|-----------|------------------|------------|
| AMPLITUDE (direct) | 98% | 0% | 0% |
| AMPLITUDE+COMPRESS | 100% | 32% | 32% |
| ANGLE | 100% | 36% | 36% |

The classical bottleneck limits the adversary's ability to craft precise perturbations.

### Functional Attacks (Photonic-Native)

The paper describes functional attacks using local unitary perturbations. For photonics, this maps naturally to local phase shifters:

```python
# Additive attack: x_adv = x + δ
# Functional attack: x_adv = x * cos(δ)  # Phase rotation
```

### Photon Loss Noise (Fig 11 Equivalent)

The paper compares adversarial perturbations against depolarizing noise.
For photonic systems, we use **photon loss** as the natural noise source:

```python
from lib.attacks import compare_noise_vs_adversarial

results = compare_noise_vs_adversarial(
    model=model,
    dataloader=test_loader,
    epsilon_values=[0.01, 0.05, 0.1, 0.2],
    attack_method="bim"
)
# Results include: adversarial, random_uniform, random_gaussian, photon_loss
```

MerLin also supports Perceval's `NoiseModel` for realistic photon loss:

```python
import perceval as pcvl

experiment = pcvl.Experiment(circuit)
experiment.noise = pcvl.NoiseModel(
    brightness=0.9,      # 10% source loss
    transmittance=0.85   # 15% propagation loss
)
```

### Black-box Transfer Attacks

We train classical CNN/FNN models as surrogates, generate adversarial examples, and measure transferability to the quantum classifier:

```
CNN/FNN (surrogate) → Generate adversarial examples → Test on Quantum classifier
```

### Topological Phase Classification

Uses the Quantum Anomalous Hall (QAH) model Hamiltonian. The Chern number determines topology:
- C₁ ≠ 0: Topological phase
- C₁ = 0: Trivial phase

## Results Summary

### MNIST Binary Classification (1 vs 9)

| Metric | Value |
|--------|-------|
| Clean Accuracy | ~98% |
| Adversarial Accuracy (BIM, ε=0.1) | ~15% |
| Average Fidelity | ~0.92 |

### Transfer Attack Results (Table III)

| Surrogate | Attack | Target Acc Drop |
|-----------|--------|-----------------|
| CNN | BIM | ~26% |
| CNN | FGSM | ~40% |
| FNN | BIM | ~24% |
| FNN | FGSM | ~35% |

### Adversarial Training Defense

After adversarial training:
- Clean accuracy: ~98%
- Adversarial accuracy (BIM): ~95%

## Differences from Original Paper

### Gate-Based (PennyLane) - Faithful Reproduction
- **Architecture**: Identical to paper (Z-X-Z rotations + CNOT ladder)
- **Encoding**: Amplitude encoding as described in paper
- **Backend**: PennyLane default.qubit simulator

### Photonic (MerLin) - Novel Comparison
- **Architecture**: Beam splitter meshes instead of RY + CNOT layers  
- **Encoding**: Phase shifter angle encoding
- **Functional attacks**: Phase shifter perturbations (photonic-native)
- **Scale**: Smaller circuit sizes for practical simulation

The photonic implementation allows studying whether adversarial vulnerability transfers across different quantum computing paradigms.

## References

```bibtex
@article{lu2020quantum,
  title={Quantum Adversarial Machine Learning},
  author={Lu, Sirui and Duan, Lu-Ming and Deng, Dong-Ling},
  journal={Physical Review Research},
  volume={2},
  pages={033212},
  year={2020}
}
```

## License

See repository root for license information.
