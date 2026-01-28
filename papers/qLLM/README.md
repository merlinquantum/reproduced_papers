# qLLM: Quantum Large Language Models

This repository implements various quantum and classical machine learning models for sentiment analysis, as described in [Quantum Large Language Model Fine Tuning](https://arxiv.org/abs/2504.08732). The models can be trained and evaluated using pre-computed embeddings from sentence transformers.

**Dataset**: SST-2 (Stanford Sentiment Treebank) binary sentiment classification dataset from [Hugging Face](https://huggingface.co/datasets/SetFit/sst2)

## Reference and Attribution

- Paper: Quantum Large Language Model Fine Tuning (arXiv, 2025)
- Authors: Quantum Large Language Model Fine-Tuning
- DOI/ArXiv: https://arxiv.org/abs/2504.08732
- Original repository (if any): Not specified in the paper.
- License and attribution notes: Cite the paper above and this reproduction repository when using these results or code.

## Overview

The codebase provides implementations of:
- **MerLin quantum models** (4 variants: basic, parallel, expectation, kernel)
- **TorchQuantum quantum models** (gate-based with multiple encoding strategies)
- **Classical models** (MLPs, SVM, Logistic Regression)

## Project Structure

```
qLLM/
├── configs/              # Model configurations
│   ├── cli.json         # CLI argument schema
│   ├── defaults.json    # Default configuration
│   ├── torchquantum.json# TorchQuantum template
│   ├── merlin-basic.json
│   ├── merlin-parallel.json
│   ├── merlin-expectation.json
│   ├── merlin-kernel.json
│   ├── mlp.json
│   ├── svm.json
│   └── log-reg.json
├── lib/                  # Model implementations
│   ├── merlin_llm_models.py     # MerLin models
│   ├── torchquantum_models.py   # TorchQuantum models
│   ├── classical_models.py      # Classical models
│   ├── merlin_kernel.py         # Quantum kernel methods
│   ├── setfit_model.py          # SetFit utilities
│   └── runner.py                # Training runner
├── embeddings/          # Pre-computed embeddings (train/eval/test splits)
├── results/             # Training outputs and results
├── tests/               # Unit tests
├── utils/               # Utility scripts
├── implementation.py    # CLI entry point
├── qllm.ipynb          # Jupyter notebook examples
└── requirements.txt     # Dependencies
```

Dataset helpers live in `papers/shared/qLLM/data_utils.py` so they can be reused across papers.
## Configuration System

The project uses JSON-based configuration files to specify model parameters. The CLI system allows overriding config values via command-line arguments.

### Configuration Files Structure

**Global settings** (shared across all models):
- `dataset`: Dataset name, size, embeddings directory path
- `embeddings`: Sentence transformer model name
- `training`: Epochs, learning rate, batch size
- `seed`, `device`, `logging`: Runtime settings

**Model-specific parameters**: Each model type has its own config file with architecture-specific parameters.

### Configuration Files by Model Type

#### MerLin Models (Photonic Quantum Computing)
**File structure for all MerLin variants:**
```json
{
  "model": {
    "name": "merlin-[basic|parallel|expectation|kernel]",
    "embedding_dim": 768,
    "hidden_dim": 100,
    "quantum_modes": 12,
    "no_bunching": false,
    "photons": 5,
    "e_dim": 1
  },
  "training": {...}
}
```

**Parameters:**
- `embedding_dim`: Input embedding dimension from sentence transformer (usually 768)
- `hidden_dim`: Dimension of compressed embedding space before quantum encoding
- `quantum_modes`: Number of photonic modes in the MerLin circuit (8-12 typical)
- `no_bunching`: Disable photon bunching for angle-based encoding (useful for parallel encoders)
- `photons`: Maximum photon number (0 = auto-computed as modes/2)
- `e_dim`: Number of parallel encoders (1 for basic/expectation/kernel, 1-2 for parallel)

**Specific variants:**
- `merlin-basic`: Simple sandwich architecture, single encoder
- `merlin-parallel`: Multiple encoders with concatenated outputs, set `e_dim: 1` or `e_dim: 2`
- `merlin-expectation`: Deep circuits with expectation value measurements, requires `no_bunching: true`
- `merlin-kernel`: Quantum kernel methods using MerLin circuits

#### TorchQuantum Models (Gate-based Quantum Computing)
**File structure:**
```json
{
  "model": {
    "name": "torchquantum",
    "embedding_dim": 768,
    "hidden_dim": 100,
    "encoder_configs": [
      {"n_qubits": 10, "n_layers": 1, "connectivity": 1}
    ],
    "pqc_config": [
      {"n_qubits": 10, "n_main_layers": 2, "connectivity": 1, "n_reuploading": 2}
    ],
    "e_dim": 1
  },
  "training": {...}
}
```

**Parameters:**

*encoder_configs* (list of objects):
- `n_qubits`: Number of qubits in each encoder (typically 6-12)
- `n_layers`: Number of parameterized layers (1-2 for basic, 2-3 for deeper)
- `connectivity`: Qubit connectivity pattern (1=nearest neighbor, 2=extended)
- Multiple encoders for parallel processing; their outputs are fused based on `e_dim`

*pqc_config* (single-element list, defines Quantum Processing Unit):
- `n_qubits`: Number of qubits in main circuit (typically same as encoder)
- `n_main_layers`: Number of main parameterized layers (2-4 typical)
- `connectivity`: Qubit connectivity (1 or 2)
- `n_reuploading`: Number of data re-uploading blocks (1-3 typical)

*e_dim*: Number of parallel encoders (1-2); must match `encoder_configs` length

#### Classical Models

**MLPs** (`mlp.json`):
```json
{
  "model": {
    "name": "mlps",
    "embedding_dim": 768,
    "hidden_dims": [0, 48, 96, 144, 192]
  },
  "training": {...}
}
```
Tests multiple MLP architectures with varying hidden dimensions. Each configuration is trained separately.

**SVM** (`svm.json`):
```json
{
  "model": {
    "name": "svm",
    "embedding_dim": 768,
    "C_values": [1.0, 100.0]
  },
  "training": {...}
}
```
SVM with RBF kernel; tests different regularization strengths.

**Logistic Regression** (`log-reg.json`):
```json
{
  "model": {
    "name": "log-reg",
    "embedding_dim": 768
  },
  "training": {...}
}
```
Standard logistic regression classifier.

### Using Configuration Files

```bash
# Use a specific config file (overrides defaults.json)
python implementation.py --config configs/merlin-basic.json

# Override specific values via CLI
python implementation.py --config configs/merlin-basic.json --learning-rate 5e-5 --epochs 100

# Or just use defaults and override via CLI
python implementation.py --model merlin-basic --hidden-dim 50 --quantum-modes 10
```
## Data Generation (Important Setup)

**We strongly recommend generating embeddings using `generate_embeddings.py` in a separate environment** to avoid library conflicts between SetFit and TorchQuantum. These libraries have incompatible dependencies that can cause issues if used in the same environment.

1. Create a separate conda/venv environment with SetFit installed
2. Install `requirements.txt` (or at least `setfit`, `datasets`, and `torch`) in that environment
3. Run `python utils/generate_embeddings.py` from `papers/qLLM/` in that environment
4. Switch back to your main environment with TorchQuantum/MerLin for model training

## Quick Start

Run a model using the shared CLI:

```bash
# From inside papers/qLLM
python implementation.py --config configs/merlin-basic.json

# From the repo root
python implementation.py --paper qLLM --config configs/merlin-basic.json

# Override parameters via CLI
python implementation.py --config configs/merlin-basic.json --learning-rate 1e-5 --epochs 100
```

## Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The CLI schema is defined in `cli.json`. Default values are in `configs/defaults.json`. Model-specific configs are in `configs/[model-name].json`.

## Output Directory and Config Snapshot

Each run writes into a timestamped folder under the base `outdir` (default: `results`):

```
results/
└── run_YYYYMMDD-HHMMSS/
    ├── config_snapshot.json  # Resolved config after merges/CLI overrides
    ├── metrics.json          # Model metrics for the run
    └── SKIPPED.txt           # Present only if a run is skipped
```

- Change the base output directory with `--outdir` or `configs/defaults.json`.
- `config_snapshot.json` is generated by the shared runner before training starts.
- `metrics.json` is written by `lib/runner.py` at the end of a successful run.

## Available Models & Architecture Details

### MerLin Models (Photonic Quantum Computing)

MerLin models use photonic quantum computing with Mach-Zehnder interferometers and photon detection. Configuration: `configs/merlin-[basic|parallel|expectation|kernel].json`

#### 1. Basic MerLin (`merlin-basic`)
**Architecture:** Sandwich architecture with trainable Mach-Zehnder interferometers
- **Encoding**: 768-dim embeddings → Linear layer → Sigmoid normalization → Phase shifters
- **Quantum layer**: Left interferometer → Data encoding → Right interferometer
- **Measurement**: Photon number detection (bunching by default)
- **Output**: Linear layer → class predictions

**Config parameters**: `quantum_modes` (8-12), `hidden_dim` (50-200), `no_bunching` (false)

#### 2. Parallel MerLin (`merlin-parallel`)
**Architecture:** Multiple independent encoders with concatenated outputs
- **First module**: E parallel encoders (E=1 or 2), each processes full input independently
- **Encoding method**: Angle encoding (typically with `no_bunching: true` for stability)
- **Second module**: Processes concatenated encoder outputs through quantum circuit
- **Advantage**: Richer representation through multiple encoding paths

**Config parameters**: `e_dim` (1 or 2), `no_bunching` (true recommended), other params as basic

#### 3. Expectation MerLin (`merlin-expectation`)
**Architecture:** Deep circuits with expectation value measurements
- **Quantum layer**: `create_quantum_circuit_deep` with layered phase shifter encoding
- **Measurement strategy**: Computes per-mode photon presence probability (expectation values)
- **Novel aspect**: Marginalize photon presence using `marginalize_photon_presence` function
- **Processing**: Two-stage architecture similar to parallel, with expectation-based outputs

**Config parameters**: `no_bunching` (true required), `quantum_modes`, `hidden_dim`

#### 4. Kernel MerLin (`merlin-kernel`)
**Architecture:** Quantum kernel methods using MerLin circuits
- **Purpose**: Compute quantum kernels k(x,y) for kernel classification methods
- **Uses**: MerLin circuit to generate feature maps, then trains kernel classifier
- **Suitable for**: Few-shot or low-data regimes where kernels provide advantages

**Config parameters**: `quantum_modes`, `hidden_dim`, `kernel_batch_size` (training parameter)

### TorchQuantum Models (Gate-based Quantum Computing)

TorchQuantum models implement gate-based quantum circuits with qubits. Configuration: `configs/torchquantum.json`

**Data Encoding Methods:**
1. **Amplitude Encoding**: Classical vector embedded as quantum amplitudes: |ψ(x)⟩ = Σᵢ xᵢ |i⟩
2. **Angle Encoding**: Data determines rotation angles: |ψ(x)⟩ = RY(x₁) ⊗ RY(x₂) ⊗ ... ⊗ RY(xₙ) |0⟩⊗ⁿ

**Model Structure:**
- **First Module**: Multi-encoder with E parallel quantum encoders
  - Each encoder: Amplitude encoding → Parameterized quantum circuit → Pauli-Z measurements
  - Output fusion: concatenation (default), averaging, or weighted combination
- **Second Module**: Quantum Processing Unit (QPU) with data re-uploading
  - Angle encoding → Re-uploading blocks → Main parameterized circuit → Single qubit measurement

## Results

> “We observe up to 3.14% improvements in accuracy over classical architectures of comparable model size, within the set of hyperparameters probed.”

Classical baselines (5-fold mean±std with best test accuracy):

| Model | Mean±Std | Best test | Params |
| --- | --- | --- | --- |
| SVM (C=1) | 0.8912±0.0038 | 0.8955 | 296 |
| SVM (C=100) | 0.8889±0.0045 | 0.8932 | 435 |
| Logistic Regression | 0.8888±0.0043 | 0.8933 | 769 |
| NN [0] | 0.8886±0.0043 | 0.8934 | 1,538 |
| NN [48] | 0.8897±0.0098 | 0.8946 | 37,010 |
| NN [96] | 0.8912±0.0038 | 0.8933 | 74,018 |
| NN [144] | 0.8839±0.0034 | 0.8896 | 111,026 |
| NN [192] | 0.8827±0.0085 | 0.8901 | 148,034 |

MerLin sweep snapshot (simple QuantumLayer): mode=8, 1 photon yields 0.8874±0.0071 with best test 0.8924.

Best MerLin results by model type (for 200 epochs):

| MerLin model | Best test accuracy | Source table | `n_modes`| `n_photons`| `computation_space` | hidden dim |
| --- | --- | --- | --- | --- |  --- | --- | 
| merlin-basic (simple QuantumLayer) | 0.8951 | Using a simple QuantumLayer | 12 | 1 | UNBUNCHED | 8 |
| merlin-parallel (angle encoding) | 0.8890±0.0069 | Using a similar architecture as in the paper (only angle encoding) | 12 | 4 | FOCK | 64 |
| merlin-expectation (expectation values) | 0.8874±0.0092 | Using a similar architecture as in the paper (only angle encoding) but with expectation values | 12 modes | 2 photons | UNBUNCHED | 128 |
| merlin-kernel (fidelity kernel) | 0.7460±0.0060 | Another approach using a Fidelity Kernel | 12 | 2 | FOCK | . |

Note: the fidelity-kernel table reports mean±std only; the value above is the best mean observed.
- **Output**: Linear layer combines both modules for final classification

**Config parameters:**

*encoder_configs* (list): Configurations for parallel encoders
- `n_qubits`: Number of qubits (typically 6-10)
- `n_layers`: Number of parameterized layers (1-2 for lightweight, 2-4 for deeper)
- `connectivity`: 1=nearest neighbor, 2=extended connectivity

*pqc_config* (single-element list): QPU configuration
- `n_qubits`: Usually same as encoder qubits
- `n_main_layers`: Number of main circuit layers (2-4 typical)
- `connectivity`: Qubit connectivity pattern
- `n_reuploading`: Data re-uploading blocks (1-3 typical)

*e_dim*: Number of parallel encoders (1-2); must match `encoder_configs` length

**Example usage:**
```bash
python implementation.py --config configs/torchquantum.json
```

### Classical Models

Classical baselines for comparison. Configurations: `configs/[mlp|svm|log-reg].json`

#### Multi-Layer Perceptrons (`mlps`)
**Architecture:** Tests multiple configurations with varying hidden dimensions
- **Configurations**: hidden_dim in [0, 48, 96, 144, 192]
- **Each MLP**: Linear → BatchNorm → ReLU → Linear (or direct Linear if hidden_dim=0)
- **Regularization**: Batch normalization and dropout
- **Optimization**: Adam with exponential learning rate decay

#### Support Vector Machine (`svm`)
**Architecture:** RBF kernel SVM
- **Configurations**: C values [1.0, 100.0] for different regularization strengths
- **Kernel**: RBF with automatic scaling
- **Parameter count**: C=1.0 (~296 params), C=100.0 (~435 params)

#### Logistic Regression (`log-reg`)
**Architecture:** Standard linear classifier with logistic loss
- **Simple baseline**: Direct linear classification without hidden layers

## Usage Examples

### Using Configuration Files

```bash
# Basic MerLin with config file
python implementation.py --config configs/merlin-basic.json

# Parallel MerLin with 2 encoders
python implementation.py --config configs/merlin-parallel.json

# TorchQuantum model
python implementation.py --config configs/torchquantum.json

# Override config parameters from CLI
python implementation.py --config configs/merlin-basic.json \
  --quantum-modes 10 --hidden-dim 50 --epochs 100
```

### Command-line Only (without config files)

```bash
# Basic MerLin
python implementation.py --model merlin-basic \
  --quantum-modes 8 --hidden-dim 100 --epochs 50 --learning-rate 1e-4

# Parallel MerLin with 2 encoders
python implementation.py --model merlin-parallel \
  --quantum-modes 10 --e-dim 2 --no-bunching

# TorchQuantum with inline encoder specs
python implementation.py --model torchquantum \
  --encoder-configs '[{"n_qubits": 10, "n_layers": 2, "connectivity": 1}]' \
  --pqc-config '[{"n_qubits": 10, "n_main_layers": 2, "connectivity": 1, "n_reuploading": 2}]'

# Classical baselines
python implementation.py --config configs/mlp.json
python implementation.py --config configs/svm.json
python implementation.py --config configs/log-reg.json
```

## Implementation Notes

#### Model Implementation Notes

This model is inspired by  [Quantum Large Language Model Fine Tuning](https://arxiv.org/abs/2504.08732). Our goal was to reproduce this model and the results of this paper. However, some specificities of the model and training parameters are not clear:

**Data Handling**
- Custom dataset splitting approach differs from the original paper, potentially affecting few-shot learning performance comparisons
- Data encoding procedure lacks specification for handling cases where embedding dimension doesn't match Hilbert space dimension (no guidance on truncation vs. padding strategies)

**Model Architecture**
- Output represents measurements of 1 qubit, but the final classification layer accepts input of shape `Q_c + 1` (discrepancy not explained in paper)
- When using E = 2 encoders in the first module, the paper doesn't specify how the two outputs are merged or concatenated before forwarding to the second module
- Final hyperparameters selected after the hyperparameter study are not clearly documented

**Training Configuration**
- Weight decay is explored in hyperparameter studies, but the learning rate scheduler implementation is not specified
- **Note**: This implementation does not incorporate noise modeling

These ambiguities may lead to differences between our results and those reported in the original paper.

### Classical Comparison
```bash
python implementation.py --model mlps --hidden-dim 100 --epochs 100
python implementation.py --model svm
python implementation.py --model log-reg
```

## Command Line Arguments

### Dataset Parameters
- `--dataset`: Dataset name (default: sst2)
- `--eval-size`: Validation set size (default: 250)

### Model Parameters
- `--model-name`: Pre-trained sentence transformer model
- `--embedding-dim`: Input embedding dimension (default: 768)
- `--hidden-dim`: Hidden layer dimension (default: 100)

### Training Parameters
- `--epochs`: Training epochs (default: 5)
- `--learning-rate`: Learning rate (default: 1e-5)
- `--batch-size`: Batch size (default: 16)

### MerLin-specific
- `--quantum-modes`: Number of photonic modes (default: 8)
- `--no-bunching`: Disable photon bunching
- `--photons`: Max photons (0 = modes/2)
- `--e-dim`: Number of parallel encoders (default: 1)

### TorchQuantum-specific
- `--encoder-configs`: JSON list of encoder configurations
- `--pqc-config`: QPU configuration

### Execution
- `--seed`: Random seed (default: 42)
- `--device`: Device (cuda/cpu/auto)
- `--verbose`: Verbose output

## Data Requirements

The models expect pre-computed embeddings in `./embeddings/` directory with:
- `train/` split for training data
- `eval/` split for validation data  
- `test/` split for test data

Each split should contain embedding files that can be loaded by the `create_dataset_from_embeddings` function in `papers/shared/qLLM/data_utils.py`.

## Architecture Comparison

| Model | Quantum Framework | Key Innovation | Measurement |
|-------|------------------|----------------|-------------|
| MerLin Basic | Photonic | Sandwich interferometer | Photon counting |
| MerLin Parallel | Photonic | Parallel encoding | Concatenated outputs |
| MerLin Expectation | Photonic | Deep circuits + expectation values | Per-mode occupancy |
| TorchQuantum | Gate-based | Dual-module + data re-uploading | Pauli-Z + single qubit |
| Classical MLPs | N/A | Multiple configurations | Softmax |

## Dependencies

- `requirements.txt`: full reproduction stack (MerLin + TorchQuantum + SetFit)
- `torch`: PyTorch framework
- `merlin`: MerLin quantum computing framework
- `perceval`: Photonic quantum computing
- `torchquantum`: Gate-based quantum ML
- `sklearn`: Classical ML algorithms
- `numpy`: Numerical computations

## Model Testing

Both quantum frameworks include gradient propagation tests:
- MerLin: `test_module_building_and_gradients()` in `lib/merlin_llm_models.py`
- TorchQuantum: `test_gradient_propagation()` in `lib/torchquantum_models.py`

Run tests individually:
```bash
python lib/merlin_llm_models.py
python lib/torchquantum_models.py
```
