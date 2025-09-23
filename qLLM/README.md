# qLLM: Quantum Large Language Models

This repository implements various quantum and classical machine learning models for sentiment analysis, as described in a recent work : [Quantum Large Language Model Fine Tuning](https://arxiv.org/abs/2504.08732). The models can be trained and evaluated using pre-computed embeddings from sentence transformers.

The dataset used in this repo is the SST-2 (Stanford Sentiment Treebank) binary sentiment classification dataset that can be found on [Hugging Face](https://huggingface.co/datasets/SetFit/sst2) .
## Overview

The codebase provides implementations of:
- **MerLin quantum models** (3 variants)
- **TorchQuantum models** 
- **Classical models** (MLPs, SVM, Logistic Regression)
- **Quantum kernel methods**

## Data Generation (Important Setup)

**We strongly recommend generating embeddings using `generate_embeddings.py` in a separate environment** to avoid library conflicts between SetFit and TorchQuantum. These libraries have incompatible dependencies that can cause issues if used in the same environment.

1. Create a separate conda/venv environment with SetFit installed
2. Run `python src/generate_embeddings.py` in that environment
3. Switch back to your main environment with TorchQuantum/MerLin for model training

## Quick Start

Run a model using the main script:

```bash
python src/main.py --model [MODEL_TYPE]
```

## Available Models

### MerLin Models (`merlin_llm_utils.py`)

MerLin models use photonic quantum computing with interferometers and photon detection.

#### 1. Basic MerLin (`--model merlin-basic`)
**Architecture:**
- **Data encoding**: 768-dimensional embeddings → Linear layer → Sigmoid normalization → Phase shifters
- **Quantum layer**: Sandwich architecture with trainable interferometers
  - Left interferometer (trainable beam splitters + phase shifters)
  - Data encoding via phase shifters
  - Right interferometer (trainable beam splitters + phase shifters)
- **Measurement**: Photon number detection with bunching/no-bunching options
- **Output**: Linear layer maps measurements to class predictions

**Key parameters:**
- `--quantum-modes`: Number of photonic modes (default: 8)
- `--no-bunching`: Enable/disable photon bunching
- `--hidden-dim`: Size of compressed embedding space

#### 2. Parallel MerLin (`--model merlin-parallel`)
**Architecture:**
- **First module**: E parallel quantum encoders (E=1 or E=2)
  - Each encoder processes full input independently
  - Uses angle encoding (no bunching for parallelization)
  - Outputs concatenated for second module
- **Second module**: Processes concatenated outputs through quantum circuit
- **Enhanced capacity**: Multiple encoding paths for richer representation

#### 3. Expectation MerLin (`--model merlin-expectation`)
**Architecture:**
- **Deep circuits**: Uses `create_quantum_circuit_deep` with layered encoding
- **Expectation measurement**: Computes probability of ≥1 photon per mode
- **Novel measurement strategy**: `marginalize_photon_presence` function computes per-mode occupation probabilities
- **Two-stage processing**: Similar to parallel but with expectation-based measurements

#### 4. Kernel MerLin (`--model merlin-kernel`)
Uses quantum kernel methods with MerLin circuits for similarity computation.

### TorchQuantum Models (`torchquantum_utils.py`)

TorchQuantum models use gate-based quantum computing with qubits from [Quantum Large Language Model Fine Tuning](https://arxiv.org/abs/2504.08732).

#### Architecture Components

**Data Encoding Methods:**
1. **Amplitude Encoding**: Classical vector embedded as quantum state amplitudes |ψ(x)⟩ = Σᵢ xᵢ |i⟩
2. **Angle Encoding**: Data determines qubit rotation angles |ψ(x)⟩ = RY(x₁) ⊗ RY(x₂) ⊗ ... ⊗ RY(xₙ) |0⟩⊗n

**Model Structure (`--model torchquantum`):**
- **First Module**: Multi-encoder with configurable parallel quantum encoders
  - Each encoder: Amplitude encoding → Parameterized quantum circuit → Pauli-Z measurements
  - Fusion methods: concatenation, averaging, or weighted combination
- **Second Module**: Quantum Processing Unit (QPU) with data re-uploading
  - Angle encoding → Multiple re-uploading blocks → Main PQC → Single qubit measurement
- **Output**: Linear layer combines both module outputs for classification

**Key parameters:**
- `--encoder-configs`: JSON list of encoder configurations
- `--pqc-config`: QPU configuration
- `--e-dim`: Number of parallel encoders

**Example configuration:**
```bash
python src/main.py --model torchquantum \
  --encoder-configs '[{"n_qubits": 10, "n_layers": 2, "connectivity": 1}]' \
  --pqc-config '[{"n_qubits": 10, "n_main_layers": 2, "connectivity": 1, "n_reuploading": 2}]'
```

### Classical Models (`classical_utils.py`)

#### Multi-Layer Perceptrons (`--model mlps`)
**Architecture:**
- Tests multiple MLP configurations with varying hidden dimensions [0, 48, 96, 144, 192]
- Each MLP: Linear → BatchNorm → ReLU → Linear (or direct Linear if hidden_dim=0)
- Batch normalization and dropout for regularization
- Adam optimizer with exponential learning rate decay

#### Support Vector Machine (`--model svm`)
**Two configurations:**
- SVM with C=1.0 (targeting ~296 parameters)
- SVM with C=100.0 (targeting ~435 parameters)
- RBF kernel with automatic scaling

#### Logistic Regression (`--model log-reg`)
Standard logistic regression classifier.

## Usage Examples

### Basic MerLin Model
```bash
python src/main.py --model merlin-basic \
  --quantum-modes 8 \
  --hidden-dim 100 \
  --epochs 50 \
  --learning-rate 1e-4
```

### Parallel MerLin with 2 encoders
```bash
python src/main.py --model merlin-parallel \
  --quantum-modes 10 \
  --e-dim 2 \
  --no-bunching
```

### TorchQuantum Model

```bash
python src/main.py --model torchquantum \
  --encoder-configs '[{"n_qubits": 8, "n_layers": 2, "connectivity": 1}, {"n_qubits": 6, "n_layers": 1, "connectivity": 1}]' \
  --pqc-config '[{"n_qubits": 8, "n_main_layers": 3, "connectivity": 2, "n_reuploading": 2}]'
```

This model is inspired by  [Quantum Large Language Model Fine Tuning](https://arxiv.org/abs/2504.08732). Our goal was to reproduce this model and the results of this paper. However, some specificities of the model and training parameters are not clear:

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
python src/main.py --model mlps --hidden-dim 100 --epochs 100
python src/main.py --model svm
python src/main.py --model log-reg
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

Each split should contain embedding files that can be loaded by the `create_dataset_from_embeddings` function in `data_utils.py`.

## Architecture Comparison

| Model | Quantum Framework | Key Innovation | Measurement |
|-------|------------------|----------------|-------------|
| MerLin Basic | Photonic | Sandwich interferometer | Photon counting |
| MerLin Parallel | Photonic | Parallel encoding | Concatenated outputs |
| MerLin Expectation | Photonic | Deep circuits + expectation values | Per-mode occupancy |
| TorchQuantum | Gate-based | Dual-module + data re-uploading | Pauli-Z + single qubit |
| Classical MLPs | N/A | Multiple configurations | Softmax |

## Dependencies

- `torch`: PyTorch framework
- `merlin`: MerLin quantum computing framework  
- `perceval`: Photonic quantum computing
- `torchquantum`: Gate-based quantum ML
- `sklearn`: Classical ML algorithms
- `numpy`: Numerical computations

## Model Testing

Both quantum frameworks include gradient propagation tests:
- MerLin: `test_module_building_and_gradients()` in `merlin_llm_utils.py`
- TorchQuantum: `test_gradient_propagation()` in `torchquantum_utils.py`

Run tests individually:
```bash
python src/merlin_llm_utils.py
python src/torchquantum_utils.py
```