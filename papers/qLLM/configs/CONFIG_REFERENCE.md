# qLLM Configuration Reference Guide

## Quick Config Selection

### MerLin Photonic Models
| Config | Purpose | Best For |
|--------|---------|----------|
| `merlin-basic.json` | Sandwich architecture with single encoder | Baseline quantum model |
| `merlin-parallel.json` | Multiple encoders with concatenation | Richer representations |
| `merlin-expectation.json` | Deep circuits with expectation values | Novel measurement strategy |
| `merlin-kernel.json` | Quantum kernel methods | Few-shot learning |

### TorchQuantum Gate-based Models
| Config | Qubits | Layers | Encoders | Best For |
|--------|--------|--------|----------|----------|
| `torchquantum-lightweight.json` | 8 | 1+2 | 1 | Fast training |
| `torchquantum-medium.json` | 10 | 2+3 | 1 | Balanced performance |
| `torchquantum-parallel.json` | 10+8 | 2+1+2 | 2 | Parallel encoding |
| `torchquantum-deep.json` | 12 | 3+4 | 1 | Maximum expressivity |
| `torchquantum.json` | 10 | 1+2 | 1 | Default template |

### Classical Baselines
| Config | Model | Purpose |
|--------|-------|---------|
| `mlp.json` | Multi-layer perceptrons | Tests 5 hidden dimensions |
| `svm.json` | Support vector machine | RBF kernel with 2 C values |
| `log-reg.json` | Logistic regression | Simple linear baseline |

## Configuration File Parameters

### Common Across All Models
```json
{
  "seed": 42,                          // Random seed for reproducibility
  "device": "auto",                    // "cuda", "cpu", or "auto"
  "outdir": "results",                 // Output directory
  "dataset": {
    "name": "sst2",                    // Dataset name
    "eval_size": 250,                  // Validation set size
    "embeddings_dir": "embeddings"     // Pre-computed embeddings directory
  },
  "embeddings": {
    "model_name": "sentence-transformers/paraphrase-mpnet-base-v2"
  },
  "training": {
    "epochs": 50,
    "learning_rate": 1e-4,
    "batch_size": 16
  }
}
```

### MerLin Model Parameters
```json
{
  "model": {
    "name": "merlin-[basic|parallel|expectation|kernel]",
    "embedding_dim": 768,               // Always 768 for sentence transformers
    "hidden_dim": 100,                  // Compressed embedding dimension (50-200)
    "quantum_modes": 12,                // Photonic modes (8-12)
    "no_bunching": false,               // Set true for parallel/expectation models
    "photons": 5,                       // Max photon number (0 = auto)
    "e_dim": 1                          // Number of encoders (1-2)
  }
}
```

### TorchQuantum Model Parameters
```json
{
  "model": {
    "name": "torchquantum",
    "embedding_dim": 768,
    "hidden_dim": 100,
    "encoder_configs": [                // List of encoder configurations
      {
        "n_qubits": 10,                 // Qubits in encoder (6-12)
        "n_layers": 2,                  // Parameterized layers (1-3)
        "connectivity": 1               // 1=NN, 2=extended
      }
    ],
    "pqc_config": [                     // Single QPU configuration
      {
        "n_qubits": 10,
        "n_main_layers": 2,             // Main circuit layers (2-4)
        "connectivity": 1,
        "n_reuploading": 2              // Data re-uploading blocks (1-3)
      }
    ],
    "e_dim": 1                          // Must match encoder_configs length
  }
}
```

### Classical Model Parameters
```json
{
  "model": {
    "name": "mlps|svm|log-reg",
    "embedding_dim": 768,
    
    // MLP-specific:
    "hidden_dims": [0, 48, 96, 144, 192],
    
    // SVM-specific:
    "C_values": [1.0, 100.0],
    "kernel": "rbf"
  }
}
```

## Usage Examples

### Run with Config File
```bash
python implementation.py --config configs/merlin-basic.json
python implementation.py --config configs/torchquantum-medium.json
python implementation.py --config configs/mlp.json
```

### Override Config Parameters
```bash
python implementation.py --config configs/merlin-basic.json \
  --quantum-modes 10 \
  --hidden-dim 150 \
  --epochs 100 \
  --learning-rate 5e-5
```

### Run Without Config File
```bash
python implementation.py --model merlin-basic \
  --quantum-modes 12 \
  --hidden-dim 100 \
  --epochs 50 \
  --learning-rate 1e-4
```

## Key Design Principles

### MerLin Models
- **Basic**: Simple, fast, good for baseline comparisons
- **Parallel**: Use `e_dim=2` and `no_bunching=true` for multiple encoders
- **Expectation**: Requires `no_bunching=true`, uses deep circuits with per-mode measurements
- **Kernel**: For quantum kernel-based classification

### TorchQuantum Models
- **encoder_configs**: One entry per encoder; length must match `e_dim`
- **pqc_config**: Single-element list defining the quantum processing unit
- **Connectivity**: 1=nearest-neighbor (default), 2=extended (more expressive, costlier)
- **Re-uploading**: More blocks = deeper effective circuit but more parameters

### Training
- Quantum models: `learning_rate` 1e-4 to 1e-5 (typically lower than classical)
- Classical models: `learning_rate` 1e-3 to 1e-4
- `kernel_batch_size` for MerLin kernel methods (default: 32)

## Common Parameter Ranges

| Parameter | Recommended Range | Default |
|-----------|-------------------|---------|
| quantum_modes | 8-12 | 12 |
| hidden_dim | 50-200 | 100 |
| n_qubits (encoder) | 6-12 | 10 |
| n_qubits (QPU) | 6-12 | 10 |
| n_layers (encoder) | 1-3 | 2 |
| n_main_layers (QPU) | 2-4 | 3 |
| connectivity | 1-2 | 1 |
| n_reuploading | 1-3 | 2 |
| learning_rate (quantum) | 1e-5 to 1e-4 | 1e-4 |
| learning_rate (classical) | 1e-4 to 1e-3 | 1e-3 |
| epochs | 50-100 | 50 |
