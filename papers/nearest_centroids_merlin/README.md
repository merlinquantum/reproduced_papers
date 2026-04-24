# Nearest Centroid Classification on a Trapped Ion Quantum Computer — MerLin Reproduction

A reproduction of the quantum nearest-centroid classification algorithm from [Johri et al. (2020)](https://arxiv.org/abs/2012.04145), implemented with three parallel backends: a Cirq quantum circuit simulator, the MerLin analytical photonic simulator, and a classical baseline using scikit-learn.

## Paper

> S. Johri, S. Debnath, A. Mocherla, A. Singh, A. Prakash, J. Kim, and I. Kerenidis, "Nearest Centroid Classification on a Trapped Ion Quantum Computer," *PRX Quantum*, vol. 2, no. 1, 2021. [arXiv:2012.04145](https://arxiv.org/abs/2012.04145)

The paper demonstrates a quantum nearest-centroid classifier that encodes classical data into quantum states via amplitude encoding, computes inner products through a photonic-style interference circuit using Reconfigurable Beam Splitter (RBS) gates, and classifies test points by assigning them to the nearest class centroid in quantum distance space.

## How It Works

1. **Dimensionality reduction** — PCA projects features down to `n_components` dimensions (must be a power of 2), which equals the number of qubits/modes.
2. **Amplitude encoding** — A `VectorLoader` circuit loads a classical vector into the amplitudes of a quantum state using a tree of RBS gates.
3. **Quantum inner product** — The loader for x is applied followed by the inverse loader for y. The probability of measuring the initial basis state gives |⟨x|y⟩|².
4. **Quantum distance** — The Euclidean distance is recovered as √(‖x‖² + ‖y‖² − 2‖x‖‖y‖·⟨x̂|ŷ⟩).
5. **Classification** — scikit-learn's `NearestCentroid` is used with the quantum distance as a custom metric.

Three classifiers run in parallel on every experiment:

| Backend | Class | Description |
|---|---|---|
| Classical | `sklearn.neighbors.NearestCentroid` | Standard Euclidean nearest centroid (baseline) |
| Cirq | `QuantumNearestCentroid` | Gate-level circuit simulation via Google Cirq |
| MerLin | `MLQuantumNearestCentroid` | Analytical matrix-level simulation via Perceval/MerLin |

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: cirq (≥1.4), scikit-learn (≥1.7.2), merlinquantum (≥0.2.3), matplotlib (≥3.8), torchvision (≥0.20), seaborn (≥0.13), pytest (≥8.0). MNIST data is downloaded automatically on first run.

## Usage

### Reproduce all paper figures

From the repo root:

```bash
./papers/nearest_centroids_merlin/run.sh
```

This runs all configs for Figures 8 (synthetic), 9 (Iris), and 11 (MNIST), then copies the final reproduced artifacts into `results/`.

### Run a single experiment

Via the shared runtime (from repo root):

```bash
python implementation.py --paper nearest_centroids_merlin --config configs/mnist_0v1.json
```

Standalone (from the project directory):

```bash
cd papers/nearest_centroids_merlin
python implementation.py --config configs/iris_ns1000.json
```

### CLI options

| Flag | Description |
|---|---|
| `--n-components` | PCA components / number of qubits (power of 2) |
| `--n-shots` | Measurement shots for quantum simulation |
| `--n-repeats` | Repeated experiments for statistics |
| `--max-samples` | Cap on dataset samples |

All flags override the values set in the config file. Run `--help` for the full list including global runtime options.

## Configuration

Each JSON config under `configs/` specifies a complete experiment. Configs inherit from `configs/defaults.json` and can override any field. Key parameters:

```jsonc
{
  "seed": 42,
  "outdir": "outdir",
  "dataset": {
    "name": "mnist",          // "mnist", "iris", or "synthetic"
    "n_components": 8          // PCA dimensions = qubits (must be power of 2)
  },
  "training": {
    "n_shots": 1000,           // Quantum circuit repetitions per distance calc
    "n_repeats": 10,           // Independent train/test splits for error bars
    "test_size": 0.5           // Fraction held out for testing
  },
  "experiments": [
    {
      "classes": [0, 1],       // Which class labels to include
      "max_samples": 80        // Total samples drawn from dataset
    }
  ]
}
```

Synthetic data configs additionally accept `n_clusters`, `n_points_per_cluster`, `min_centroid_distance`, `gaussian_variance`, and `sphere_radius`.

### Pre-built configs and their paper figures

| Config(s) | Paper Figure | Description |
|---|---|---|
| `synthetic_4q_*.json`, `synthetic_8q_*.json` | Fig 8 | Synthetic data across qubit/class counts |
| `iris_ns100.json`, `iris_ns500.json`, `iris_ns1000.json` | Fig 9 | Iris dataset: effect of shot count |
| `mnist_0v1.json`, `mnist_2v7.json` | Fig 11 | MNIST binary classification |
| `mnist_4class.json`, `mnist_10class.json` | — | Extended MNIST multi-class experiments |

## Outputs

This project follows the repo-wide output policy:

- **`outdir/`** (gitignored) — Raw run outputs. Each run creates a timestamped subdirectory containing the config snapshot, run log, per-experiment JSON results, generated figures, and a `done.txt` marker on completion. This is ephemeral workspace; you can delete old runs freely.
- **`results/`** (tracked) — Curated reproduced artifacts only: final figures, summary results, label comparison tables. This is what gets committed to the repo. The `run.sh` script copies artifacts from the latest completed run automatically; for single experiments, copy them manually.

## Tests

```bash
pytest tests/ -v
```

Smoke tests verify that both quantum classifiers can fit and predict on synthetic data and that accuracy exceeds random chance on well-separated clusters.

## License

See the repository root for license information.