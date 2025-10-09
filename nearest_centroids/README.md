# Nearest Centroid Classification on a Trapped Ion Quantum Computer — Reproduction Template

## Reference and Attribution

- **Paper:** *Nearest centroid classification on a trapped ion quantum computer* (npj Quantum Information, 2021)  
- **Authors:** Sonika Johri, Shantanu Debnath, Avinash Mocherla, Alexandros Singh, Anupam Prakash, Jungsang Kim, Iordanis Kerenidis  
- **DOI:** [https://doi.org/10.1038/s41534-021-00456-5](https://doi.org/10.1038/s41534-021-00456-5)  
- **Original repository (if any):** QC Ware’s *Forge* platform (not publicly released)  
- **License and attribution notes:** Cite this paper when reproducing or adapting the algorithms. All rights belong to the authors and Nature Publishing Group.

---

## Overview

This paper introduces and experimentally demonstrates a **Quantum Nearest Centroid (QNC)** classifier on an **11-qubit trapped-ion quantum computer** from IonQ. The study combines efficient **quantum data loading**, **distance estimation circuits**, and **error mitigation** to perform supervised classification tasks with accuracy comparable to classical methods.

### Reproduction Scope

- Implemented QNC classifier (data loaders, distance estimation circuits)  
- Datasets: synthetic (4D/8D), MNIST (PCA reduced to 8D), IRIS (4D)  
- Performance comparison with classical Nearest Centroid algorithm  
- Error mitigation and noise modeling  

### Deviations/Assumptions for Reproduction

- Simulated environment (IonQ hardware not required)  
- Ideal noise-free or Qiskit-based simulation backends  
- Classical preprocessing (PCA, centroids) performed on CPU  

### Hardware/Software Environment

- **Quantum backend:** IonQ trapped-ion system (171Yb⁺ qubits)  
- **Classical libraries:** Python 3.9+, NumPy, scikit-learn, matplotlib  
- **Optional:** QC Ware Forge or Qiskit backend for simulation  
- **Qubits:** 11 physical, circuits up to 8 logical qubits  
- **Native gates:** Molmer–Sorensen two-qubit entangling gates, RBS(θ) rotations  

---

## How to Run

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Command-line interface

Main entry point: `implementation.py`

```bash
python implementation.py --help
```

Supported options:

- `--config PATH` Load config from JSON (example files in `configs/`).
- `--seed INT`    Random seed for reproducibility.
- `--outdir DIR`  Output base directory (default: `outdir`). A timestamped run folder `run_YYYYMMDD-HHMMSS` is created inside.

Example reproduction specific options:
- `--device STR`  Device string (e.g., cpu, cuda:0, mps).
- `--epochs INT`  Number of training epochs.
- `--batch-size INT` Batch size.
- `--lr FLOAT`    Learning rate.

Example runs:

```bash
# From a JSON config
python implementation.py --config configs/example.json


```

The script saves a snapshot of the resolved config alongside results and logs.


## Configuration

Configuration files are in `configs/` directory.

- `example.json` shows the structure and defaults.
- `mnist_config.json` runs the full mnist experiment
- `iris_config.json` runs the full iris experiment

## Results and Analysis

- The results are stored in the runs folder directories
- This paper implements a way to encode data differently, here allowing to define a quantum inner product. This encoding method opens the door for new researches that are yet to be explored. 