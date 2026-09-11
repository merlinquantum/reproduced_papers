# BVE-QNN — Photonic Dual-Rail Reproduction (Experiment 1)

## Reference and Attribution

- Paper: *Potential of quantum scientific machine learning applied to weather modelling* (Phys. Rev. A 110, 052423, 2024)
- Authors: Ben Jaderberg, Antonio A. Gentile, Atiyo Ghosh, Vincent E. Elfving (PASQAL), Caitlin Jones, Davide Vodola, John Manobianco, Horst Weiss (BASF)
- DOI/ArXiv: [arXiv:2404.08737](https://arxiv.org/abs/2404.08737), [10.1103/PhysRevA.110.052423](https://doi.org/10.1103/PhysRevA.110.052423)
- Original repository (if any): not public at the time of writing
- Related personal work (neutral-atom side + dataset notebooks): [CyrilDeloince/qml_PINN_pasqal](https://github.com/CyrilDeloince/qml_PINN_pasqal)
- License and attribution notes: results and figures are our own reproduction; please cite the original paper when using this code.

## Overview

The paper introduces the barotropic vorticity equation (BVE) as a model of the atmosphere and trains parameterised quantum circuits (PQCs) both (a) directly on real weather data (Experiment 1, Section V.A) and (b) as physics-informed solvers of the BVE PDE (Experiment 2, Section V.B).

This folder reproduces **Experiment 1 with MerLin**: a photonic dual-rail QNN trained to regress the stream function $\psi(t, x, y, z)$ against a reference Spectral Element Method (SEM) solution, at 4° global resolution.

The original paper's **neutral-atom** QNN is built from:
- a serial trainable-frequency feature map ($R_y$ with trainable frequencies $\gamma_{r,m}$) over $N = 6$ logical qubits,
- a Hardware-Efficient Ansatz (HEA) with $l = 32$ layers and native CNOT entangling gates,
- a total-magnetisation observable $C = \sum_m Z_m$,
- a learnable affine output map $\psi = \alpha_{\text{scale}} \cdot \text{QNN}(\cdot) + \alpha_{\text{shift}}$.

**What lives in this package:**
- **Photonic (MerLin):** dual-rail implementation, shared-runtime CLI (`lib/runner.py`), committed checkpoint, metrics, notebook, and figure utility.
- **Neutral-atom (Qadence):** the faithful reproduction of the original paper's QNN architecture (median MRE 9.15%, PPMCC 0.873) and the dataset-generation notebooks, under `notebooks/neutral_atom/`.

### Photonic adaptations

1. **Dual-rail sum-Z readout.** Each logical qubit is one photon on two modes ($|0\rangle = |1,0\rangle$, $|1\rangle = |0,1\rangle$). Then $Z_m \equiv n_{\text{left},m} - n_{\text{right},m}$, so $\sum_m Z_m$ becomes $\sum_m (\langle n_{\text{left},m}\rangle - \langle n_{\text{right},m}\rangle)$.
2. **Same learnable output scaling** as the paper, applied after the photonic observable.
3. **Trainable photonic mixing** instead of fixed CNOTs. Linear optics cannot implement a native CNOT (KLM theorem). Nearest-neighbour beamsplitters replace CNOTs; making them trainable partially compensates for weaker entanglement.

### Why Perceval primitives inside MerLin

MerLin's `QuantumLayer` accepts a Perceval circuit. Training and differentiation stay in MerLin. We build the circuit with raw Perceval primitives (`pcvl.BS`, `pcvl.PS`) rather than MerLin's high-level generic interferometer/`CircuitBuilder`, because the paper's HEA needs a **sparse mode-level topology**: per-qubit dual-rail blocks on modes `(2q, 2q+1)` and nearest-neighbour mixers on `(2q+1, 2(q+1))`. A dense all-to-all MZI mesh is the wrong primitive for that structure.

**Hardware/software:** Colab CPU, `float64`, `merlinquantum==0.4.0`, `perceval-quandela>=1.2.1`, `torch>=2.0`. Full 5000-step training takes several hours on CPU.

## Results and Analysis

| Model | Params | Median MRE | Median PPMCC |
|---|---|---|---|
| Paper (neutral-atom HEA) | 654 | 7.1% – 10.9% | 0.870 |
| Our neutral-atom reproduction ([qml_PINN_pasqal](https://github.com/CyrilDeloince/qml_PINN_pasqal)) | 654 | 9.15% | 0.873 |
| Photonic v1 (fixed mixing) | 654 | 17.6% | 0.723 |
| Photonic v2 (fixed mixing + learnable output) | 654 | 16.98% | 0.718 |
| **Photonic v3 (trainable mixing) — this package** | **1006** | **14.85%** | **0.754** |

The photonic dual-rail model learns real stream-function dynamics (PPMCC 0.754), but does not match the neutral-atom baseline. We treat that as an honest physics finding: beamsplitter mixing is a weaker entangling resource than CNOT. Trainable mixing improves over fixed mixing (~0.72 → 0.75 PPMCC) but cannot fully close the gap.

These results validate MerLin as a viable framework for quantum scientific machine learning, while highlighting that future work on photonic entanglement schemes (e.g., measurement-based feed-forward, non-linear interactions, or alternative encoding strategies beyond dual-rail) could further close the gap.

Artifacts:
- `results/exp1_merlin_results.npz` — predictions, SEM reference, MRE/PPMCC arrays
- `models/qnn_exp1_merlin_dualrail_depth32_step5000.pt` — trained checkpoint (step 5000)
- `notebook.ipynb` — interactive walkthrough + Mollweide figure at $t = 22\text{h}$
- `utils/plot_mollweide.py` — regenerates the comparison figure from saved results

## How to Run

### Install dependencies

```bash
cd papers/bve_qnn
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Command-line interface

```bash
# From inside papers/bve_qnn
python ../../implementation.py --help

# From the repo root
python implementation.py --paper bve_qnn --help
```

Project-specific overrides (see `cli.json`):

- `--total-steps INT` Total training steps (resumable — skipped if checkpoint already reached this step)
- `--batch-size INT` Dataloader batch size
- `--lr FLOAT` Adam learning rate
- `--checkpoint NAME` Checkpoint filename under `models/` (omit to train from scratch)
- `--n-qubits INT` / `--depth INT` Logical qubits / HEA layers

Global flags such as `--config`, `--seed`, `--dtype`, `--device`, `--outdir` come from `runtime_lib/global_cli.json`.

### Example runs

```bash
# Fast smoke run (trains 5 steps from scratch)
python ../../implementation.py --config configs/defaults.json

# Paper-faithful evaluation: loads the committed step-5000 checkpoint
python ../../implementation.py --config configs/example.json

# Force a fresh full 5000-step run from scratch (several hours on CPU)
python ../../implementation.py --config configs/example.json --checkpoint null
```

Each run writes a timestamped folder under `outdir` / `results` with `checkpoint.pt`, `exp1_merlin_results.npz`, `metrics.json`, `config_snapshot.json`, and `done.txt`.

### Data location

Smoke tests and the notebook use a committed subset:

`data/bve_qnn/sem_supervised_subset.npz` (~6 KB, 128 points on a 2×8×8 grid).

The full SEM file `data/bve_qnn/sem_supervised_dataset.npz` (~3.2 MB) is **not** in git. Paper-faithful evaluation (`configs/example.json`) needs it locally. The original paper does not publish the dataset; regenerate it from ERA5 via Copernicus CDS and the SEM solver (Appendix B):

- `notebooks/neutral_atom/quantum_bve_step_by_step.ipynb` — full pipeline
- `utils/generate_dataset.py` — CDS setup instructions
- `utils/make_subset.py` — rebuilds the smoke subset from the full file

Published MRE / PPMCC and the Mollweide figure come from `results/exp1_merlin_results.npz`, which does not require the 3.2 MB file.

### Neutral-atom reproduction (Qadence)

The neutral-atom QNN reproduction uses [Qadence](https://github.com/pasqal-io/qadence) (not MerLin). The notebooks are in `notebooks/neutral_atom/`:

- `quantum_bve_step_by_step.ipynb` — dataset generation from ERA5 reanalysis via Copernicus CDS API
- `running_exp1.ipynb` — Qadence QNN training for Experiment 1 (HEA, 6 qubits, depth 32)

Configuration reference: `configs/neutral-atom.json`

```bash
pip install qadence
```

## Extensions and Next Steps

- Stronger photonic entanglement (measurement feed-forward, non-linear interactions, encodings beyond dual-rail) to close the gap with the neutral-atom baseline
- Explore hybrid spin-optical architectures such as SPOQC as a longer-term path to richer entanglement
- Photonic reproduction of Experiment 2 (physics-informed BVE)
- Ablations of mixing depth / parameter budget

## Reproducibility Notes

- Seed: `42`
- Precision: `float64`
- Optimizer: Adam, `lr=1e-2`, resumable checkpoints every 500 steps during training

## Testing

```bash
cd papers/bve_qnn
pytest -q
```

- `tests/test_smoke.py` — short end-to-end train+eval against the committed dataset
- `tests/test_cli.py` — CLI schema and defaults sanity checks
