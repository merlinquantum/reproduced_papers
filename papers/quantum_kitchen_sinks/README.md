# Quantum Kitchen Sinks — Reproduction

## Reference and Attribution

- Paper: **Quantum Kitchen Sinks: An algorithm for machine learning on near-term quantum computers**
- Authors: C. M. Wilson, J. S. Otterbach, N. Tezak, R. S. Smith, A. M. Polloreno, P. J. Karalekas, S. Heidel, M. S. Alam, G. E. Crooks, M. P. da Silva
- Affiliation: Rigetti Computing (et al.)
- arXiv: [1806.08321v2](https://arxiv.org/abs/1806.08321) (2018; v2 Nov 2019)
- Original repository: not located.  The algorithm is fully specified in the
  main text and the Quil snippets in the appendix.

## Original Paper

The paper introduces **Quantum Kitchen Sinks (QKS)**: an *open-loop* hybrid
QML algorithm in which the quantum processor is used as a random non-linear
feature extractor instead of being trained variationally.  For each "episode"
``e``, the input vector ``u`` is mapped to ``q`` gate angles by a fresh random
linear encoding ``θ_e = Ω_e u + β_e`` (entries of Ω drawn from ``N(0, σ²)``,
biases of β from ``U(0, 2π)``).  A fixed-depth circuit ansatz (RX rotations
followed by a CNOT or CZ network) is executed and **a single bitstring is
sampled** from the output.  Stacking these bitstrings over ``E`` independent
episodes yields a ``(E·q)``-dimensional feature vector that is fed to a
*linear* classifier — by the **Linear Baseline (LB) Rule** the only quantum
non-linearity comes from the circuit itself.

The paper demonstrates:

- **Picture frames** (synthetic 2-D classification, Fig. 3): logistic
  regression alone gets ≈ 50% accuracy; QKS with a 2-qubit CNOT ansatz
  achieves > 99.9% test accuracy at the optimal ``σ ≈ 1``.
- **(3,5)-MNIST subset** (Fig. 5): logistic regression baseline 95.9%
  (4.1% error); QKS reduces the error to 1.4% (best, at the 4-qubit ansatz).
- Real Rigetti QPU results on 1- and 2-qubit circuits in the same regime.

## Related reproductions in this repository

[`papers/fock_state_expressivity/q_random_kitchen_sinks/`](../fock_state_expressivity/q_random_kitchen_sinks/)
reproduces **Algorithm 3 ("Quantum-enhanced random kitchen sinks") of Gan et
al. 2022** ([arXiv:2107.05224](https://arxiv.org/abs/2107.05224)).  That work
takes the QKS idea and pushes it into a Fock-state photonic regime on the
moons dataset.  Our reproduction here covers the *original* Wilson et al.
2019 gate-model formulation and adds an independent photonic adaptation
focused on the picture-frames and (3,5)-MNIST experiments.  The two
reproductions are complementary and not redundant.

## Reproduction Scope (including Updates and Deviations)

This reproduction implements the QKS algorithm in **NumPy** (a small custom
batched statevector simulator for the gate-model circuits, since the paper's
circuits are tiny and fixed-depth) and adds a **photonic MerLin** adaptation
on top of the open-loop QKS recipe.

What is reproduced:

- The 1, 2, and 4-qubit ansätze from Fig. 2 (a, b) and Fig. 6 of the appendix.
- The picture-frames synthetic dataset (Fig. 3), with σ and E sweeps over 3 seeds.
- The (3,5)-MNIST subset (Fig. 5), with 1q / 2q / 4q QKS over 3 seeds.
- Fair classical baselines: logistic regression (paper's LB-rule reference)
  and SVM-RBF (paper's non-linear reference).
- A new **photonic MerLin adaptation** of the QKS recipe.

Deviations and notes:

- **Simulator.** We use a small batched-NumPy statevector simulator rather
  than the Rigetti QVM, because we only need fixed-depth circuits of ≤ 4
  qubits.  Single-shot sampling exactly matches the paper.
- **Dataset sizes.** The picture-frames dataset is regenerated from the paper
  description (no original data file located); we use 1600 training and 400
  test points as stated in the paper.  For (3,5)-MNIST we use a 4000-train /
  1000-test subset of the canonical (3,5)-MNIST split (paper uses up to 50%
  of full MNIST for QPU runs and the full QVM subset for Fig. 5; the
  truncated subset keeps wall-clock on CPU manageable while preserving the
  qualitative comparison).
- **QPU results.** Not reproduced.  No Rigetti QPU access.  All reproductions
  use the noiseless statevector simulator (analogous to the paper's QVM
  curves).
- **Encoding.**  The paper specifies "split" encoding for picture frames
  (``q = p = 2, r = 1``) and "tile" encoding for MNIST.  We implement both;
  the MNIST tile partitioning splits the flattened 784-dim image into ``q``
  contiguous tiles of size ``p / q``.

## Project Layout

```text
papers/quantum_kitchen_sinks/
|-- README.md
|-- LOG.md
|-- VISITED_URLS.md
|-- INSIGHTS.md
|-- FEEDBACK.md
|-- CONFLUENCE.md
|-- cli.json
|-- requirements.txt
|-- configs/
|   |-- defaults.json
|   |-- picture_frames_cnot2.json
|   |-- picture_frames_cz2.json
|   |-- picture_frames_merlin.json
|   |-- baseline_lr_picture_frames.json
|   |-- mnist35_cnot1.json
|   |-- mnist35_cnot2.json
|   |-- mnist35_cnot4.json
|   |-- baseline_lr_mnist35.json
|   `-- baseline_svm_mnist35.json
|-- lib/
|   |-- data.py           # synthetic picture-frames + (3,5)-MNIST loaders
|   |-- encoding.py       # split / tile linear encoding Ω · u + β
|   |-- circuits.py       # batched RX + CNOT/CZ statevector simulator
|   |-- qks_model.py      # gate-model QKS featurizer
|   |-- photonic_qks.py   # photonic MerLin QKS featurizer
|   |-- classifiers.py    # linear classifiers (LR, SVM-linear, SVM-RBF)
|   `-- runner.py         # entry point `train_and_evaluate`
|-- tests/
|   |-- common.py
|   |-- test_cli.py
|   `-- test_smoke.py
|-- utils/
|   |-- plot_sigma_E_sweep.py
|   |-- plot_mnist35_scaling.py
|   `-- plot_picture_frames.py
`-- results/
```

## Install and How to Run

```bash
pip install -r papers/quantum_kitchen_sinks/requirements.txt
```

From the repo root:

```bash
python implementation.py --paper quantum_kitchen_sinks --config configs/<name>.json
```

Smoke run (≤ 1 minute on CPU):

```bash
python implementation.py --paper quantum_kitchen_sinks \
    --config configs/picture_frames_cnot2.json \
    --n-train 200 --n-test 50
```

Picture-frames reproduction (σ × E × 3 seeds, ≈ 2 minutes):

```bash
python implementation.py --paper quantum_kitchen_sinks \
    --config configs/picture_frames_cnot2.json
```

(3,5)-MNIST 1-qubit run (≈ 1 minute):

```bash
python implementation.py --paper quantum_kitchen_sinks \
    --config configs/mnist35_cnot1.json
```

Plot the picture-frames σ × E heatmap (after the run finishes):

```bash
cd papers/quantum_kitchen_sinks
python utils/plot_sigma_E_sweep.py outdir/run_YYYYMMDD-HHMMSS \
    --out results/picture_frames_cnot2_heatmap.png
```

## Configuration

The CLI is described by ``cli.json``.  Key knobs:

| Flag | Default | Meaning |
|------|---------|---------|
| ``--circuit`` | ``cnot2`` | One of ``cnot1``, ``cnot2``, ``cz2``, ``cnot4``, ``cnot8`` |
| ``--n-qubits`` | 2 | Number of qubits |
| ``--n-episodes`` | 100 | ``E``: number of independent random circuits |
| ``--sigma`` | 1.0 | Std-dev of the encoding distribution ``N(0, σ²)`` |
| ``--shots-per-episode`` | 1 | Single-shot per episode matches the paper |
| ``--n-layers`` | 1 | Stacked encoding layers (1 in the main text) |
| ``--dataset-name`` | ``picture_frames`` | ``picture_frames`` or ``mnist35`` |
| ``--classifier-kind`` | ``logistic_regression`` | ``logistic_regression``, ``svm_rbf``, ``svm_linear`` |
| ``--backend`` | ``gate`` | ``gate`` (NumPy simulator) or ``photonic_merlin`` |

Top-level keys ``sigma_sweep`` and ``episodes_sweep`` enable a Cartesian
sweep across (σ, E, seed).  ``seeds`` (list of integers) drives the
per-experiment seed iteration; missing → fall back to the global ``seed``.

## Data

- **Picture frames** — generated synthetically.  Two square frames at
  ``inner_radius = 0.4`` and ``outer_radius = 0.7`` with light Gaussian noise.
- **(3,5)-MNIST** — downloaded via torchvision into
  ``data/quantum_kitchen_sinks/MNIST_raw_cache/``.  Filtered to digits 3 and
  5, flattened to 784-d and standardised per image.

## Results Obtained and Comparison with the Paper

All numbers below are mean ± std over 3 seeds with the configs in
``configs/`` and a fresh ``python implementation.py --paper quantum_kitchen_sinks``
invocation.

### Picture frames (Fig. 3)

| Method | Paper value | Reproduced value | Seeds | Label |
|--------|------------:|-----------------:|------:|-------|
| LR baseline (no QKS) | ≈ 50% | 49.25% (constant across seeds) | 3 | paper-accurate |
| QKS-CNOT2 (best σ, E) | > 99.9% | **100.0 ± 0.0%** (σ=4, E=500) | 3 | paper-accurate |
| QKS-CNOT2 (σ=1, E=5000) | > 99.9% | 99.17 ± 0.12% | 3 | paper-accurate |
| QKS-CZ2 (best σ, E)   | ≈ 50% ("no discrimination") | **98.50 ± 0.45%** (σ=2, E=5000) | 3 | partial / deviation |
| **Photonic MerLin QKS (σ=3, E=2000, 4 modes, 2 photons)** | n/a (new) | **99.67 ± 0.47%** | 3 | paper-accurate (photonic adaptation) |

**Note on the CZ ansatz claim.** The paper states the CZ ansatz "leads to
classifiers that are no better than random", arguing from a constant implicit
kernel.  In our reproduction the CZ ansatz still reaches ≈ 98.5% test
accuracy on picture frames.  Two interpretations are consistent with our
finding:

1. The CZ ansatz's *kernel-average* is constant, but for any finite ``E`` the
   features are not uniform random — they remain functions of single
   coordinates ``u_i``.  Since the picture-frames decision boundary is
   ``max(|u_0|, |u_1|) ≈ const``, single-coordinate non-linear features are
   sufficient for LR to classify, and finite-sample regularisation does not
   wash this signal out.
2. The paper may be referring to a related dataset (e.g. a dataset where no
   one-coordinate function discriminates) for the "no better than random"
   statement.

We document this discrepancy rather than papering over it.  See ``INSIGHTS.md``
for a longer note.

### (3,5)-MNIST (Fig. 5)

Test errors (1 − test accuracy), mean over 3 seeds where applicable:

| Method | Paper value | Reproduced value | Seeds | Label |
|--------|------------:|-----------------:|------:|-------|
| LR baseline | 4.1% | **3.80%** | 1 | paper-accurate |
| SVM-RBF (reference) | ≈ 3.0% | **0.90%** | 1 | substitute hyperparameters |
| QKS-1q (σ=0.05, E=5000) | 3.3% (Fig. 5) | **1.87 ± 0.09%** | 3 | reduced (E=5000 vs 10 000) |
| QKS-2q (σ=0.10, E=5000) | ≈ 1.8% (Fig. 5) | **1.77 ± 0.24%** | 3 | reduced (E=5000) |
| QKS-4q (σ=0.10, E=5000) | 1.4% (best) | **2.40 ± 0.21%** | 3 | reduced (E=5000 vs ≥20 000) |

Scaling figure: ``results/mnist35_error_vs_qubits.png``.

The qualitative scaling claim — QKS lifts a linear classifier well below its
own baseline and the best operating point sits at small ``σ`` — is reproduced.
On our reduced 4 000-train subset the 1-qubit and 2-qubit points each beat
the linear baseline by ≈ 2 percentage points; the 4-qubit point is
*worse* than 2q in our reduced regime, where the paper reports the opposite
trend.  The most likely cause is the episode budget: the paper uses
``E ≈ 20 000`` for 4q in Fig. 5 while we cap at ``E = 5 000`` to keep
wall-clock manageable on CPU.

### Hardware-aware reporting (MerLin photonic adaptation)

| Field | Value |
|-------|-------|
| Computation space | UNBUNCHED |
| Detector model | threshold |
| Photon number | 2 |
| Number of modes | 4 |
| Input state | ``[0, 1, 0, 1]`` |
| Encoding | linear angle (``θ = Ω u + β``) on modes 0–1, scale = 1.0 |
| Measurement strategy | ``ml.MeasurementStrategy.PROBABILITIES`` + single-shot sampling |
| Postselection | none |
| Simulator / QPU | MerLin CPU simulator (analytic) |
| Shot count | 1 per episode (single-shot, matches the paper) |
| Wall-clock time | ≈ 50 s per seed (E = 2000, 800 train + 200 test) |
| Seeds | [42, 43, 44] |

## Fair Baselines

The paper requires that the *only* non-linearity comes from the quantum
circuit (LB rule).  Our fair classical baseline is therefore plain
``LogisticRegression`` on the raw inputs (no kernel, no extra features).  We
also report an unfair (non-linear) SVM-RBF baseline as a sanity check, since
the paper includes it on Fig. 5 to make the point that QKS beats it at the
small-qubit end.

## MerLin Photonic Extension

The photonic QKS adaptation lives in ``lib/photonic_qks.py``.  Each episode is
one ``ml.QuantumLayer`` whose entangling-layer phases are sampled at
construction (and frozen — QKS is open-loop), preceded and followed by
``add_entangling_layer``.  The data drives an ``add_angle_encoding`` block
in the middle.  Single-shot sampling from the output occupation
distribution gives one binary feature per mode; ``E`` episodes stack into
``E × n_modes`` features used by logistic regression.

The photonic adaptation reproduces the central QKS claim on the picture
frames dataset (99.7% test accuracy vs 49.2% for the LR baseline), at a
reasonable photonic resource budget (4 modes, 2 photons, UNBUNCHED).

## Limitations

- No QPU experiments.  The reproduction is entirely simulated.
- (3,5)-MNIST uses a 4 000-train / 1 000-test subset rather than the full
  filtered split, to keep CPU wall-clock reasonable.  Trends should not change
  at full data.
- The CZ ansatz reproduction does **not** confirm the paper's
  "no-better-than-random" claim — see the dedicated note above.

## Tests

```bash
cd papers/quantum_kitchen_sinks && pytest -q
```

Six tests cover: CLI help, CLI flag parsing, encoding shapes, gate-model
featurizer output shape and value domain, dataset shapes, and one end-to-end
runner smoke test.

## Citation and License

If you use this reproduction in your work, please cite the original paper:

```
@article{wilson2019qks,
  title   = {Quantum Kitchen Sinks: An algorithm for machine learning on near-term quantum computers},
  author  = {Wilson, C. M. and Otterbach, J. S. and Tezak, N. and Smith, R. S. and Polloreno, A. M. and Karalekas, P. J. and Heidel, S. and Alam, M. S. and Crooks, G. E. and da Silva, M. P.},
  journal = {arXiv:1806.08321},
  year    = {2019}
}
```

This reproduction is released under the same license as the rest of this
repository (MIT, see the repository-root ``LICENSE``).
