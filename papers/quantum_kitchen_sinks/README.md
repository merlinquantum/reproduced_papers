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
- Real Rigetti QPU results on 1- and 2-qubit circuits.

## Related reproductions in this repository

[`papers/fock_state_expressivity/q_random_kitchen_sinks/`](../fock_state_expressivity/q_random_kitchen_sinks/)
reproduces **Algorithm 3 ("Quantum-enhanced random kitchen sinks") of Gan et
al. 2022** ([arXiv:2107.05224](https://arxiv.org/abs/2107.05224)).  That work
takes the QKS idea and pushes it into a Fock-state photonic regime on the
moons dataset.  Our reproduction here covers the *original* Wilson et al.
2019 gate-model formulation and adds an independent photonic adaptation on
picture frames and (3,5)-MNIST.  The two reproductions are complementary.

## Reproduction Scope (including Updates and Deviations)

This reproduction implements the QKS algorithm in **NumPy** (a small custom
batched statevector simulator for the gate-model circuits, since the paper's
circuits are tiny and fixed-depth) and adds a **photonic MerLin** adaptation
on top of the open-loop QKS recipe, evaluated on both the synthetic
picture-frames dataset and the (3,5)-MNIST subset.

What is reproduced:

- The 1, 2, and 4-qubit ansätze from Fig. 2 (a, b) and Fig. 6 of the appendix.
- The picture-frames synthetic dataset (Fig. 3), with σ and E sweeps over 3 seeds.
- The (3,5)-MNIST subset (Fig. 5), with 1q / 2q / 4q QKS over 3 seeds.
- Fair classical baselines: logistic regression (paper's LB-rule reference)
  and SVM-RBF (paper's non-linear reference).
- A **photonic MerLin adaptation** of the QKS recipe, run on **both**
  picture-frames and (3,5)-MNIST.

Deviations and notes:

- **Simulator.** We use a small batched-NumPy statevector simulator rather
  than the Rigetti QVM.  Single-shot sampling exactly matches the paper.
- **Dataset sizes.** The picture-frames dataset is regenerated from the paper
  description.  For (3,5)-MNIST we use a 4 000-train / 1 000-test subset to
  keep CPU wall-clock manageable.
- **QPU results.** Not reproduced.  No Rigetti QPU access.

## Project Layout

```text
papers/quantum_kitchen_sinks/
|-- README.md, LOG.md, INSIGHTS.md, FEEDBACK.md, CONFLUENCE.md, VISITED_URLS.md
|-- cli.json, requirements.txt
|-- configs/                       # 11 named experiment configs
|-- lib/
|   |-- data.py, encoding.py, circuits.py, qks_model.py
|   |-- photonic_qks.py            # MerLin photonic adaptation
|   |-- classifiers.py, runner.py
|-- tests/                         # 6 unit + smoke tests
|-- utils/                         # plotting scripts
|-- outdir/                        # timestamped run artifacts
`-- results/                       # curated figures
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

Photonic MerLin QKS on picture frames (≈ 3 minutes total, 3 seeds):

```bash
python implementation.py --paper quantum_kitchen_sinks \
    --config configs/picture_frames_merlin.json
```

Photonic MerLin QKS on (3,5)-MNIST (≈ 15 minutes per seed):

```bash
python implementation.py --paper quantum_kitchen_sinks \
    --config configs/mnist35_merlin.json
```

## Configuration

The CLI is described by ``cli.json``.  Key knobs (see ``cli.json`` for full
schema):

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

## Data

- **Picture frames** — generated synthetically.  Two square frames at
  ``inner_radius = 0.4`` and ``outer_radius = 0.7`` with light Gaussian noise.
- **(3,5)-MNIST** — downloaded via torchvision into
  ``data/quantum_kitchen_sinks/MNIST_raw_cache/``.

## Results Obtained and Comparison with the Paper

All numbers below are mean ± std over 3 seeds where noted.

### Picture frames (Fig. 3)

| Method | Paper value | Reproduced value | Seeds | Label |
|--------|------------:|-----------------:|------:|-------|
| LR baseline (no QKS) | ≈ 50% | 49.25% | 3 | paper-accurate |
| QKS-CNOT2 (best σ, E) | > 99.9% | **100.0 ± 0.0%** (σ=4, E=500) | 3 | paper-accurate |
| QKS-CNOT2 (σ=1, E=5000) | > 99.9% | 99.17 ± 0.12% | 3 | paper-accurate |
| QKS-CZ2 (best σ, E)   | ≈ 50% ("no discrimination") | 98.50 ± 0.45% (σ=2, E=5000) | 3 | partial / deviation |
| **Photonic MerLin QKS (σ=3, E=2000, 4 modes, 2 photons)** | n/a | **99.67 ± 0.47%** | 3 | paper-accurate (photonic adaptation) |

The CZ "no discrimination" claim does not reproduce in our finite-``E``
experiments; see `INSIGHTS.md` for a detailed discussion.

### (3,5)-MNIST (Fig. 5, gate-model)

Test errors (1 − test accuracy):

| Method | Paper value | Reproduced value | Seeds | Label |
|--------|------------:|-----------------:|------:|-------|
| LR baseline | 4.1% | **3.80%** | 1 | paper-accurate |
| SVM-RBF (reference) | ≈ 3.0% | 0.90% | 1 | substitute hyperparameters |
| QKS-1q (σ=0.05, E=5000) | 3.3% | **1.87 ± 0.09%** | 3 | reduced (E=5000 vs 10 000) |
| QKS-2q (σ=0.10, E=5000) | ≈ 1.8% | **1.77 ± 0.24%** | 3 | reduced |
| QKS-4q (σ=0.10, E=5000) | 1.4% (best) | 2.40 ± 0.21% | 3 | episode-budget-limited |

### (3,5)-MNIST — Photonic MerLin QKS (new)

Test errors (1 − test accuracy):

| Variant | Setting | Test error (mean ± std, 3 seeds) |
|---------|---------|---------------------------------:|
| Photonic QKS, ``m=4 / k=2 / E=2000`` | UNBUNCHED, tile encoding, σ=0.05 | 7.80 ± 0.08% |
| Photonic QKS, ``m=6 / k=3 / E=10000`` | DUAL_RAIL, tile encoding, σ=0.05 | 3.83 ± 0.50% |
| **Photonic QKS, ``m=6 / k=3 / E=10000``** | **DUAL_RAIL, tile encoding, σ=0.07** | **3.60 ± 0.42%** |
| Photonic QKS, ``m=8 / k=4 / E=5000`` | DUAL_RAIL, tile encoding, σ=0.05 | 5.43 ± 0.45% |
| Photonic QKS, ``m=8 / k=4 / E=5000`` | DUAL_RAIL, tile encoding, σ=0.07 | 5.40 ± 0.22% |
| LR baseline (raw pixels, for reference) | n/a | 3.80% |
| Gate-model QKS-1q, ``E=5000`` | tile encoding, σ=0.05 | 1.87 ± 0.09% |

The small photonic setting (`m=4, k=2`) remains clearly below the gate-model
QKS and above the LR baseline. In the enlarged settings, constraining the
photonic model to the logical ``DUAL_RAIL`` subspace helps substantially, and
the ``m=6, k=3`` geometry benefits strongly from more episodes and a slightly
larger sigma. Our best photonic MNIST result is now **3.60 ± 0.42%** with
``m=6, k=3, E=10000, σ=0.07`` in ``DUAL_RAIL``, which is effectively on par
with the LR baseline and much closer to the gate-based regime than the compact
UNBUNCHED setting.

### Hardware-aware reporting (MerLin photonic adaptation)

| Field | Picture frames value | Best MNIST photonic value |
|-------|----------------------|---------------------------|
| Computation space | UNBUNCHED | DUAL_RAIL |
| Detector model | threshold | threshold |
| Photon number | 2 | 3 |
| Number of modes | 4 | 6 |
| Input state | ``[1, 1, 0, 0]`` | ``[1, 0, 1, 0, 1, 0]`` |
| Encoding | linear angle, modes 0–1, scale = 1.0 | linear angle (tile), modes 0/2/4, scale = 1.0 |
| Measurement strategy | ``MeasurementStrategy.probs(computation_space=UNBUNCHED)`` + single-shot sampling | ``MeasurementStrategy.probs(computation_space=DUAL_RAIL)`` + single-shot sampling |
| Postselection | none | none |
| Simulator | MerLin CPU simulator (analytic) | same |
| Shot count | 1 / episode | 1 / episode |
| Seeds | [42, 43, 44] | [42, 43, 44] |

## Fair Baselines

The paper requires that the *only* non-linearity comes from the quantum
circuit (LB rule).  Our fair classical baseline is therefore plain
``LogisticRegression`` on the raw inputs.  We also report an unfair (non-linear)
SVM-RBF baseline as a Fig. 5 reference.

## MerLin Photonic Extension

`lib/photonic_qks.py` implements per-episode `ml.QuantumLayer`s with frozen
entangling-mesh phases, data driving an `add_angle_encoding`, and single-shot
sampling.  The photonic adaptation reproduces the central QKS claim on
picture frames. On (3,5)-MNIST, the original UNBUNCHED setting remains weak,
but enlarged DUAL_RAIL runs close much of the gap: the best current photonic
result is ``m=6, k=3, E=10000, σ=0.07`` with **3.60 ± 0.42%** test error.

## Limitations

- No QPU experiments. Entirely simulated.
- (3,5)-MNIST uses a 4 000-train / 1 000-test subset.
- CZ ansatz reproduction does **not** confirm the paper's "no-better-than-random" claim — see the dedicated note in `INSIGHTS.md`.
- Photonic adaptation on MNIST still trails the best gate-model QKS result,
  but DUAL_RAIL plus larger ``n_modes``/``n_photons`` closes much of the gap.
- As with classical Random Kitchen Sinks, the benefit is not universal: it
  likely depends strongly on the dataset and on how well the chosen feature map
  matches the underlying structure.

## Tests

```bash
cd papers/quantum_kitchen_sinks && pytest -q
```

## Citation and License

```
@article{wilson2019qks,
  title   = {Quantum Kitchen Sinks: An algorithm for machine learning on near-term quantum computers},
  author  = {Wilson, C. M. and Otterbach, J. S. and Tezak, N. and Smith, R. S. and Polloreno, A. M. and Karalekas, P. J. and Heidel, S. and Alam, M. S. and Crooks, G. E. and da Silva, M. P.},
  journal = {arXiv:1806.08321},
  year    = {2019}
}
```

This reproduction is released under the same license as the rest of the
repository (MIT, see the repository-root ``LICENSE``).
