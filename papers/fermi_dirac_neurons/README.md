# Fermi-Dirac Machines as Quantizations of Neurons - Reproduction

Status (2026-11-10): reproduction, photonic adaptation and fair comparison
complete (Phases 3–5, all audits passed), run on a local CPU.

**Verdict.**
- **C4 and C6** (Figs. 7 and 9): the paper's direction is reproduced, and
  the quantum neuron still beats a tuned classical neuron on the same
  features. But it wins because the paper's targets lie inside the quantum
  model's own class, not because of a general advantage of quantization.
- **C5** (Fig. 8 / Table II): reversed by the fair baseline. Logistic
  regression on the same Pauli features beats the Heisenberg neuron by
  3–5 points.
- **Theory identities:** they hold, except two misprinted theorems.
- **Photonic version (MerLin, simulation):** the passive-Hamiltonian
  neuron, exact in MerLin, shows the same pattern.

The full record, including every sweep, rule, deviation and audit, is in
[LOG.md](LOG.md); reusable lessons are in [INSIGHTS.md](INSIGHTS.md).

## Reference and Attribution

- Paper: Fermi-Dirac machines as quantizations of neurons (arXiv preprint,
  quant-ph, submitted 2026-05-23; 87 pages, 12 figures, 2 tables)
- Authors: Alexander He, Nana Liu, Mark M. Wilde
- arXiv: https://arxiv.org/abs/2605.24386 (PDF:
  https://arxiv.org/pdf/2605.24386)
- Original repository: https://github.com/Alexheeeee/quantized-neurons
  (ref. [70] of the paper), used at commit `9c2094c`. It has no license
  file (GitHub API `license: null`), so its notebooks and CSVs are not
  redistributed here. This reproduction is an independent reimplementation.
  It reads the released CSVs only as reference values.

## Original Paper

A classical neuron computes φ(wᵀz + b) on spins z ∈ {−1, 1}ⁿ. The paper
reads this as an activation function φ applied to a classical spin
Hamiltonian. It "quantizes" the neuron by replacing the spins with Pauli
operators and allowing any parameterized k-local Hamiltonian
H(θ) = Σ_j θ_j H_j. The model output for an input quantum state ρ is
Tr[φ(H(θ)) ρ]. For φ = tanh(·/T) this is the Fermi-Dirac machine.

- **Theory.** For commuting (Z-only) H, the model reduces exactly to a
  classical neuron averaged over the measurement distribution of ρ. The
  objective and its gradient are written as expectations over classical
  random variables of Hadamard-test quantities. This gives unbiased hybrid
  estimators (Algorithms 1, 2, 8-15) and single-shot "firing" algorithms
  using a control qumode (Algorithms 3-7). Further results: quantizations of
  softplus, SiLU, erf, smoothed ReLU and GeLU, and a proof that deciding
  Tr[g_T(H)ρ] ≥ α is BQP-complete (Theorem 11).
- **Experiments.** Synthetic, 2-7 qubits. The targets are random TFIM or
  Heisenberg Hamiltonians.
  - Fig. 7: squared-loss regression, tanh and softplus, at 2 and 7 qubits.
  - Fig. 8 and Table II: logistic-loss classification of Haar-random states,
    n = 2…7.
  - Fig. 9: nonlinear tanh model vs linear model Tr[H(θ)ρ].
- **Main claims.** Non-commuting Hamiltonian neurons (TFIM, Heisenberg)
  reach lower loss and higher accuracy than commuting Ising neurons with
  matched structure (C4, C5). The nonlinearity helps, except where the
  operator algebra "closes" for 2-qubit Heisenberg (C6). The advantage
  needs quantum, non-orthogonal input data (C8).

## Reproduction Scope, Claims, and Deviations

Reproduced:

- **E0 theory checks.**
  - Classical-reduction identity.
  - Integral formulas: Theorems 1, 2, 5, 6, 8, 13 and Corollary 1.
  - Monte Carlo of Algorithms 1, 2, 4, 5, 7.
  - Convolution identities: Eqs. 41, 73, 90, 94.
- **Fig. 7** (tanh and softplus, 2 and 7 qubits), **Fig. 8 with Table II**
  (n = 2…7), and **Fig. 9** (TFIM and Heisenberg, 2 and 3 qubits).
  - 10 seeds each (0-9). The paper reports one unseeded run.
  - Two protocols: `_original` follows the authors' notebooks, which
    produced the released figures. `_papertext` follows the protocol as the
    paper's text describes it.
- **Fair baselines**, added because the paper compares only against
  commuting models:
  - B2: a classical neuron on the same Pauli-expectation features.
  - B3: the quantum linear model.
- **Paper-claim extensions.**
  - E8: classical vs quantum training data (C8).
  - E9: robustness across 3 learning rates, 4 temperatures and finite-shot
    gradients.
- **User-requested extensions** (not in the paper):
  - E10, E11: barren plateaus.
  - E12: the contextuality of φ(H) vs temperature.
  - E15, E16: contextual vs noncontextual term sets at equal parameter count.

Updates and deviations:

- **Gradients.** Exact gradients from the Daleckii-Krein formula, as in the
  authors' notebooks. The paper's sampled estimators are simulated only in
  E0c and E9b (finite shots).
- **Settings resolved from the code.** Where the paper text and the code
  disagree, both versions were run. These cover the softplus target, label
  and prediction rules, training-state sets, the logistic-loss scale, and
  the 6-qubit epochs (`LOG.md`, D1-D20).
- **Fig. 9 settings.** No code was released for Fig. 9. Its settings are
  inferred from Fig. 7, with two sensitivity variants: η = 0.3, and target
  coefficients ζ ~ U[−1, 1].
- **Printed theorems.** Theorems 8 and 13 contain printed errors (D14, D16).
  A corrected Theorem 13 is checked.
- **Reduced or partial runs.**
  - E10 stops at n = 11. Both n = 12 cells failed twice in the macOS
    Accelerate eigensolver, and a retry was not authorized.
  - E11 covers n = 2…8. n = 9 was dropped by user decision (≈10 h).
- **Hardware and precision.** CPU only, float64. No QPU and no GPU.

Not reproduced in Phase 3:

- Theorem 12, Corollary 2, Theorems 14 and 15 (`# TODO`).
- Theorem 11 (complexity; not testable numerically).
- Sections V and VII (CV neurons, networks; the paper gives no numerics).
  Phase 4 implements the passive (number-conserving) case of the Sec. V
  neuron photonically.

Not run:

- **E14** ("Can a hybrid network solve a task that no noncontextual model
  can solve?", user-requested). It was scheduled for Phase 5 but left out of
  the Phase 5 plan; the user then chose to move on to documentation.
- A teacher outside each model's class with tuned learning rates. This
  would test the confounder behind C4 and C6.
- The two 12-qubit E10 cells. They failed twice, and a retry was not
  authorized.

Claims and outcomes (details in the sections below and in `LOG.md`):

| Claim | Statement (short) | Outcome |
|---|---|---|
| C1 | Commuting H reduces exactly to a classical neuron | Supported (error ≤ 8.9e-16) |
| C2 | Objective and gradient as classical expectations; unbiased estimators | Partially supported (Theorems 8 and 13 misprinted; Theorem 12, Corollary 2, Theorems 14, 15 not checked) |
| C3 | Single-shot algorithms have the right mean | Supported, with the stated T conventions |
| C4 | Non-commuting neuron beats commuting neuron (Fig. 7) | Reproduced against the paper's baseline; supported against tuned B2, confounded (teacher inside the model class) |
| C5 | Heisenberg neuron beats FCIM (Fig. 8, Table II) | Reproduced against FCIM; **unsupported** against tuned B2 (B2 better by 3–5 points) |
| C6 | Nonlinear beats linear; 2-qubit Heisenberg exception (Fig. 9) | Ordering reproduced and supported against tuned B2 and B3 (confounded); absolute values and the exception not reproduced |
| C7 | BQP-completeness | Not testable numerically |
| C8 | Advantage needs quantum data | Supported (E8, and photonic P4-L) |
| C9 | CV neurons, networks | Out of scope (passive CV case implemented photonically in Phase 4) |

## Project Layout

```text
papers/fermi_dirac_neurons/
|-- README.md, LOG.md, VISITED_URLS.md, requirements.txt, notebook.ipynb
|-- cli.json                  # paper-specific CLI flags
|-- configs/                  # defaults.json (smoke), 396 generated configs, sweeps/*.json
|-- lib/runner.py             # train_and_evaluate: tasks train, theory_checks,
|                             #   gradient_variance, contextuality_scan, pauli_compressibility
|-- lib/pauli.py              # term sets: TFIM, IM, Heisenberg, FCIM, tfim_global, ising_x0, explicit
|-- lib/activations.py        # activations, matrix functions, divided differences
|-- lib/models.py             # HamiltonianNeuron (paper), PauliFeatureNeuron (B2, B3)
|-- lib/data.py               # synthetic states and targets (Sec. VI A)
|-- lib/training.py           # full-batch gradient descent (Eq. 119), finite-shot option
|-- lib/theory_checks.py      # E0
|-- lib/trainability.py       # E10 gradient-variance cells
|-- lib/contextuality.py      # Kirby-Love test, fast Pauli transform, K_eps
|-- lib/extension_runs.py     # E10/E12/E15/E16 task bodies
|-- lib/photonic.py           # Phase 4: MerLin photonic neuron, baselines, Fock-space utilities
|-- lib/photonic_runs.py      # Phase 4 tasks: photonic_train, photonic_gradient_variance, kerr_gradient_variance
|-- lib/linear_reference.py   # Phase 5 baseline B4: ridge / logistic regression on Pauli expectations
|-- lib/paper_values.py       # paper and CSV reference values (read from screenshots)
|-- lib/plotting.py           # shared figure style
|-- tests/                    # gradients, data, theory, extensions, sweeps, CLI
|-- utils/generate_configs.py # writes every named config
|-- utils/run_sweep.py        # sweep coordinator (memory budget, 2-attempt limit)
|-- utils/sweep_statistics.py # paired seed statistics
|-- utils/analyze_e*.py, analyze_p4.py, analyze_p5.py, select_p5.py, plot_*.py, check_config_drift.py
|-- results/                  # curated summaries and figures
`-- outdir/                   # raw runs and sweeps (not committed)
```

## Install and How to Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r papers/fermi_dirac_neurons/requirements.txt
```

Quick smoke run (Fig. 7a setup, 2 qubits, 50 epochs, a few seconds):

```bash
# From the repository root
python implementation.py --paper fermi_dirac_neurons --config configs/defaults.json

# From inside papers/fermi_dirac_neurons
python ../../implementation.py --config configs/defaults.json
```

Single paper experiments (seconds to minutes on a CPU; add `--seed N`):

```bash
python implementation.py --paper fermi_dirac_neurons --config configs/e0_theory_checks.json
python implementation.py --paper fermi_dirac_neurons --config configs/fig7_tanh_2q_original.json
python implementation.py --paper fermi_dirac_neurons --config configs/fig8_logistic_4q_original.json
python implementation.py --paper fermi_dirac_neurons --config configs/fig9_tfim_3q_original.json
```

Full reproduction: one sweep per spec in `configs/sweeps/`. A
`--memory-budget-gib` value is required; it caps the summed memory estimate
of concurrent runs.

```bash
python papers/fermi_dirac_neurons/utils/run_sweep.py \
  papers/fermi_dirac_neurons/configs/sweeps/paper_seeds_v1.json --memory-budget-gib 4
python papers/fermi_dirac_neurons/utils/sweep_statistics.py \
  papers/fermi_dirac_neurons/outdir/sweeps/paper_seeds_v1 \
  --out papers/fermi_dirac_neurons/results/paper_seeds_v1
```

Approximate wall-clock times on an 8-core M2:

| Sweep | Time |
|---|---|
| `paper_seeds_v1` (460 runs) | 29 min |
| `e9_robustness_v1` | 25 min |
| `e10_gradient_variance_v1` | 40 min |
| `e11_training_vs_n_v4` | 35 min |
| `e16_teacher_v1` | 1 h 42 min |

The analysis command for each sweep is listed in `LOG.md` next to that
sweep's result.

Phase 4 (MerLin) and Phase 5 (fair comparison) sweeps run the same way.

Specs:
- `configs/sweeps/p4_learning_v1.json`, `p4_trainability_v1.json`,
  `p4_kerr_v1.json`;
- `p5_select_v1.json`, `p5_photonic_select_v1.json`, `p5_reference_v1.json`;
- then `utils/select_p5.py <p5_select sweep> <p5_photonic_select sweep>`,
  which writes `p5_final_v1.json` and `p5_photonic_final_v1.json`.

Tables and figures: `utils/analyze_p4.py`, `utils/analyze_p5.py`,
`utils/plot_p4.py`, `utils/plot_p5.py`, `utils/results_table.py`.

## Configuration

- `cli.json` is the authoritative list of paper-specific flags:
  - `--task` sets `experiment.task`.
  - `--sweep-id`, `--run-id` and `--repetition` are set by the sweep
    coordinator.
- Global flags are injected by the shared runtime: `--config`, `--outdir`,
  `--seed`, `--dtype` (must stay `float64`), `--device` (must stay `cpu`),
  `--log-level`, `--data-root`.
- Named configs are written by `utils/generate_configs.py`. Do not hand-edit
  them.
  - `fig*_original`: the authors' notebook protocol.
  - `fig*_papertext`: the protocol as written in the paper.
  - `fig*_classical`: the B2 and B3 baselines on the `_original` data.
  - `fig9_*_sens_{eta03,zeta1}`: the Fig. 9 sensitivity variants.
  - `e8_*` … `e16_*`: the extensions.
- Each run writes `outdir/run_YYYYMMDD-HHMMSS/` (`config_snapshot.json`,
  `run.log`, `run_status.json`, `metrics.json`, plus `history.csv` for
  training runs).
- Each sweep writes `outdir/sweeps/<id>/` with `sweep.log`,
  `sweep_status.json` and `summary.json`.

## Data

The paper uses no external dataset. Training and validation data are
generated at run time, as in Sec. VI A and the notebooks:

- Targets: random TFIM or Heisenberg Hamiltonians, with coefficients drawn
  from U[−2, 2].
- Squared-loss tasks: basis, GHZ, maximally mixed and random mixed states.
- Classification: 1000 training and 500 validation Haar-random pure states.

All randomness derives from the run seed (`lib/data.py`).

Photonic data (Phase 4, `lib/photonic.py`):
- **States:** N-photon Fock state |1^N 0…0⟩ sent through Haar-random m × m
  interferometers. Amplitudes are computed from permanents (Ryser) and fed
  as FOCK-space state vectors. The "classical data" arm uses all Fock basis
  states.
- **Targets:** tanh(H*/T) for a random nearest-neighbour passive
  Hamiltonian (coefficients U[−2, 2]), computed by dense Fock-space
  diagonalization.
- **Trainability study (E13) only:** Haar-random states on the full
  N-photon space.

Phase 5 holds out 20 % of each training set (at least 1 state), chosen by a
seeded permutation (`lib/data.py::selection_split`), for learning-rate
selection. The 500 held-out states are never used for selection. No
dataset is downloaded at any phase. The authors'
released per-epoch CSVs and plots (commit `9c2094c`) are kept locally under
`data/fermi_dirac_neurons/reference_outputs/` and are not committed.
Recreate them from the repository root:

```bash
git clone https://github.com/Alexheeeee/quantized-neurons data/fermi_dirac_neurons/sources/repo_snapshot
mkdir -p data/fermi_dirac_neurons/reference_outputs
git -C data/fermi_dirac_neurons/sources/repo_snapshot archive \
  9c2094c07aba08f190b91005c60e4db7926b924d outputs plots \
  | tar -x -C data/fermi_dirac_neurons/reference_outputs
```

The paper PDF and the equation and figure screenshots, from which every
paper value was read, are under `data/fermi_dirac_neurons/sources/` (not
committed).

## Results Obtained and Comparison with the Paper

**Labelling and statistics.**

- Every table states its seed count.
- "10 seeds" means seeds 0-9 of the `_original` protocol.
- An advantage is the median paired difference over seeds, in decades of
  loss or in accuracy points.
- p is from a two-sided Wilcoxon test (0.002 is the smallest attainable
  with 10 seeds).
- Tie tolerances are 0.1 decade for loss and 1 point for accuracy.
- Paper values are single unseeded runs.

### Summary: paper value vs reproduction (original protocol, 10 seeds)

Generated by `python3 papers/fermi_dirac_neurons/utils/results_table.py`
from `results/paper_seeds_v1_paper_placement.csv` and
`results/p5_comparison.json`.

- **Delta:** reproduced median minus paper value, in decades of training
  loss or in accuracy points.
- **Metric agreement:** "within seed range" means the paper value lies
  inside our 10-seed min–max.
- **Trend agreement:** the quantum neuron beats the paper's commuting
  baseline (for the linear rows of Fig. 9: tanh beats linear).
- **Claim support:** the Phase 5 outcome against the tuned fair baseline.

| Paper item | Task | Model | Claim tested | Paper value | Reproduced median [min, max] | Delta | Metric agreement | Trend agreement | Claim support (tuned B2, Phase 5) | Seeds | Comment |
|---|---|---|---|---:|---:|---:|---|---|---|---:|---|
| Fig. 7 | fig7_tanh_2q | quantum_tfim | C4 | 3.8e-05 | 1.2e-05 [1e-16, 0.00032] | -0.49 dec | within seed range | yes | SUPPORTED | 10 | paper value from released CSV |
| Fig. 7 | fig7_tanh_2q | classical_im | C4 | 0.012 | 0.083 [0.032, 0.13] | +0.85 dec | outside seed range | yes | SUPPORTED | 10 | paper value from released CSV |
| Fig. 7 | fig7_tanh_7q | quantum_tfim | C4 | 0.00074 | 0.0012 [0.00065, 0.0021] | +0.21 dec | within seed range | yes | SUPPORTED | 10 | paper value from released CSV |
| Fig. 7 | fig7_tanh_7q | classical_im | C4 | 0.16 | 0.14 [0.11, 0.2] | -0.06 dec | within seed range | yes | SUPPORTED | 10 | paper value from released CSV |
| Fig. 7 | fig7_softplus_2q | quantum_tfim | C4 | 0.00011 | 0.0039 [0.00012, 0.022] | +1.56 dec | outside seed range | yes | SUPPORTED | 10 | paper value from released CSV |
| Fig. 7 | fig7_softplus_2q | classical_im | C4 | 0.82 | 0.16 [0.037, 7.1] | -0.71 dec | within seed range | yes | SUPPORTED | 10 | paper value from released CSV |
| Fig. 7 | fig7_softplus_7q | quantum_tfim | C4 | 5.2 | 7.5 [0.7, 1.9e+02] | +0.16 dec | within seed range | yes | SUPPORTED | 10 | paper value from released CSV |
| Fig. 7 | fig7_softplus_7q | classical_im | C4 | 16 | 62 [5.2, 1.1e+03] | +0.58 dec | within seed range | yes | SUPPORTED | 10 | paper value from released CSV |
| Table II | fig8_logistic_2q | quantum_heisenberg | C5 | 95.2 % | 96.1 % [94.6 %, 97.6 %] | +0.9 pts | within seed range | yes | UNSUPPORTED | 10 | paper value from Table II |
| Table II | fig8_logistic_2q | classical_fcim | C5 | 64.0 % | 70.5 % [63.2 %, 76.0 %] | +6.5 pts | within seed range | yes | UNSUPPORTED | 10 | paper value from Table II |
| Table II | fig8_logistic_3q | quantum_heisenberg | C5 | 97.6 % | 95.0 % [93.6 %, 97.2 %] | -2.6 pts | outside seed range | yes | UNSUPPORTED | 10 | paper value from Table II |
| Table II | fig8_logistic_3q | classical_fcim | C5 | 65.8 % | 67.8 % [59.0 %, 76.8 %] | +2.0 pts | within seed range | yes | UNSUPPORTED | 10 | paper value from Table II |
| Table II | fig8_logistic_4q | quantum_heisenberg | C5 | 96.8 % | 94.4 % [90.8 %, 96.0 %] | -2.4 pts | outside seed range | yes | UNSUPPORTED | 10 | paper value from Table II |
| Table II | fig8_logistic_4q | classical_fcim | C5 | 68.6 % | 69.8 % [60.2 %, 76.0 %] | +1.2 pts | within seed range | yes | UNSUPPORTED | 10 | paper value from Table II |
| Table II | fig8_logistic_5q | quantum_heisenberg | C5 | 92.2 % | 92.8 % [90.2 %, 95.2 %] | +0.6 pts | within seed range | yes | UNSUPPORTED | 10 | paper value from Table II |
| Table II | fig8_logistic_5q | classical_fcim | C5 | 63.8 % | 69.7 % [61.8 %, 75.8 %] | +5.9 pts | within seed range | yes | UNSUPPORTED | 10 | paper value from Table II |
| Table II | fig8_logistic_6q | quantum_heisenberg | C5 | 91.2 % | 91.0 % [89.4 %, 94.0 %] | -0.2 pts | within seed range | yes | UNSUPPORTED | 10 | paper value from Table II |
| Table II | fig8_logistic_6q | classical_fcim | C5 | 73.6 % | 66.8 % [61.6 %, 71.8 %] | -6.8 pts | outside seed range | yes | UNSUPPORTED | 10 | paper value from Table II |
| Table II | fig8_logistic_7q | quantum_heisenberg | C5 | 91.0 % | 88.3 % [85.8 %, 90.4 %] | -2.7 pts | outside seed range | yes | UNSUPPORTED | 10 | paper value from Table II |
| Table II | fig8_logistic_7q | classical_fcim | C5 | 69.6 % | 66.4 % [59.8 %, 70.0 %] | -3.2 pts | within seed range | yes | UNSUPPORTED | 10 | paper value from Table II |
| Fig. 9 | fig9_tfim_2q | quantum_tfim_tanh | C6 | 8.4e-30 | 1.9e-13 [4.9e-30, 1.4e-05] | +16.34 dec | within seed range | yes | SUPPORTED | 10 | paper value from Fig. 9 pixel reading; settings inferred (D9) |
| Fig. 9 | fig9_tfim_2q | quantum_tfim_linear | C6 | 0.0013 | 0.0023 [1.3e-05, 0.025] | +0.26 dec | within seed range | tanh < linear: yes | SUPPORTED | 10 | paper value from Fig. 9 pixel reading; settings inferred (D9) |
| Fig. 9 | fig9_tfim_2q | classical_ising | C6 | 0.042 | 0.083 [0.032, 0.13] | +0.29 dec | within seed range | yes | SUPPORTED | 10 | paper value from Fig. 9 pixel reading; settings inferred (D9) |
| Fig. 9 | fig9_tfim_3q | quantum_tfim_tanh | C6 | 3.9e-23 | 3.2e-07 [2.3e-15, 7e-06] | +15.91 dec | outside seed range | yes | SUPPORTED | 10 | paper value from Fig. 9 pixel reading; settings inferred (D9) |
| Fig. 9 | fig9_tfim_3q | quantum_tfim_linear | C6 | 0.0037 | 0.0028 [0.0012, 0.019] | -0.13 dec | within seed range | tanh < linear: yes | SUPPORTED | 10 | paper value from Fig. 9 pixel reading; settings inferred (D9) |
| Fig. 9 | fig9_tfim_3q | classical_ising | C6 | 0.11 | 0.12 [0.059, 0.19] | +0.06 dec | within seed range | yes | SUPPORTED | 10 | paper value from Fig. 9 pixel reading; settings inferred (D9) |
| Fig. 9 | fig9_heis_2q | quantum_heis_tanh | C6 | 2.3e-05 | 8.7e-05 [3.2e-05, 0.0005] | +0.58 dec | outside seed range | yes | SUPPORTED | 10 | paper value from Fig. 9 pixel reading; settings inferred (D9) |
| Fig. 9 | fig9_heis_2q | quantum_heis_linear | C6 | 4.6e-05 | 0.0047 [8.7e-05, 0.018] | +2.01 dec | outside seed range | tanh < linear: yes | SUPPORTED | 10 | paper value from Fig. 9 pixel reading; settings inferred (D9) |
| Fig. 9 | fig9_heis_2q | classical_fcim | C6 | 0.12 | 0.087 [0.062, 0.13] | -0.14 dec | within seed range | yes | SUPPORTED | 10 | paper value from Fig. 9 pixel reading; settings inferred (D9) |
| Fig. 9 | fig9_heis_3q | quantum_heis_tanh | C6 | 2.6e-05 | 0.00032 [9.4e-05, 0.0009] | +1.09 dec | outside seed range | yes | SUPPORTED | 10 | paper value from Fig. 9 pixel reading; settings inferred (D9) |
| Fig. 9 | fig9_heis_3q | quantum_heis_linear | C6 | 0.0012 | 0.008 [0.0046, 0.018] | +0.82 dec | outside seed range | tanh < linear: yes | SUPPORTED | 10 | paper value from Fig. 9 pixel reading; settings inferred (D9) |
| Fig. 9 | fig9_heis_3q | classical_fcim | C6 | 0.055 | 0.097 [0.022, 0.14] | +0.25 dec | within seed range | yes | SUPPORTED | 10 | paper value from Fig. 9 pixel reading; settings inferred (D9) |

### Sweep records

Every sweep ran under a plan recorded in `LOG.md` before its launch, with at
most 2 attempts per planned run.

- **Comparison sweeps** make no selection: their rows are the paired
  comparisons of the linked artifact.
- **Learning-rate selection sweeps:** their full candidate tables, every
  rate for every model with per-seed values, are in
  `results/p5_selection.json`.

| Sweep | Purpose | Runs (completed / planned) | Status | Metric, direction, split, checkpoint | Aggregation and tie rule | Result artifact |
|---|---|---:|---|---|---|---|
| paper_seeds_v1 | Figs. 7–9, Table II, B2/B3 untuned | 460 / 460 | COMPLETE | training loss ↓ (train), validation accuracy ↑ (500 Haar); final epoch | paired median over 10 seeds, Wilcoxon, tie 0.1 dec / 1 pt | `results/paper_seeds_v1_comparisons.csv`, `…_paper_placement.csv` |
| e8_classical_data_v1 | C8 | 100 / 100 | COMPLETE | as above | as above | `results/e8_classical_data_summary.json` |
| e9_robustness_v1 | η, T, finite shots | 510 / 510 | COMPLETE | as above | as above | `results/e9_robustness_summary.json` |
| e10_gradient_variance_v1 | barren plateaus | 90 / 92 | **PARTIAL** (two 12-qubit cells FAILED) | gradient variance vs n | exp vs poly by ΔAIC > 2 | `results/e10_gradient_variance_summary.json` |
| e11_training_vs_n_v4 | training vs n (v1–v3 interrupted, superseded) | 70 / 70 | COMPLETE | final loss ↓ | medians, seed counts | `results/e11_training_vs_n_summary.json` |
| e12_contextuality_v1 | contextuality vs T | 3 / 3 | COMPLETE | Kirby–Love flag, graded fraction | counts over draws | `results/e12_contextuality_summary.json` |
| e15_learning_v1, e15_structure_v1 | contextual vs noncontextual sets | 30 / 30, 114 / 114 | COMPLETE | loss, gradient variance, Pauli count | medians, seed counts | `results/e15_learning_summary.json`, `results/e15_structure_summary.json` |
| e16_structure_v1, e16_teacher_v1 | gap scaling, decay rate, cross-class learning | 30 / 30, 135 / 135 | COMPLETE | Pauli-count ratio, slopes, log10 loss | bootstrap CI, 2-SE rule, sign counts | `results/e16_structure_summary.json`, `results/e16_teacher_summary.json` |
| p4_learning_v1 | photonic learning (MerLin) | 60 / 60 | COMPLETE | test MSE ↓, training loss ↓ | paired median, Wilcoxon, tie 0.1 dec | `results/p4_learning_summary.json` |
| p4_trainability_v1, p4_kerr_v1 | photonic gradient variance; Kerr (not MerLin) | 23 / 23, 14 / 14 | COMPLETE | slope of log Var vs log d | weighted LS, 2-SE rules | `results/p4_trainability_summary.json`, `results/p4_kerr_summary.json` |
| p5_select_v1 | learning-rate selection (qubit) | 378 / 378 | runs COMPLETE; coordinator FAILED in its summary step (bug, fixed; runs kept, see `LOG.md`) | selection-split log10 MSE or own loss ↓; final epoch | median over seeds 0–2; diverged = ineligible; tie → smaller η | `results/p5_selection.json` |
| p5_photonic_select_v1 | learning-rate selection (photonic, Adam) | 36 / 36 | COMPLETE | as above | as above | `results/p5_selection.json` |
| p5_reference_v1 | B4 reference | 140 / 140 | COMPLETE | λ by the same rule; test metric | as Phase 5 | `results/p5_comparison.json` |
| p5_final_v1, p5_photonic_final_v1 | tuned comparison | 140 / 140, 30 / 30 | COMPLETE | test MSE ↓ / test accuracy ↑ (500 held-out states); final epoch | paired median over 10 seeds, Wilcoxon, tie 0.1 dec / 1 pt | `results/p5_comparison.json` |

### Theory checks (E0, single run, `results/e0_theory_checks.json`)

| Check | Max absolute error | Verdict |
|---|---|---|
| Classical reduction (Eqs. 7-8, 13-15), n = 2-4, tanh/softplus/SiLU | ≤ 8.9e-16 | exact |
| Theorems 1, 2, 5, 6, Corollary 1 | ≤ 2.7e-15 | exact |
| Theorem 8 as printed | 0.066-0.12 | printed formula wrong |
| Theorem 13 as printed / corrected | 1.3-1.6 / ≤ 8.9e-16 | printed formula wrong; correction holds |
| Eqs. 41, 90 (T = 2T₁T₂); Eqs. 73, 94 (T = T₁T₂) | ≤ 9.9e-14 | hold with these conventions |
| Algorithms 1, 2 (100-10⁴ shots); 4, 5, 7 (2·10⁵ shots) | within 1.3 standard errors | unbiased |

### Fig. 7: squared loss, quantum TFIM vs commuting IM (10 seeds)

Final training loss, our median [min, max], against the released CSV
(`results/paper_seeds_v1_paper_placement.csv`):

| Panel | TFIM ours | TFIM paper | IM ours | IM paper | Advantage vs IM | Advantage vs B2 |
|---|---|---|---|---|---|---|
| tanh 2q | 1.2e-5 [1.0e-16, 3.2e-4] | 3.8e-5 | 8.3e-2 [3.2e-2, 1.3e-1] | 1.2e-2 | +3.77 (10/10) | +2.24 (10/10) |
| tanh 7q | 1.2e-3 [6.5e-4, 2.1e-3] | 7.4e-4 | 1.4e-1 [1.1e-1, 2.0e-1] | 1.6e-1 | +2.07 (10/10) | **−0.03 (tie)** |
| softplus 2q | 3.9e-3 [1.2e-4, 2.2e-2] | 1.1e-4 | 0.16 [0.04, 7.1] | 0.82 | +2.00 (10/10) | +0.81 (8/10) |
| softplus 7q | 7.5 [0.70, 194] | 5.2 | 62 [5.2, 1079] | 16 | +0.89 (10/10) | +0.16 (10/10) |

- The direction is reproduced: TFIM beats IM in every panel, 10/10 seeds,
  p = 0.002.
- Paper values fall inside our seed range, with two exceptions just
  outside: tanh 2q IM, and softplus 2q TFIM (1.1e-4 vs our minimum 1.2e-4).
- Against B2, the classical neuron on the same TFIM features, the quantum
  model ties at tanh 7q. At 7 qubits, the gap to IM comes from the X
  features IM lacks, not from quantization.
- B2 is run at the paper's η, untuned. The tuned comparison is Phase 5.
- Figure: [fig7_reproduction.png](results/figures/fig7_reproduction.png).

### Fig. 8 and Table II: classification accuracy, Heisenberg vs FCIM (10 seeds, linear prediction rule)

| n | Heisenberg ours median [min, max] | Table II | FCIM ours | Table II | Advantage (10/10, p = 0.002) |
|---|---|---|---|---|---|
| 2 | 96.1 [94.6, 97.6] | 95.2 | 70.5 [63.2, 76.0] | 64.0 | +24.9 |
| 3 | 95.0 [93.6, 97.2] | 97.6 | 67.8 [59.0, 76.8] | 65.8 | +27.9 |
| 4 | 94.4 [90.8, 96.0] | 96.8 | 69.8 [60.2, 76.0] | 68.6 | +25.2 |
| 5 | 92.8 [90.2, 95.2] | 92.2 | 69.7 [61.8, 75.8] | 63.8 | +23.6 |
| 6 | 91.0 [89.4, 94.0] | 91.2 | 66.8 [61.6, 71.8] | 73.6 | +25.5 |
| 7 | 88.3 [85.8, 90.4] | 91.0 | 66.4 [59.8, 70.0] | 69.6 | +22.9 |

- The separation is reproduced at every n.
- Training logistic losses match the released CSVs to the third digit.
- Table II's quantum values for n = 3, 4, 7 lie 0.4-0.8 points above our
  best seed, and FCIM at n = 6 lies 1.8 points above. Table II is a single
  run and differs from the released CSVs (D7).
- B2 at the paper's learning rate underfits: its accuracy falls with n
  while training accuracy equals validation accuracy. So "quantum beats a
  classical neuron" is not assessed here (Phase 5).
- Figures: [fig8_reproduction.png](results/figures/fig8_reproduction.png),
  [table2_accuracy.png](results/figures/table2_accuracy.png).

### Fig. 9: tanh vs linear vs commuting (10 seeds, 10 000 epochs, settings inferred)

| Panel | tanh ours median | tanh paper | linear ours | linear paper | commuting ours | commuting paper | tanh − linear |
|---|---|---|---|---|---|---|---|
| TFIM 2q | 1.9e-13 | 8.4e-30 | 2.3e-3 | 1.3e-3 | 8.3e-2 | 4.2e-2 | +10.49 (10/10) |
| TFIM 3q | 3.2e-7 | 3.9e-23 | 2.8e-3 | 3.7e-3 | 0.12 | 0.11 | +3.86 (10/10) |
| Heis 2q | 8.7e-5 | 2.3e-5 | 4.7e-3 | 4.6e-5 | 8.7e-2 | 0.12 | +1.27 (10/10) |
| Heis 3q | 3.2e-4 | 2.6e-5 | 8.0e-3 | 1.2e-3 | 9.7e-2 | 5.5e-2 | +1.45 (10/10) |

Paper values were read by pixel position from the figure.

- **Ordering reproduced** (tanh < linear < commuting) in all four panels,
  under both sensitivity variants as well.
- **Absolute values not reproduced.** Some paper values lie below our range,
  and no single variant matches all panels, so the settings cannot be
  identified from one unseeded run (`LOG.md`, Fig. 9 gap diagnosis).
- **The 2-qubit Heisenberg exception is not reproduced.** tanh beats linear
  by +1.27 decades (minimum +0.21). The paper's "algebra closure"
  explanation is wrong: tanh(H) has 7 Pauli strings that H lacks (D20).
- **The quantum linear model does not beat B2;** the gain comes from the
  nonlinearity.
- Figures: [fig9_reproduction_original.png](results/figures/fig9_reproduction_original.png),
  [sensitivity η = 0.3](results/figures/fig9_reproduction_sens_eta03.png),
  [ζ ~ U[−1, 1]](results/figures/fig9_reproduction_sens_zeta1.png),
  [paired advantages](results/figures/paired_advantage_loss.png).

### C8: quantum vs classical training data (E8, 10 seeds)

| Training data | Fig. 7 tanh 2q / 7q (loss decades) | Fig. 8 n = 3 / 5 / 7 (accuracy points) |
|---|---|---|
| computational-basis states | +0.05 / +0.05 (ties) | +0.4 / +1.0 / +0.9 (ties or at tolerance) |
| Haar-random states (same count) | +1.05 / +0.23 (10/10) | +11.3 / +10.2 / +18.6 (10/10) |

With classical data the quantum neuron can only use the diagonal of its
observable, which the commuting neuron also has. **C8 supported.**
(`results/e8_classical_data_summary.json`.)

### Robustness (E9, 10 seeds)

- Quantum beats commuting in all 72 (η, T, metric) cells, 10/10 seeds,
  with η ∈ {0.03, 0.1, 0.3} and T ∈ {0.5, 1, 2, 4}.
- Finite-shot gradients from Algorithms 1 and 2, Fig. 7a, quantum
  neuron's median final loss:

  | Shots per state and parameter | Final loss | Exact gradients |
  |---|---|---|
  | 10 | 7.3e-4 | 1.2e-5 |
  | 100 | 7.1e-5 | 1.2e-5 |
  | 1000 | 1.3e-5 | 1.2e-5 |

  The advantage stays above +1.98 decades (10/10).
- Details: `results/e9_robustness_summary.json`.

### Extensions requested by the user (not in the paper)

- **Barren plateaus (E10: 90/92 cells, n ≤ 11, 200 θ draws; E11: n = 2…8,
  5 seeds).**
  - With Haar-random inputs, the gradient variance falls ≈ 2⁻ⁿ (per-qubit
    factor 0.43-0.52) for every term set, commuting or not, and at every
    temperature. This is data-induced concentration.
  - With basis, GHZ or product inputs at T = 2 there is no exponential
    decay (factor 0.80-1.32; Heisenberg with GHZ
    or |0…0⟩ is the low end, classified polynomial).
  - Training on the authors' states works up to n = 8: TFIM beats IM by
    2.0-3.6 decades, 5/5 seeds.
  - Training on Haar states stalls by n ≈ 6-8.
  - Figures: [e10](results/figures/e10_gradient_variance_vs_n.png),
    [e11](results/figures/e11_training_vs_n.png).
- **Contextuality vs temperature (E12, single run per scan, 20 draws).**
  - Kirby-Love noncontextuality of a term set is preserved by φ(H) at every
    temperature: 0/120 scans became contextual.
  - So no temperature switches contextuality on.
  - Figure: [e12](results/figures/e12_contextuality_vs_temperature.png).
- **Contextual vs noncontextual term sets at equal parameter count
  (E15, E16).**
  - **Learning.** Contextual sets do not learn better when they don't match
    the target: E15, 5 seeds. In the cross-class teacher-student test (E16b,
    135 runs, 5 seeds), every class-advantage median is within ±0.1 decade.
    Caveat: at the paper's budget, mismatched students barely leave their
    initialization, so a difference at convergence is not excluded.
  - **Gradient decay rate.** Not distinguishable by contextuality in 8 of 9
    (input, parameter) cells (E16c, n = 4…10).
  - **Pauli-measurement cost.** Here contextuality separates. The number of
    Pauli strings needed to compute the output within ε grows faster for
    contextual sets. The ratio, contextual over noncontextual, grows by a
    factor of 1.3-1.95 per qubit. At θ ~ U[−2, 2] it is about 1 at n = 4
    and about 25 at n = 10, and all 8 bootstrap CIs are above 0.
  - Figure: [e15_structure.png](results/figures/e15_structure.png);
    summaries in `results/e15_*`, `results/e16_*`.

## Fair Baselines

**Why the paper's baseline is not enough.** The paper compares its
quantum neurons with commuting Ising neurons (IM, FCIM). Those see only
Z-basis measurements, while the targets depend on X and Y information, so
the paper's gap mixes "quantization" with "access to measurements"
(baseline state `MISSING_OR_UNFAIR`).

**The fair control (planned in Phase 1).** B2 is a classical neuron
φ(Σ θ_j⟨H_j⟩) on the same measured Pauli features, with the same parameter
count. Every comparison also includes:
- B1: the paper's commuting neuron;
- B3: the quantum linear model;
- B4: a ridge or logistic-regression reference on all ≤ 2-local Pauli
  expectations, with more parameters.

**Protocol (frozen in `LOG.md` before any run):**
- **Tuning:** every model's learning rate is tuned separately. Stage A
  tries η ∈ {0.01 … 100} with seeds 0–2 on a 20 % selection split held out
  of the training states. Stage B re-runs each model at its selected rate
  with seeds 0–9.
- **Test set:** 500 held-out states, never used for selection.
- **Statistics:** paired by seed, two-sided Wilcoxon, tolerance 0.1 decade
  or 1 accuracy point.
- **Photonic models:** all use Adam with lr ∈ {0.003 … 0.1}.
- **Claim rule:** a claim is SUPPORTED only if the quantum neuron beats
  tuned B2 in every task.

**Tuned quantum neuron vs tuned B2**, primary metric, median paired
advantage over 10 seeds (`results/p5_comparison.json`,
[figure](results/figures/p5_paired_advantage.png)):

| Claim | Tasks | Advantage over tuned B2 | Seeds favouring quantum | Outcome |
|---|---|---|---|---|
| C4 (Fig. 7) | tanh 2q, 7q; softplus 2q, 7q | +27.8, +21.4, +1.6, +1.9 decades of test MSE | 10/10 each | **SUPPORTED** (confounded, see below) |
| C5 (Fig. 8, Table II) | n = 2…7 | −3.1, −3.6, −4.7, −5.3, −5.4, −5.3 accuracy points | 0/10 each | **UNSUPPORTED** |
| C6 (Fig. 9) | TFIM 2q, 3q; Heis 2q, 3q | +28.7, +28.6, +5.8, +2.4 decades (and tanh beats the tuned quantum linear model in every panel) | 10, 10, 9, 10 /10 | **SUPPORTED** (confounded) |
| MerLin adaptation | (N, m) = (2, 4), (3, 6), (4, 8) | +0.16, +0.65, +1.04 decades vs the correlation neuron | 10/10 each | **SUPPORTED** (adaptation, simulation; confounded) |

**How to read these outcomes:**
- **The confounder behind C4, C6 and MerLin.** With a tuned learning rate
  the quantum neuron fits the Fig. 7 tanh and Fig. 9 TFIM targets to
  round-off (test MSE ≈ 1e-31 to 1e-24). The paper's targets are tanh of a
  random Hamiltonian from the model's own family, so the teacher lies
  exactly in the quantum model's hypothesis class. B2 and the linear
  models cannot represent ⟨tanh(H)⟩ exactly.
- **So these wins show a well-specified model beating a misspecified
  one,** not a general learning advantage of quantization. Phase 3 (E16b)
  found that students whose terms differ from the teacher's barely learn.
- **The paper's η = 0.1 under-trains the quantum neuron.** Phase 3 had
  found tanh 7q tied with untuned B2.
- **C5 reverses.** In the released code the labels are the sign of
  Tr[H*ρ], linear in B2's features, so tuned logistic regression (97.4–
  99.4 % test accuracy) beats the tuned Heisenberg neuron (91.6–96.1 %).
  The paper's large gap to FCIM (+22 to +27 points) remains, but it
  reflects FCIM's missing X and Y features.
- **B4** (more parameters) beats the quantum neuron on Fig. 8 for n ≤ 5,
  ties at n = 6 and loses at n = 7. It cannot reach the quantum neuron's
  round-off fit on the TFIM tasks.
- **MerLin.** With matched, tuned optimizers the photonic neuron now beats
  its classical correlation neuron, reversing the Phase 4 result obtained
  under the paper's plain gradient descent.

**Wall-clock** (Apple M2, one thread per run):
- **Per model at Fig. 7 tanh 7q:** the quantum neuron takes 35.6 s per
  2000 epochs, B2 0.12 s. Exact simulation of the quantum model costs
  O(d³) per step; B2 needs only J measured expectations.
- **Per sweep:** 15–49 min (`LOG.md`).

**Not compared:**
- qubit vs photonic models (different tasks);
- Phase 5 tuned values vs paper values (Phase 5 trains on 80 % of the
  states; paper values are compared in "Results Obtained…" above).

## Photonic Assessment and MerLin Extension

**Translation status: `PARTIAL_MERLIN_TRANSLATION`.** This is an adaptation,
not part of the paper's experiments.

**Why the mapping is exact.** The paper's continuous-variable neuron
(Sec. V, eq. 108) applies an activation to a quadratic bosonic
Hamiltonian. Take the passive (photon-number-conserving) case,
H = Σ h_ij a_i†a_j with h = U† diag(ε) U. Its N-photon eigenstates are
Fock states of the normal modes that the interferometer U reaches, so

  Tr[tanh(H/T) ρ] = Σ_n p_U(n) tanh(ε·n / T).

The neuron is therefore one trainable MerLin mesh, photon-number-resolving
detection and a classical tanh. It matches dense Fock-space numerics to
1e-12 (`tests/test_photonic.py`).

**Not translated:**
- the paper's qubit tasks (ZZ couplings are interactions, not linear
  optics);
- squeezing terms;
- the control-qumode Algorithms 1–7 (replaced by sampling in the
  eigenbasis: one bounded shot per measurement).

The cross-Kerr ablation is dense numerics, **not MerLin**.

Models, all trained with the paper's Fig. 7a protocol (squared loss,
T = 2, full-batch GD, η = 0.1, 2000 epochs):
- **photonic neuron**: MerLin mesh + energies ε (m² parameters, m/2 of them
  inert output phases);
- **photonic linear**: Tr[Hρ];
- **commuting photon-count neuron**: a classical second-order neuron on the
  photon counts, the counterpart of IM / FCIM;
- **classical correlation neuron**: tanh of a linear function of the m²
  one-photon correlations ⟨a_i†a_j⟩, the B2 counterpart.

Tasks:
- **Teacher**: tanh of a random nearest-neighbour passive Hamiltonian
  (ζ ~ U[−2, 2]).
- **Quantum data**: Fock state |1^N 0…0⟩ through Haar-random
  interferometers.
- **Classical data**: Fock basis states.

**Learning (10 seeds, paired, two-sided Wilcoxon, 0.1-decade tolerance;
`results/p4_learning_summary.json`).**

Advantage of the photonic neuron in validation MSE, decades:

| (N photons, m modes) | vs commuting, interferometer data | vs commuting, Fock data | vs photonic linear, interferometer | vs correlation (B2), interferometer |
|---|---|---|---|---|
| (2, 4) | +0.16 (8/10, supported) | −0.05 (tie) | +0.17 (not supported) | +0.07 (tie) |
| (3, 6) | +0.13 (10/10, supported) | −0.12 | −0.07 (tie) | −0.45 (0/10) |
| (4, 8) | +0.14 (10/10, supported) | −0.13 | −0.04 (tie) | −1.08 (0/10) |

Against the declared preservation criteria:
- **C4 preserved** at every size: non-commuting beats commuting on quantum
  data, but by only +0.13 to +0.62 decade (training and validation),
  against +2 to +4 on qubits.
- **C8 preserved:** no advantage with classical (Fock-basis) data.
- **C6 not preserved:** tanh does not beat linear.
- **The classical correlation neuron wins** at (3, 6) and (4, 8).

**Caveat (post-hoc diagnostic and one exploratory seed, not a sweep
result):**
- The photonic model contains the teacher exactly, yet under the paper's
  GD it removes only 46 % of the target variance at (4, 8).
- With Adam (lr 0.02, 2000 epochs) the same seed reaches validation MSE
  2.7e-4. The correlation neuron reaches 1.4e-3 (10-seed median).
- So the B2 win reflects the paper's optimizer on mesh phases, not the
  model. A same-optimizer, tuned comparison belongs to Phase 5.

[Learning curves](results/figures/p4_learning_curves.png).

**Detectors and shots** (trained photonic neuron, median log10 of the
validation-MSE ratio to exact PNR, interferometer data, sizes (2,4), (3,6),
(4,8)):

| Readout | (2, 4) | (3, 6) | (4, 8) |
|---|---|---|---|
| Threshold detectors | +0.003 | +0.05 | +0.10 |
| Collision-free postselection | +0.11 | +0.13 | +0.12 |
| 100 PNR shots per state | +0.02 | +0.03 | +0.03 |
| 1000 PNR shots per state | ≤ +0.005 | ≤ +0.005 | ≤ +0.005 |

Threshold detection costs more as photons bunch; 1000 shots per state are
as good as exact.

**Trainability (E13, MerLin, seed 0, 200 draws per point;
`results/p4_trainability_summary.json`,
[figure](results/figures/p4_trainability.png)).** At half filling
(N = m/2, d = 10…2002):
- **X2(a), linear-optics inputs:** the gradient variance falls more slowly
  than 1/d in 16/16 cells, a factor of 0.52–0.99 per added mode.
- **X2(b), Haar-random states on the whole N-photon space:** mesh-phase
  gradients follow 1/d, but energy gradients keep a non-zero mean (≈ 0.22 at
  every size, because n₁ is not traceless) and do not vanish. X2(b) is not
  supported (3/8 cells).
- m/2 mesh phases of MerLin's entangling layer only set output phases and
  have zero gradient; they are excluded.

**Cross-Kerr (E13-K, NOT MerLin, dense Fock-space numerics;
`results/p4_kerr_summary.json`, [figure](results/figures/p4_kerr.png)).**
- Strong Kerr (κ = 2) at half filling turns the Fock-input gradient
  variance from almost flat to the Haar rate 1/d.
- Weak Kerr and fixed photon numbers change little (11 of 36 comparisons
  differ).
- Interactions, not linear optics, bring the barren-plateau rate back for
  structured inputs.

**Phase 5 update.** Once every photonic model gets the same tuned
optimizer (Adam), the photonic neuron beats the classical correlation
neuron at every size: +0.16, +0.65 and +1.04 decades, 10/10 seeds. The
learning comparison above (paper GD) therefore reflects an optimizer
handicap of the mesh phases. See "Fair Baselines".

## Hardware-Aware Settings

Every executed MerLin result (Phase 4 P4-L and E13, Phase 5 photonic tuning, which uses the same settings with Adam):


| Field | P4-L (learning) | E13 (trainability) |
|---|---|---|
| Computation space | `FOCK` | `FOCK` |
| Detector model | photon-number-resolving (ablations at evaluation: threshold, UNBUNCHED postselection) | photon-number-resolving |
| Photon number | 2, 3, 4 | 2, 3, 4, 5 |
| Number of modes | 4, 6, 8 | 2–20 |
| Input state | Φ(V)\|1^N 0…0⟩ with Haar-random V (quantum data); Fock basis states (classical data) | \|1^N 0…0⟩; Φ(V)\|1^N 0…0⟩; Haar-random state on the N-photon space (control, not linear-optics producible) |
| Encoding | data as the input state (state-vector amplitude input; physically a data interferometer V in front of the trainable mesh) | same |
| Measurement strategy | `MeasurementStrategy.probs(FOCK)`, readout Σ_n p(n) tanh(ε·n/T) | same |
| Postselection | none (UNBUNCHED ablation = collision-free postselection) | none |
| Simulator / QPU | MerLin 0.4.1 CPU simulator, analytic, float64; no hardware | same |
| Shot count | exact probabilities; shot ablation 100 and 1000 PNR samples per state | exact |
| Gradients | exact, torch autograd through MerLin (simulation only) | same |
| Wall-clock per run (1 CPU thread) | 14–30 s at (2,4) and (3,6); 3.5 min at (4,8) | 6 s to 15 min (d = 2002) |
| Seeds | 0–9 | 0 |

The Kerr ablation is dense numpy, not MerLin: 3 s to 51 min per run, seed 0.

Phase 5 photonic runs: the same detector, space and inputs, with Adam
(lr ∈ {0.003, 0.01, 0.03, 0.1}). Wall-clock 49 min (selection, 36 runs)
and 32 min (confirmation, 30 runs) on one worker; seeds 0–2, then 0–9.

## Limitations

- **Single runs and unseeded originals.** The paper reports one unseeded
  run per setting, so a paper value can only be placed inside or outside our
  10-seed range, not matched. Fig. 9 settings are not stated and could not
  be identified.
- **Baselines.** All B2 comparisons are at the paper's learning rate. B2
  underfits on Fig. 8, so the quantum-vs-classical-neuron question is open
  until the tuned comparison (Phase 5).
- **Small systems.** Every result is at n ≤ 8 for training and n ≤ 11 for
  gradient variances. Exponential vs polynomial decay is hard to separate
  over n = 4…10; many cells are classified "undecided".
- **Failed or untested items.**
  - E10's two n = 12 cells failed twice and were not retried.
  - Theorem 12, Corollary 2, Theorems 14 and 15 were not checked.
- **E15 and E16.** These use 5 seeds and report medians and seed counts
  only, with no significance tests. The E16b training budget is short.
- **Photonic adaptation (Phase 4).**
  - Simulation only, with lossless and indistinguishable photons.
  - The tasks are photonic counterparts of the paper's, not its datasets.
  - Training uses the paper's optimizer, which under-trains the mesh;
    tuned optimizers are Phase 5.
  - E13 and E13-K use one parameter seed.
  - The passive neuron may be classically approximable through few-body
    correlators (not tested).
- **Fair comparison (Phase 5).**
  - Learning rates are tuned on 3 seeds.
  - The selection split is small for the 2-qubit Fig. 7 / 9 code-state
    sets (3 states).
  - Some selections sit at the edge of the grid (B2 on Fig. 8 at η = 100;
    photonic lr = 0.1), so those baselines may be under-tuned.
  - Fig. 8 uses the code label rule only.
  - The decisive confounder, a teacher inside the quantum model's own
    class, was not removed. The test that would remove it (a teacher
    outside each model's class, with tuned rates) was not run.
- **Not done yet.** The hybrid-network contextuality task (E14).

## Tests

```bash
cd papers/fermi_dirac_neurons
pytest -q
```

802 tests cover:

- gradients against finite differences;
- the data contract;
- theory identities;
- the extension modules;
- the photonic neuron against dense Fock-space numerics, its baselines and
  detector ablations (`tests/test_photonic.py`);
- the Phase 5 selection split, tuned training, divergence handling, the B4
  reference and the selection rule (`tests/test_phase5.py`);
- the sweep coordinator and memory scheduler;
- the CLI.

## Citation and License

```bibtex
@misc{he2026fermidirac,
  title         = {Fermi-Dirac machines as quantizations of neurons},
  author        = {He, Alexander and Liu, Nana and Wilde, Mark M.},
  year          = {2026},
  eprint        = {2605.24386},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph}
}
```

The paper's arXiv license was not recorded in this reproduction; cite the
paper as above. The authors' repository has no license, so none of its code
or data is included here.
