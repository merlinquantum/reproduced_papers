# The effect of data encoding on the expressive power of variational quantum machine learning models - Reproduction

Status: reproduction complete (planning, implementation, reproduction, photonic adaptation, fair comparison, documentation). Photonic status: `PARTIAL_MERLIN_TRANSLATION`. `INSIGHTS.md` holds the reusable lessons.

## Reference and Attribution

- Paper: The effect of data encoding on the expressive power of variational quantum machine learning models (Phys. Rev. A 103, 032430, 2021)
- Authors: Maria Schuld, Ryan Sweke, Johannes Jakob Meyer
- DOI/ArXiv: arXiv:2008.08605 (https://arxiv.org/abs/2008.08605)
- Original repository (if any): https://github.com/XanaduAI/expressive_power_of_quantum_models (tutorial notebook, commit `fe3ad00`, no license file)
- License and attribution notes: cite the paper; arXiv preprint terms apply

## Original Paper

Schuld, Sweke and Meyer study which functions a variational quantum model can
express as a function of its input. A model `f(x) = <0|U(x)† M U(x)|0>` whose
data enters only through encoding gates `exp(-ixH)` is a Fourier-type sum
`Σ c_ω e^{iωx}`:
- the frequency spectrum `Ω` depends only on the eigenvalues of the encoding
  generator;
- the trainable blocks and the observable set the coefficients.

Consequences claimed in the paper:
- A single Pauli-rotation encoding gives a sine of one fixed frequency,
  whatever the width or depth of the rest of the circuit.
- Repeating the encoding `r` times, in sequence or in parallel, gives
  integer frequencies up to `r`.
- The spectrum size is bounded.
- With arbitrary global unitaries and an arbitrary observable, a model
  family whose spectra eventually contain every integer is a universal
  approximator of `L²`.

Experiments are three small PennyLane simulations on synthetic targets:
Fig. 3 (single-qubit fits and frequency mismatch), Fig. 4 (degree-5 fit with
r = 1, 3, 5 repetitions) and Fig. 5 (Fourier-coefficient distributions of 100
random circuits). The paper reports no numbers, seeds or uncertainty for them.

## Reproduction Scope, Claims, and Deviations

**Reproduced**:

| ID | Claim | Experiment |
|---|---|---|
| C1, C4a, C6, C10, C11 | Spectrum fixed by the encoding; repetition gives `{-r..r}`; size bound; multivariate and non-integer cases | E4, exact checks |
| C2, C3 | A single encoding fits degree 1 only; a frequency mismatch fails | E1, Fig. 3 |
| C4b, C5 | Repetition r = 1, 3, 5 fits the degree-5 target, serial and parallel | E2, Fig. 4 |
| C8 | Coefficient distributions by ansatz and depth | E3, Fig. 5 |
| C9 | Universality, constructive proof | E5 |
| C7 | A phase-shift generator gives a full integer spectrum | E6, photonic adaptation |

Not reproduced: the paper's PennyLane code (the circuits are re-implemented
in torch), its Section V design guidance (interpretation, not testable as
stated), and any hardware run.

**Deviations from the paper** (details in "Notes on the Paper"):
- PennyLane is replaced by a torch statevector simulator with PennyLane's
  gate conventions. It is verified against an independent dense-matrix build
  and parameter-shift gradients.
- Fig. 4 was reproduced with input scaling 1, as in the published panel and
  the authors' notebook. The caption's scaling 2 was run as well.
- Missing settings were taken from the authors' notebook: grid
  `linspace(-6, 6, 70)`, initial angles `U[0, 2π)`, Adam with lr 0.3,
  mini-batches of 25 with replacement, and exactly 200 steps.
  The paper's "at most 200 steps" implies an unstated stopping rule.
  Fig. 3 gives no learning rate; the notebook's 0.3 was used. The published
  Fig. 3 loss curves span about 100 steps, not 200. Extra steps can only
  lower the training MSE, and rows b and c sit at their floor in any case.
- The notebook minimises `0.5·MSE`; here the loss is plain MSE. Adam is
  invariant to a constant loss scale apart from its `ε`.
- Six seeds (0–4 and the notebook's 42) replace the paper's single run.
  All claims are compared at trend level.
- The paper's "typical run" shows the r = 5 fit working. Under its own
  protocol that happens in 25–35% of seeds (20-seed diagnostic).

**Added beyond the paper:** exact claim checks (E4, E5); a closed-form
least-squares baseline (B1) and a trained classical baseline (B2) with the
same frequency set; a MerLin photonic adaptation; training protocols P1–P5
(labelled non-paper); and three Phase 5 extensions.

**Hardware and software:**
- Apple M5, 10 cores, 16 GB, CPU only;
- Python 3.12.7, torch 2.4.1, merlinquantum 0.2.3, perceval-quandela 1.0.0;
- the two configured remote GPU machines were unreachable and not needed.

## Project Layout

```text
papers/data_encoding_expressivity/
├── README.md, INSIGHTS.md
├── notebook.ipynb   the reproduction as a scientific walkthrough (runs in ~40 s on CPU)
├── cli.json, requirements.txt, implementation.py
├── configs/      one JSON per experiment variant (see "Configuration")
├── lib/          circuits.py, photonic_merlin.py, data.py, fourier.py, baselines.py,
│                 spectrum.py, universality.py, training.py, experiments.py,
│                 plotting.py, runner.py, experiment_logging.py
├── tests/        145 focused tests (see "Tests")
├── utils/        summarize_*.py, compare_phase5.py, plot_*.py
├── results/      curated tables (.md/.json) and figures (.png)
├── models/, outdir/   (outdir/ holds raw runs and is not committed)
```

## Install and How to Run

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # run from papers/data_encoding_expressivity
```

### Command-line interface

Main entry point: the repo-level `implementation.py`. The CLI is entirely described in `cli.json`, so updating/adding arguments does not require editing Python code.

```bash
# From inside papers/data_encoding_expressivity
python ../../implementation.py --help

# From the repo root
python implementation.py --paper data_encoding_expressivity --help
```

Example overrides (see `cli.json` for the authoritative list):

- `--config PATH` Load an additional JSON config (merged over `defaults.json`).
- `--seed INT` Random seed applied via the generic runtime.
- `--dtype STR` Force a global torch/NumPy dtype (e.g., `float32`, `float64`).
- `--device STR` Device override (`cpu`, `cuda:0`, `mps`, etc.).
- `--log-level LEVEL` Logging verbosity (`DEBUG`, `INFO`, ...).
- `--outdir DIR` Base output directory (default in `defaults.json`).
- `--steps INT` Override `fit.training.steps` (optimizer steps of a fitting experiment).
- `--lr FLOAT` Override `fit.training.lr`.
- `--batch-size INT` Override `fit.training.batch_size`.
- `--seeds 0,1,2` Override `fit.seeds`.
- `--full-batch true|false` Override `fit.training.full_batch` (protocols P1–P5 use `true`).
- `--optimizer adam|adam_cosine|lbfgs` Override `fit.training.optimizer` (P4 = `adam_cosine`, P5 = `lbfgs`, which needs `--full-batch true`).

Example runs:

```bash
# Smoke run (reduced; checks the pipeline, not the paper's results), from the repo root
python implementation.py --paper data_encoding_expressivity --config configs/e2_fig4_reduced.json

# Default config: lightweight E4 exact spectrum checks
python implementation.py --paper data_encoding_expressivity

# Paper experiments (full settings)
python implementation.py --paper data_encoding_expressivity --config configs/e4_spectrum_original.json
python implementation.py --paper data_encoding_expressivity --config configs/e1_fig3_original.json
python implementation.py --paper data_encoding_expressivity --config configs/e2_fig4_original.json
python implementation.py --paper data_encoding_expressivity --config configs/e3_fig5_original.json
python implementation.py --paper data_encoding_expressivity --config configs/e5_universality_original.json

# Photonic adaptation and fair comparison (paper protocol by default; see "Configuration" for P1-P5 flags)
python implementation.py --paper data_encoding_expressivity --config configs/e6_spectrum_merlin.json
python implementation.py --paper data_encoding_expressivity --config configs/e8_fig4_improved_merlin.json --full-batch true --steps 1000 --lr 0.05
python implementation.py --paper data_encoding_expressivity --config configs/e10_fig4_published_classical_zero_init.json

# Inline overrides, from inside papers/data_encoding_expressivity
python ../../implementation.py --config configs/e2_fig4_original.json --steps 1000 --seeds 0,1,2
```

The script saves a snapshot of the resolved config alongside results and logs.

## Data


- No data is read from or written to the shared data root by the experiments. `data/data_encoding_expressivity/sources/` only caches the paper PDF and LaTeX source and the original repository snapshot (not committed).
- **Data for this paper:** nothing to download. The paper uses no external
  dataset. Its targets are synthetic truncated Fourier series, defined in the
  captions of Figs. 3–4 and generated on the fly from config on the grid
  `linspace(-6, 6, 70)` (taken from the authors' notebook). Fig. 3:
  `c_0 = 0.1`, `c_n = 0.15 − 0.15i`, degree 1 or 2. Fig. 4: `c_0 = 0`,
  `c_n = 0.05 − 0.05i`, degree 5, input scaling 1 (the published panel; the caption's
  scaling 2 is the variant). The square-wave target of the photonic universality test is the
  degree-31 truncation of `sign(sin x)` on 128 periodic points. These synthetic targets are
  the paper's original data, not a substitute. Their only limitation is the
  paper's own: noiseless 1D targets with no held-out split, so no
  generalisation claim is tested. The full generator specification is in
  `lib/data.py`. `data/data_encoding_expressivity/sources/`
  only caches the paper source and the original repository snapshot (commit
  `fe3ad00`).

### Output directory and generated files

At each run, a timestamped folder is created under the base `outdir` (default: `outdir`):

```
<outdir>/run_YYYYMMDD-HHMMSS/
├── config_snapshot.json   # Resolved configuration used for the run
├── run.log                # Lifecycle, every optimizer step, every evaluation
├── run_status.json        # Status, code commit, dataset identity, file hashes
├── metrics.json           # Structured results of the experiment
└── <artifacts>            # fit: fit_predictions.npz, fit_cases.png
                           # coefficients: coefficients.npz, coefficient_clouds.png
                           # universality: universality_square_wave.png
```

Notes:
- Change the base output directory with `--outdir` or the `outdir` config key.
- `outdir/` is ignored by Git; curated results go to `results/`.

## Configuration

Configuration files live in `configs/`. Each named config overlays `defaults.json` and selects one experiment through the top-level `experiment` key (`fit`, `coefficients`, `spectrum`, `universality`, `photonic_spectrum` or `coefficient_reach`), whose settings live in the section of the same name.

| Config | Experiment |
|---|---|
| `defaults.json` | E4 exact spectrum checks, lightweight (r ≤ 3) |
| `e4_spectrum_original.json` | E4 exact spectrum checks (Sections II–III, App. A–B) |
| `e1_fig3_original.json` / `e1_fig3_classical.json` | E1, Fig. 3: quantum model / baseline B2 |
| `e2_fig4_original.json` / `e2_fig4_classical.json` | E2, Fig. 4 with input scaling 2: quantum / B2 |
| `e2_fig4_scaling1.json` | E2 with the reference notebook's scaling 1 |
| `e2_fig4_reduced.json` | Smoke run |
| `e3_fig5_original.json` | E3, Fig. 5 coefficient distributions |
| `e5_universality_original.json` | E5, App. C universality construction |
| `e6_spectrum_merlin.json` | Phase 4 (MerLin adaptation): photonic spectrum checks, E6 / C7 |
| `e2_fig4_merlin.json` | Phase 4: photonic fits of the Fig. 4 and Fig. 3c targets |
| `e2_fig4_merlin_reduced.json` | Phase 4 smoke run |
| `e3_fig5_merlin.json` | Phase 4: photonic coefficient distributions |
| `e7_coefficient_reach_merlin.json` | Phase 4 diagnostic: largest reachable top coefficient |
| `e8_fig4_improved_merlin.json` | Phase 4 amendment: improved photonic models (universal meshes, trainable readout, multi-scale, single photon) |
| `e8_fig4_trainable_observable.json` | Phase 4 amendment: qubit fairness control, `<Z_0>` vs trainable observable |
| `e9_universality_merlin.json` | Phase 4 amendment: photonic universality test (single photon, square wave) |
| `e10_fig4_published_classical.json` / `…_zero_init.json` | Phase 5 baseline B2 on the published Fig. 4 target, uniform / zero initial coefficients |
| `e11_multiscale_square_merlin.json` / `…_classical.json` | Phase 5 extension X2: multi-scale photonic encodings and B2 on the square wave |
| `e12_fixed_readout_ceiling_merlin.json` / `…_qubit.json` | Phase 5 extension X1: fixed-readout ceiling under L-BFGS |

**Protocols.** Every fitting config defaults to the paper's P0 (Adam, mini-batch 25, lr 0.3, 200 steps).
The others are non-paper and are set from the command line:

| Protocol | Flags |
|---|---|
| P1 | `--full-batch true --steps 1000` |
| P2 | `--full-batch true --steps 1000 --lr 0.05` |
| P3 | `--full-batch true --steps 1000 --lr 0.01` |
| P4 | `--full-batch true --optimizer adam_cosine --lr 0.05 --steps 1000` |
| P5 | `--full-batch true --optimizer lbfgs --lr 1 --steps 500` |

- `cli.json` is the authoritative list of paper-specific flags.
- The top-level `"dtype"` is `float64`. The torch models then simulate in `complex128`.
- Baseline B1 (closed-form least-squares fit with the model's own frequency set) is computed in every fitting run.

**Placeholder guard:** If any config value still contains a `<<...>>` placeholder (e.g., `"teacher_path": "<<TEACHER_PATH>>"`), the shared runner aborts early with a clear error. Replace these placeholders with real paths/values before launching a run.

## Results Obtained and Comparison with the Paper

**Comparison scope.** The paper gives no numbers for Figs. 3–5: they are
single "typical" runs or scatter plots, with no seeds or uncertainty. The
comparison with the paper is therefore **qualitative (trend-level)**.
Quantitative statements below are about this reproduction only. Every value
comes from the `metrics.json` of its run, via
the tables in `results/*_summary.md`. "Delta" is not defined where the paper gives
no value. "B1" is the closed-form least-squares fit using exactly the model's frequency set: the lowest error any model of that
class can reach. "B2" is the same frequency set as a classical model trained
with the paper's protocol.

| Paper item | Claim tested | Paper value | Reproduced value (label) | Delta | Metric agreement | Trend agreement | Claim support | Seeds | Comment |
|---|---|---|---|---|---|---|---|---:|---|
| Sections II–III, App. A–B | Spectrum = eigenvalue differences; `{-r..r}`; bounds | Derivation only | 19/19 checks pass; largest stray coefficient 8.4e-16 (exact check) | N/A | N/A (analytic) | Yes | Supported | 5 draws/check | Serial, parallel, dense, rotated, degenerate, 2D, non-integer generators |
| Fig. 3 row a | Degree-1 target fits with RX(x) | Fits (qualitative) | Final MSE 4.2e-9 ± 6.8e-9; 6/6 reach B1 (paper-accurate) | N/A | N/A (no paper value) | Yes | Supported | 6 | |
| Fig. 3 row b | Frequency mismatch RX(2x) cannot fit | Cannot fit | 0.103 ± 0.013; every seed ≥ B1 = 0.0897 | N/A | N/A | Yes | Supported | 6 | Loss axis of the paper's panel lies below the achievable floor (see "Notes on the Paper") |
| Fig. 3 row c | Degree-2 target cannot be fit by one encoding | Cannot fit | 0.0988 ± 0.0048; every seed ≥ B1 = 0.0897 | N/A | N/A | Yes | Supported | 6 | B2 with the same frequency set hits the same ceiling |
| Fig. 4 | Fit improves with r; r = 5 "almost perfect" | Qualitative, one run | Mean final MSE r = 1, 3, 5: serial 4.1e-2, 3.7e-2, 2.5e-2; parallel 4.6e-2, 3.4e-2, 3.3e-3 (paper-accurate, scaling 1). B1 at r = 5 ≈ 0. Near-perfect r = 5 fits: serial 5/20, parallel 7/20 (diagnostic) | N/A | N/A | Yes | Partially supported | 6 (20 diagnostic) | Expressible; reliably reached only under non-paper protocols (see "Fair Baselines") |
| Fig. 5 | Coefficient distributions by ansatz and depth | Qualitative (scatter) | Circuit B zeroes c₀, c₂, c₄; variance falls with order; c₅ ≡ 0; c₄ nonzero only at l = 5 for Circuit A (paper-accurate) | N/A | N/A | Yes | Supported | 100 draws/setting | Depth matters for which coefficients appear |
| Section IV, App. C | Universality construction | Existence proof | Rebuilds random truncated series to 1.1e-14; RMS error on sign(sin x) matches Parseval to 3.9e-5 relative (exact check) | N/A | N/A (analytic) | Yes | Supported | 1 construction/degree | Needs `H_m = ½Σσ_z` (Eq. 28 typo) |

Where a paper value is "qualitative", the comparison is trend-level only, so
metric agreement, delta and quantitative claims are not defined.

Figures (in `results/`):
- `fig3_reproduction.png`
- `fig4_reproduction.png` and `fig4_mse_vs_r.png` (input scaling 1, as in the published figure)
- `fig4_reproduction_s2.png` and `fig4_mse_vs_r_s2.png` (scaling 2, as in the caption, with B2 and the 1000-step sensitivity runs)
- `fig5_reproduction.png`
- `e4_spectrum_checks.png`
- `e5_universality.png`

Each figure is stamped with its result-scope label.

Issues found in the paper's text and figures are listed in "Notes on the
Paper" below.

To regenerate a table or figure, run its config (see "Install and How to Run"),
then the matching `utils/` script on the run directory. Each script's docstring
gives its exact usage.

## Notes on the Paper

None of the points below affects the paper's central claims. The frequency
spectrum is fixed by the encoding gates, repetition extends it, and the
universality construction works: all three reproduce exactly (E4, E5). One
point is substantive, five are errors or unexplained details in the text and
figures, and two are mild overstatements or loose statements.

### Substantive: expressible is not the same as trainable (Fig. 4)

Fig. 4 shows near-perfect fits of the degree-5 target at r = 5 for both
layouts, after at most 200 Adam steps. The theory says the target is exactly
representable at r = 5. Our oracle B1 confirms this, with an error of about
1e-32. But the figure also suggests the stated protocol reliably gets there,
and it does not:

| Model at r = 5 (paper protocol, input scaling 1) | Seeds with final MSE ≤ 1e-3 | Seeds reaching B1 (within 1e-4) |
|---|---:|---:|
| Serial, 1 qubit, 18 parameters | 5/20 | 3/20 |
| Parallel, 5 qubits, 90 parameters | 7/20 | 1/20 |
| B2: classical model with the same frequencies and protocol, 11 parameters (run at input scaling 2) | 6/6 | 6/6 (MSE ≈ 2e-7) |

Rows 1–2 come from the 20-seed diagnostic. B2 was run with 6 seeds on the
scaling-2 variant, where the circuit models do no better (serial 0/6,
parallel 0/6 seeds reach B1 at 200 steps). The 1000-step runs (labelled
sensitivity, non-paper step budget) raise this only to 1–3/6 serial and
2–3/6 parallel seeds that reach B1, depending on the scaling. The failure is in
optimisation, not expressivity:
- the target is representable;
- the classical model with the same frequencies trains to it with the same
  optimizer;
- the circuit gradients are verified exact.

The published panel is consistent with a favourable draw, roughly one in three
or four. The authors' notebook notes that "some initial weights may not even
converge", but the paper does not say so.

Reading: the paper's expressivity claims hold, but they say nothing about
whether a given circuit and optimizer will reach the expressible function.
Treat Fig. 4 as an existence illustration, not a trainability result.

### Errors and unexplained details

| Location | Issue | Evidence | Impact |
|---|---|---|---|
| Fig. 4 caption | The target is written `Σ c_n e^{2inx}` (input scaling 2), but the plotted target uses scaling 1, as in the authors' notebook. | The published panel's shape (one peak at x ≈ −0.26, f(6) ≈ +0.65) matches only scaling 1; scaling 2 would peak at x ≈ −3.0, 0.1 and 3.2. | None on conclusions: both scalings were run and agree. |
| Eq. 28 | The universal Hamiltonian family `H_m = Σ σ^(i)` has eigenvalue differences in steps of 2, so it reaches only even frequencies. The stated `m = K` needs `H_m = ½ Σ σ^(i)`, the Pauli-rotation generator of Section III. | Direct calculation; with the ½ the construction is exact (E5). | Typo-level. |
| App. A | With `S(x) = e^{−ixH}`, entry `M_jk` multiplies `e^{ix(λ_j − λ_k)}`; the appendix writes `λ_k − λ_j`. | Our construction uses the simulated sign and rebuilds targets to 1e-14. | Relabelling only. |
| Section III.C | The spectrum formula has swapped index labels. | Eq. 9 gives the correct form. | Typo. |
| Fig. 3 loss axis | Rows b and c level off at an "mse" of about 0.02. The model class cannot get below 0.0897, or 0.045 for the notebook's 0.5·MSE. | Pixel reading of the published figure; B1 floor computed on the same grid. | The plotted quantity cannot be the full-grid loss of the plotted fit, and its definition is unknown. The qualitative claim (the fit fails) is reproduced. |

### Mild overstatements and loose statements

| Location | Statement | What we find |
|---|---|---|
| Fig. 5 caption | Ansatz depth l "has little influence" on the coefficient distribution, compared with ansatz type. | Mostly true, but depth changes which coefficients are reachable: for Circuit A, c₄ is exactly zero at l = 1 and nonzero at l = 5. The paper's own panel shows the same. |
| Eq. 12 | `K ≤ d^{2L}/2 − 1`. | Valid in every case tested, but loose (127 against a measured K = 4 for one qubit encoded 4 times). The count of distinct eigenvalue sums gives `K ≤ m(m−1)/2` with `m = d^L`. That bound is reached exactly by the generator `{0, 1, 4, 6}` (K = 6). |

## Photonic Assessment and MerLin Extension

This section is an **adaptation**, not part of the reproduction: the paper
contains no photonic model. Status: **`PARTIAL_MERLIN_TRANSLATION`**.
- **Translated:** the spectrum theory (C1, C4, C6, C7); the fitting and
  trainability experiments; the coefficient distributions; and, after a
  user-requested amendment, the universality theorem (C9) in the
  single-photon setting where its assumptions hold.
- **Not translated:** multivariate and non-integer spectra (C10, C11). They
  share the C1 mechanism already checked exactly in E4.

### Circuit

```text
input |1,…,1,0>  (n single photons in m modes)
mode 0 ─┤      ├─[e^{i k₀x n̂₀}]─┤ ϕ ├─┤      ├─ … ─┤      ├─ detectors
  ⋮     ┤ MESH ├─[e^{i kⱼx n̂ⱼ}]─┤ ϕ ├─┤ MESH ├─ … ─┤ MESH ├─ (threshold, all modes)
mode m−1┤      ├────────────────┤ ϕ ├─┤      ├─ … ─┤      ├─
         W⁽¹⁾     encoding layer ×L     screen + W⁽ˡ⁺¹⁾
```

- **MESH:** MerLin `add_entangling_layer()`, a trainable MZI array.
- **Encoding:** phase `kⱼ·x` on encoded mode j. Its generator is the photon
  number, the photonic version of the paper's encoding Hamiltonian and of its
  continuous-variable phase-shift remark. The degree is the largest
  difference of `Σ kⱼ nⱼ` over the Fock basis, summed over L layers.
  - one encoded mode with k = 1: degree n·L;
  - one photon with kⱼ = j on modes 1..m−1: degree m − 1.
- **ϕ:** an optional trainable phase screen before each later mesh.
- **Readout:** either the fixed `2·P(no click in mode 0) − 1`, the
  counterpart of the paper's `<Z>`, or a trainable linear readout over all
  click-pattern probabilities, i.e. a trainable observable.
- Code: `lib/photonic_merlin.py`.

### MerLin and simulation choices (with evidence)

- **`FOCK`, not the default `UNBUNCHED`.** MerLin 0.2.3's `UNBUNCHED`
  returns the true dynamics' collision-free outcomes, postselected and
  renormalised (verified exactly in `tests/test_photonic_merlin.py`).
  - This is not threshold detection.
  - Its output is a ratio of trigonometric polynomials with unbounded
    spectrum: support up to the grid limit for a predicted degree of 2.
  - Threshold detection is instead emulated exactly by grouping `FOCK`
    outcomes into click patterns.
- **The bare mesh is not universal.** One `add_entangling_layer()` on m modes
  has rank m(m − 1), short of the m² − 1 of a general mode unitary; it lacks
  the input-side phases. A phase screen *before* the mesh restores rank
  m² − 1 (tested).
- **Per-call encoding scales misbehave.** Several `add_angle_encoding` calls
  with non-unit `scale` raise `IndexError` or silently encode a degree-1
  spectrum. Inputs are therefore pre-scaled into a single scale-1 encoding.
  The original Phase 4 runs used only scale 1 and were verified exactly.

### Hardware-Aware Settings

Summary for the original and amendment runs:
- `FOCK` space; threshold detectors (emulated); 1–5 photons in 2–8 modes;
  single-photon inputs in the first n modes; no postselection;
- MerLin 0.2.3 analytic CPU simulator, no shots, exact autograd gradients;
- lossless, noiseless, indistinguishable photons;
- seeds {0, 1, 2, 3, 4, 42}; runs take 2–253 s.

These are simulation results, not hardware validation.

### Results

Protocols: P0 is the paper's (Adam lr 0.3, mini-batches of 25, 200 steps).
P1–P3 are labelled non-paper protocols: full batch, 1000 steps, lr 0.3 / 0.05
/ 0.01. On the Fig. 4 target as published, B1 ≈ 0 for every model below.
Each cell gives seeds reaching B1 out of 6, then the median final MSE.

| Model (Fig. 4 target) | Params | P0 (paper) | P2 | P3 |
|---|---:|---|---|---|
| Photonic n = 2, L = 3, fixed readout (original Phase 4 model) | 24 | 0, 1.2e-2 | 0, 3.4e-3 | 0, 3.6e-3 |
| + universal meshes, fixed readout | 30 | 0, 3.6e-2 | 0, 2.0e-3 | 0, 2.0e-3 |
| + universal meshes, trainable readout | 37 | 0, 1.5e-2 | 6, 4.4e-6 | 6, 4.7e-7 |
| Photonic n = 5, one encoding, universal + trainable | 128 | 0, 4.0e-2 | 4, 3.5e-5 | 6, 1.7e-7 |
| Photonic n = 2, multi-scale (1, 3), universal + trainable | 38 | 0, 1.8e-3 | 6, 1.0e-7 | 6, 1.5e-13 |
| Qubit parallel r = 5, paper's `<Z_0>` | 90 | 2, 2.8e-3 | 6, 1.5e-10 | 6, 7.0e-32 |
| Qubit parallel r = 5, trainable observable | 123 | 0, 8.8e-4 | 6, 3.9e-6 | 6, 1.2e-9 |
| Qubit serial r = 5, `<Z_0>` / trainable observable | 18 / 21 | 0 / 0 | 5 / 6 | 1 / 6 |

Other results:

| Result | Value (label) | Meaning |
|---|---|---|
| Photonic spectrum (9 checks) | Degree n for n = 1..4 photons; 2L for re-uploading; 2 photons on 2 of 4 encoded modes still degree 2; all modes encoded → constant; max stray coefficient 1.4e-16 (exact check) | The spectrum theory transfers exactly. In photonics the resource is photon number, re-uploading or per-mode phase scales, not the number of encoded modes. |
| Universality theorem, one photon in m = 2..8 modes, phase k·x on mode k, trainable observable, square-wave target | Under P3, 6/6 seeds reach the degree-K least-squares floor for every K = 1..7 (0.177, 0.177, 0.087, 0.087, 0.054, 0.054, 0.038); under P0, 0/6 (theorem test; single photon, classically simulable) | C9 holds photonically wherever its assumptions do. |
| Fixed-readout coefficient cap | With the fixed readout, n = 5 photons reach \|c₅\| ≤ 0.0024 against 0.0707 needed (diagnostic). With a trainable readout the same circuit fits the target (P3 6/6). | The cap belongs to the fixed observable, not to the circuit. |

### Conclusions

Confidence HIGH: 6 seeds, consistent across protocols, exact B1 references.

1. The paper's spectrum theory holds exactly for linear optics.
2. The fixed single-mode observable is what limits the photonic models.
   Universal meshes alone do not help. With a trainable observable, the
   paper's practical insight 5, photonic models reach the target as the
   qubit models do.
3. The universality theorem holds photonically in the single-photon setting
   that satisfies its assumptions. This is a theorem test, not a claim of
   quantum value.
4. The paper's training protocol (P0) is a bottleneck for both modalities.
   Full-batch training with a smaller learning rate is what makes the
   expressible models train.
5. With fixed ±1 observables, qubit models outperform the photonic ones.
   With trainable observables, the two are comparable. **No photonic
   advantage is claimed.**

### Artifacts

Figures:
- `results/merlin_amendment_protocols.png`
- `results/merlin_universality.png`
- `results/merlin_mse_vs_degree.png`
- `results/merlin_coefficient_reach.png`

Tables: `results/merlin_*_summary.md`.

## Fair Baselines

**What is compared.** The paper claims no quantum advantage over classical
models. Its comparative statements are about expressivity *within* quantum
models (Figs. 3–5), so the fair control is a classical model with **exactly
the same frequency set**, defined in the Phase 1 plan:
- **B1:** the closed-form least-squares fit, the lowest error any model of
  that class can reach;
- **B2:** the same Fourier model with 2D + 1 trainable coefficients, trained
  with the identical protocol and seeds.

The paper itself has no baseline (state `MISSING_OR_UNFAIR`). B2 was run on
the published Fig. 4 target at degrees 1–6 under:
- the paper's protocol P0 (mini-batch 25, lr 0.3, 200 steps);
- three full-batch protocols P1–P3 (1000 steps; lr 0.3, 0.05, 0.01).

**A disclosed amendment.** B2 was first initialised like the circuit angles
(`U[0, 2π)`), and that failed at lr 0.01. The quantum and photonic trainable
readouts it is compared with are zero-initialised. A zero-initialised B2 was
therefore added, *after* the first result was seen, and both variants are reported. Even this linear model is sensitive
to protocol and initialisation: under P1/P2 both variants reach the exact
floor at every degree; under P0 and P3 one of them fails.

| Claim | Outcome | Evidence (seeds reaching the B1 floor out of 6) |
|---|---|---|
| Single encoding fits degree 1 but not degree 2 or a frequency mismatch (Fig. 3) | **Supported** | Degree 1: 6/6. Mismatch and degree 2: every seed ≥ the floor 0.0897, as for B2 with the same frequencies. The limit is the frequency class. |
| Repetition r = 5 fits the degree-5 target "almost perfectly" (Fig. 4) | **Partially supported** | The target is expressible at r = 5 (B1 = 0). Under the paper's protocol: parallel 2/6, serial 0/6. Under P2/P3: parallel 6/6, serial `<Z_0>` 5/6 (P2). B2 is equally protocol-sensitive, so the gap is the protocol, not the circuits. |
| Ansatz type matters more than depth (Fig. 5) | **Supported (qualitative)** | Internal comparison; the depth nuance (c₄ only at l = 5) is noted. |
| Photonic (MerLin adaptation) vs qubit | **Partially supported; no advantage either way** | Fixed observables: qubit parallel 6/6 at P2/P3, photonic fixed readout 0/6 under every protocol. Trainable observables: both 6/6 at P2/P3 (tie). |
| Quantum or photonic vs classical | **No advantage** (none claimed by the paper) | Under P2, B2 reaches the floor 6/6 (both inits), every qubit model 5–6/6, and every photonic model with a trainable readout 4–6/6; photonic fixed-readout models 0/6. Under P1 the results are mixed (B2 6/6; quantum and photonic models 0–6/6). No model class consistently beats the frequency-matched classical baseline. |

Full table: `results/phase5_comparison.md`, 130 rows from 24 runs, generated
by `utils/compare_phase5.py` from `results/phase5_manifest.json`. Figure:
`results/phase5_comparison.png`.

Timing (local CPU, Apple M5, one run at a time; whole runs of 6 seeds × all
cases):
- B2: 2 s (P0) to 8 s (P1–P3);
- qubit control: 20 s to ~110 s;
- photonic models: 28 s to ~140 s.

Speed is not a claim axis here. Limitations: 6 seeds per cell; synthetic 1D
targets with no held-out split; noiseless photonic simulation; trend-level
comparison with the paper, which reports no numbers.

### Phase 5 extensions (beyond the paper's claims)

Three extensions were declared before they ran. Each carries the
**advantage-evidence** label, because the classical baseline B2 ran under
identical conditions. None supports a quantum or photonic advantage, and
none changes the claim outcomes above.

- **X1: why fixed-readout photonic models stall.** Trained with L-BFGS over
  20 seeds:
  - the qubit models with the paper's `<Z_0>` reach the target 20/20;
  - B2 reaches it 20/20;
  - the photonic models with the fixed single-mode threshold readout reach it
    0/20, plateauing at 1.6e-3 to 4.5e-3.

  Two different photonic circuits share the same best value, 1.59e-3. The
  plateau is a probable limit of that observable class, not of the optimiser
  (strong evidence, not a proof).
- **X2: multi-scale photonic encodings** (square-wave target, degrees 6–9).
  - Encoding x with different phase scales on two modes lets 2–3 photons in
    4 modes reach degree 6–9 with about 40 parameters. A single photon needs
    D + 1 modes and up to 200 parameters for the same degree.
  - Both reach the degree-D floor in most seeds: 4–5/6 for multi-photon,
    5–6/6 for single photon.
  - B2 reaches it 6/6 with 2D + 1 parameters, the minimal parametrisation of
    the class.
- **X3: better optimisers.** With L-BFGS, every model whose class contains
  the Fig. 4 target reaches it in 6/6 seeds: qubit (both readouts), photonic
  with a trainable readout, and classical. Cosine-decayed Adam does nearly as
  well (qubit serial `<Z_0>` 3/6). With 6 seeds per cell, this suggests
  that the paper's training protocol, rather than the circuits, was the main
  obstacle to reliable fits.

Tables: `results/phase5_extensions.md`. Figure:
`results/phase5_extensions.png`. Run manifest:
`results/phase5_extensions_manifest.json`.

## Limitations

- Synthetic one-dimensional targets, no held-out split: trainability is
  measured on the training grid, and no generalisation claim is tested (the
  paper makes none).
- The paper reports no numbers, so every comparison with it is trend-level.
- Six seeds per cell (20 for two diagnostics); no statistical-superiority
  statement is made.
- All photonic results are noiseless MerLin 0.2.3 simulation: no loss, no
  partial distinguishability, no shots, no hardware.
- Single-photon photonic models (the universality test) are classically
  simulable; they test the theorem, not quantum value.
- Trainable readouts add classical parameters and change the model class
  relative to the paper's fixed `<Z>`; both are reported.
- Non-paper protocols P1–P5 were chosen to separate training from
  expressivity; they are labelled and never replace the paper protocol.
- The fixed-readout photonic plateau is strong evidence of a model-class
  limit, not a proof.

## Extensions and Next Steps

- Noise and finite shots for the photonic models (loss, distinguishability).
- A scaling study of multi-scale encodings beyond degree 9, with more photons.
- Re-run with MerLin 0.4.x in a separate environment to see whether the mesh
  and encoding-scale issues reported to the MerLin team persist.
- A held-out-data task, to test whether higher degree helps or hurts
  generalisation (the paper's open question).

## Reproducibility Notes

- All randomness is seeded: the initial angles and the mini-batch sampler
  share one seed per fit; global RNGs are seeded by the shared runtime.
- Precision: `dtype` is `float64` (`complex128` for statevectors).
- Runs are deterministic on one machine; late-training values can differ at
  the ~1e-1 relative level after floating-point re-ordering (a regression
  check changed 9% of one case's final MSE, with identical conclusions).
- Every run writes `run.log`, `run_status.json`, `config_snapshot.json` and
  `metrics.json`, validated by the workflow's logging validator.
- Library versions: `requirements.txt`; the environment used is listed above.

## Tests

Run tests from inside the `papers/data_encoding_expressivity/` directory:

```bash
cd papers/data_encoding_expressivity
pytest -q
```

The tests cover:
- the target generator (the seven tests from the data specification);
- the statevector simulator against an independent dense-matrix build;
- autograd against parameter-shift gradients;
- predicted spectra and the size bounds;
- the universality construction;
- the MerLin model (spectrum, universal meshes, readouts, MerLin `UNBUNCHED` semantics);
- the optimisers (cosine Adam, L-BFGS) and the classical baseline;
- one tiny end-to-end run per experiment kind, plus the failure path.

The suite has 145 tests and runs in about 10 s on a laptop CPU.

## Citation and License

Cite the original paper: M. Schuld, R. Sweke and J. J. Meyer, *The effect of
data encoding on the expressive power of variational quantum machine learning
models*, Phys. Rev. A **103**, 032430 (2021), arXiv:2008.08605.

The authors' repository (https://github.com/XanaduAI/expressive_power_of_quantum_models)
has no license file; it was consulted for settings only and is not
redistributed. The code here is released under the repository's MIT license.
