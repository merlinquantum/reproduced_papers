# The trainability of photonic quantum circuits — Reproduction

Status: reproduction complete at the scope below. The analytic and numerical
variance results reproduce quantitatively. The training demonstrations are
reproduced at reduced size or as simulated substitutes. A MerLin hardware-aware
adaptation and a fair classical comparison were added. `LOG.md` holds the plans,
decisions, run ledgers and phase audits; `INSIGHTS.md` holds the durable
lessons.

## Reference and Attribution

- Paper: *The trainability of photonic quantum circuits*, arXiv preprint, 2026.
- Authors: Alexander Makarovskiy, Adam Taylor, Zhenghao Li, Michael Hanks,
  Aubrey Clark, M. S. Kim, Ian Walmsley, William Clements (ORCA Computing,
  Imperial College London, University of Oxford).
- arXiv: [2607.21544](https://arxiv.org/abs/2607.21544) (v1, 23 July 2026,
  quant-ph).
- Original repository: none released (paper, arXiv, Zenodo, Hugging Face and
  GitHub searched on 2026-10-19).
- Reused reference material: the second-moment framework of Mhiri et al.,
  [arXiv:2604.14323](https://arxiv.org/abs/2604.14323), and their MIT-licensed
  [code](https://github.com/quantumsoftwarelab/Representation_Theory_Framework-Passive_Linear_Optics),
  used only as a cross-check. The classical estimator of Lim and Oh,
  [arXiv:2502.12882](https://arxiv.org/abs/2502.12882), is reimplemented from
  their paper.
- All scientific content of the paper belongs to its authors. This folder is an
  independent reimplementation; cite the paper when using it.

## Original Paper

Variational photonic circuits are trained by estimating a loss from a finite
number of measurement shots. The paper asks when that is efficient for passive
linear optics (Haar-random interferometers on M modes, Fock inputs with n
photons, photon-number-resolving detection). It introduces two quantities:

- the sample variance, `Var_samp = E_U[Var(O | U)]`, which sets the shot noise;
- the circuit variance, `Var_circ = Var_U[<O>_U]`, which sets the size of the
  loss differences that must be resolved.

The ratio `Var_samp / Var_circ` sets the number of shots needed. Unlike qubit
observables, photon-number observables are unbounded, so the sample variance
can grow even without a barren plateau.

Using exact Haar moments (a closed form for the first moment and an irrep
decomposition for the second), the paper claims:

- fixed-order photon-number monomials and polynomials (PNMs, PNPs) at constant
  photon density are trainable, because their circuit variance decays only
  polynomially (a numerical conjecture);
- PNMs whose order grows with M, single-outcome projectors and coarse-grained
  output bins need exponentially many shots;
- for balanced-sign PNPs with |A| terms, quantum sampling needs O(|A|) shots
  where term-wise classical estimation (Lim-Oh) needs O(|A|^2);
- neural networks with quadratic activations implement high-order PNPs
  efficiently.

Training is demonstrated in simulation (Fig. 7: 20 and 26 modes) and on ORCA's
PT-2 photonic processor (Fig. 9: 48 time-bin modes, 6 post-selected photons).

## Reproduction Scope, Claims, and Deviations

Thirteen claims (C1-C13) and thirteen experiments (E1-E13) are inventoried in
`LOG.md`. Final status per claim:

| Claim | Statement (short) | Experiments | Outcome |
|---|---|---|---|
| C1 | `N ~ Var_samp / Var_circ` shots resolve losses and gradients | E10 | supported (Prop. 1 bound vacuous at tested N) |
| C2 | Exact Haar first moment; factorial growth with order | E1, E2 | supported, quantitative |
| C3 | Fixed-order circuit variance decays polynomially | E3, E4, extension | supported; decay ~ M^-1 (nu=0.5) and ~ M^-3 (nu=1) |
| C4 | Fixed-order sample variance tends to `mu1(2 alpha)`; ratio poly(M) | E3, E7 | supported |
| C5 | Order-growing PNMs need exponentially many shots | E7 | conclusion supported; the Eq. 16 bound fails at tested sizes |
| C6 | Random fixed-order PNPs stay trainable | E5 | supported; Fig. 5c,d after diagnosing the term-selection scheme |
| C7 | Projectors: ratio exponential in system size | E6 | supported (Eq. 23 wording inconsistent with its source) |
| C8 | Coarse-grained bins concentrate | E6 | supported |
| C9 | Quadratic quantum speed-up over term-wise Lim-Oh | E13 (Phase 05) | partially supported |
| C10 | Quadratic NN is a PNP of order 2^L | E12 | supported, exact |
| C11 | Shot-based GPSR trains (Fig. 7) | E8 | supported at reduced size (M = 8, 12) |
| C12 | PT-2 hardware trains (Fig. 9) | E9 | partially supported (16-mode simulated substitute) |
| C13 | Loss concentration ⇔ gradient barren plateaus (Thm. 2) | E11 | unresolved (E11 not run) |

Deviations and assumptions (details in `LOG.md`, "Reproducibility Assessment"
and "Scientific Ambiguity Resolution"):

- Second moments follow Mhiri et al. (Cor. 1) in exact rational arithmetic,
  truncated at irreps `k <= order`. This truncation is exact for PNMs, and the
  tests check it. It makes the paper's M ranges compute in seconds.
- Fig. 5 input density is not stated; nu = 0.5 matches Fig. 5a,b. The Fig. 5c,d
  term-selection scheme is not stated; it was diagnosed (see Results).
- Fig. 3c,d at nu = 1 uses the input `|1, ..., 1>` (inferred).
- NN observable: `L` squaring hidden layers plus a linear read-out. This is the
  only reading consistent with "order 8 for 3 layers" and with the negative
  losses of Fig. 7. The output is rescaled to unit standard deviation (an
  assumption).
- Haar initialisation uses an exact Clements decomposition of a Haar unitary
  (the paper cites the Haar-dialling measure instead).
- Fig. 7 runs at M = 8 and 12 instead of 20 and 26. MerLin computes full output
  distributions, and the paper's sizes need 2e7-dimensional states.
- Fig. 9 is a simulated two-loop processor with 16 modes, 6 Fock photons and
  one-bin delays (assumed). It is a substitute for PT-2 hardware with heralded
  squeezed light.
- The paper's "N samples per gradient estimate" was run as N per shifted loss;
  the per-gradient-component reading gives the same conclusion.

## Project Layout

```text
papers/photonic_trainability/
├── README.md, LOG.md, INSIGHTS.md, FEEDBACK.md, CONFLUENCE.md, VISITED_URLS.md
├── cli.json, requirements.txt, implementation.py
├── notebook.ipynb      guided walkthrough of the claims and evidence (start here)
├── configs/            one JSON per experiment and variant (see Configuration)
├── lib/
│   ├── runner.py              entry point, dispatch on cfg["experiment"]
│   ├── moments.py             exact Haar moments (Thm. 1, Mhiri Cor. 1, P2)
│   ├── pnp_moments.py         fast exact moments of large PNPs
│   ├── combinatorics.py, fock.py, observables.py, specs.py, fits.py
│   ├── photonic_sim.py, haar_mc.py, data.py       Haar sampling, Perceval SLOS, Monte Carlo
│   ├── clements.py, training.py                   Clements mesh, MerLin forward model, GPSR
│   ├── correlators.py, lim_oh.py                  baselines B2 (exact) and B1 (Lim-Oh)
│   ├── merlin_hardware.py                         Phase 04 hardware conditions
│   ├── exp_*.py                                   one module per experiment type
│   └── experiment_logging.py
├── utils/              plot_results.py, run_lr_sweep.py, summarize_*.py, curate_results.py, build_notebook.py
├── tests/              87 tests (exact identities, Monte Carlo, GPSR, baselines, smoke runs)
├── results/            curated figures and CSV tables referenced below
└── outdir/             raw run directories (git-ignored)
```

## Install and How to Run

```bash
pip install -r papers/photonic_trainability/requirements.txt
```

Smoke runs (seconds), from the repository root:

```bash
python implementation.py --paper photonic_trainability --help
python implementation.py --paper photonic_trainability --config configs/defaults.json
python implementation.py --paper photonic_trainability --config configs/smoke_training.json
```

From inside the paper folder: `python ../../implementation.py --config configs/defaults.json`.

Paper experiments (each a few seconds to a few minutes on a laptop CPU, except
Fig. 7 at M = 12, about 17 min):

```bash
python implementation.py --paper photonic_trainability --config configs/validate_moments.json
python implementation.py --paper photonic_trainability --config configs/fig10_first_moment_original.json
python implementation.py --paper photonic_trainability --config configs/fig3_pnm_variances_original.json
python implementation.py --paper photonic_trainability --config configs/fig4_single_mode_original.json
python implementation.py --paper photonic_trainability --config configs/fig5ab_pnp_variances_original.json
python implementation.py --paper photonic_trainability --config configs/fig5cd_pnp_variances_uniform_pattern.json
python implementation.py --paper photonic_trainability --config configs/output_probability_original.json
python implementation.py --paper photonic_trainability --config configs/sample_complexity_original.json
python implementation.py --paper photonic_trainability --config configs/fig7_nn_training_reduced.json
python implementation.py --paper photonic_trainability --config configs/fig9_loop_training_substitute.json
python implementation.py --paper photonic_trainability --config configs/estimator_scaling_classical.json
python implementation.py --paper photonic_trainability --config configs/merlin_variance_hardware.json
```

Learning-rate selection as a coordinated sweep (one per training config):

```bash
python papers/photonic_trainability/utils/run_lr_sweep.py --config configs/fig7_nn_training_reduced.json --sweep-id <id>
```

Each run writes `outdir/run_YYYYMMDD-HHMMSS/` (or the `--outdir` given) with
`run.log`, `run_status.json`, `config_snapshot.json`, `metrics.json` and CSV
tables. Plot a run:

```bash
python papers/photonic_trainability/utils/plot_results.py <run dir> --name <name> --label "<scope label>"
```

The notebook `notebook.ipynb` walks through every claim with live exact
computations, small live smoke runs and the curated stored results in
`results/` (provenance in `results/PROVENANCE.md`). It runs in about 1 minute on
a laptop CPU:

```bash
cd papers/photonic_trainability
python utils/curate_results.py   # refresh results/ tables from outdir/ runs (only needed after new runs)
python utils/build_notebook.py   # regenerate the notebook source
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

## Configuration

- `configs/defaults.json`: runnable smoke default (two cases of the exact
  versus Monte Carlo validation).
- `*_original.json`: paper settings. `*_reduced.json`: reduced size.
  `*_substitute.json`: hardware substitute. `*_positive.json`,
  `*_pergradient.json`, `*_uniform_pattern.json`, `*_order_then_partition.json`:
  controls, ambiguity variants and gap-diagnosis variants.
  `variance_scaling_extended_*.json`: extension to M = 2000.
  `merlin_*.json`: Phase 04 adaptation. `estimator_scaling_*.json`: Phase 05
  comparison. `smoke_*`, `*_smoke.json`: smoke runs.
- `experiment` selects the module (`validate_moments`, `first_moment`,
  `variance_scaling`, `pnp_variances`, `output_probability`,
  `sample_complexity`, `network_polynomial`, `training`, `training_hardware`,
  `merlin_variance`, `estimator_scaling`). Missing required keys raise
  `KeyError`.
- `cli.json` declares no paper-specific flags. The shared runtime injects
  `--config`, `--outdir`, `--seed`, `--dtype`, `--device`, `--log-level` and
  `--data-root`. `--device` reaches MerLin; the exact-moment code is CPU integer
  arithmetic.

## Data

The paper uses no external dataset, and none exists (searched 2026-10-19; see
`VISITED_URLS.md`). Every input is synthetic by the paper's own protocol and is
regenerated from the config seed:

- Haar-random interferometers (QR of a Ginibre matrix with phase correction),
  served in seeded batches by `lib/data.py::build_dataloader`.
- Fock inputs `|1,0,1,0,...>` (nu = 0.5), `|1,...,1>` (nu = 1) and `|2,...,2>`
  (nu = 2).
- Random observables: PNM sets, PNPs with U[-1, 1] (or U[0, 1]) coefficients,
  quadratic NNs with Gaussian weights.

Nothing is downloaded to run the reproduction, and there is no
`data/photonic_trainability/<dataset>/` folder. The ORCA PT-2 measurements
behind Fig. 9 are not published.

Reference material is cached (git-ignored) under
`data/photonic_trainability/sources/`: the paper PDF, Mhiri et al. (PDF and code
at commit `79de1ce`) and Lim-Oh (PDF). Recreate it with:

```bash
curl -L -o data/photonic_trainability/sources/paper_main_v1.pdf https://arxiv.org/pdf/2607.21544v1
git clone https://github.com/quantumsoftwarelab/Representation_Theory_Framework-Passive_Linear_Optics \
  data/photonic_trainability/sources/mhiri2026_repo_snapshot
git -C data/photonic_trainability/sources/mhiri2026_repo_snapshot checkout 79de1ce8e4275edda3200a884775a4aa6c941574
```

## Results Obtained and Comparison with the Paper

How to read this section:

- Paper values are read off the paper's plots, which are mostly log scale
  (about ±20% reading accuracy); the paper prints no tables.
- Δ is reproduced / paper.
- "Exact" results are deterministic rational arithmetic, so seeds do not apply.
  Monte Carlo results carry batch-means errors. "± std" is over instances or
  seeds as stated.
- Run directories and commands are in `LOG.md` (Phase 03 Run Log).

### Main results table

| Paper item | Claim tested | Paper value | Reproduced value | Δ | Metric agreement | Trend agreement | Claim support | Seeds | Comment |
|---|---|---:|---:|---:|---|---|---|---:|---|
| Fig. 10, mu1 at weight 25, M = 50, nu = 0.5 / 1 / 2 | C2 | ~1e5 / ~1e11 / ~1e17 | 1.7e5 / 9.2e10 / 1.9e17 | 1.7 / 0.9 / 1.9 | within reading accuracy (≤0.3 decade) | yes (superexponential) | supported | n/a (exact) | `results/fig10_first_moment.png` |
| Fig. 10, M → ∞ limit at weight 25 | C2 | ~1e8 / ~1e13.5 / ~1e19.5 | 2.0e8 / 4.6e13 / 5.9e19 | ≈1.4-2 | within reading accuracy | yes | supported | n/a (exact) | Fubini product, Eq. 95 |
| Fig. 3a, Var_samp [0,0,6], M = 250, nu = 0.5 | C4 | ~1e8 | 6.93e7 (limit 9.40e7) | 0.69 | yes | yes (→ limit) | supported | n/a (exact) | `results/fig3ab_nu0p5_variance_scaling.png` |
| Fig. 3b, Var_circ [0,0,6] / [2,2,2], M = 250 | C3 | ~10^3.5 / ~2e-2 | 3175 / 0.0213 | 1.00 / 1.07 | quantitative | yes (peak at M 40-60) | supported | n/a (exact) | |
| Fig. 3c, Var_samp [0,0,6], M = 200, nu = 1 | C4 | ~1e10 | 1.97e10 | ≈2 | within reading accuracy | yes | supported | n/a (exact) | input `|1,...,1>` inferred |
| Fig. 3d, Var_circ [0,0,6], M = 200, nu = 1 | C3 | ~10^2.7 | 602 | ≈1.2 | yes | yes | supported | n/a (exact) | `results/fig3cd_nu1_variance_scaling.png` |
| Fig. 4, Var_circ of n^2 at M = 10 / 340 | C3 | ~0.128 / ~0.008 | 0.1283 / 0.0065 | 1.00 / 0.81 | quantitative | yes | supported | n/a (exact) | `results/fig4_variance_scaling.png` |
| Fig. 4, peak of n^8 / n^10 | C3 | ~1.75e7 at M~70 / ~1.0e11 at M~110 | 1.78e7 at 70 / 9.98e10 at 110 | 1.02 / 1.00 | quantitative | yes | supported | n/a (exact) | |
| Fig. 5a,b, all PNMs with |alpha| <= 2, M = 90 | C6 | ~1400 / ~2.5 | 1403 ± 90 / 2.54 ± 0.34 | 1.00 / 1.02 | quantitative | yes (quadratic / linear) | supported | 20 PNPs per M | `results/fig5ab_pnp_variances.png` |
| Fig. 5c,d, 5M PNMs of order <= 4, M = 60, monomials drawn uniformly | C6 | ~48,000 / ~65 | 322 ± 396 / 0.27 ± 0.49 | 0.007 / 0.004 | no | partial | n/a | 20 PNPs per M | first attempt; gap diagnosed below |
| Fig. 5c,d, same, exponent pattern drawn uniformly | C6 | ~48,000 / ~65 | 56,190 ± 10,800 / 72.5 ± 14.6 | 1.17 / 1.12 | within reading accuracy | yes | supported | 20 PNPs per M | `results/fig5cd_uniform_pattern_pnp_variances.png` |
| Fig. 5c,d, same, order then partition | C6 | ~48,000 / ~65 | 38,130 ± 9,950 / 50.9 ± 13.0 | 0.79 / 0.78 | within reading accuracy | yes | supported | 20 PNPs per M | alternative reading of the scheme |
| Eqs. 21-22, projector variances, M <= 10 | C7 | analytic | exact equality in all 6 cases | 0 | exact | ratio 9.5 → 1095 as |H| 10 → 2002 | supported | n/a (exact) | `results/eq21_26_projectors_projector_ratio.png` |
| Eq. 25 / Eq. 152, coarse-grained bins | C8 | analytic | Eq. 25 exact for every bin; bin average 2.15e-4 vs Eq. 152 2.28e-4 (M = 10) | 0.94 | yes | yes | supported | 3 random bins per case | structured-bin control escapes concentration |
| Eqs. 15-16, (1,...,1) with K = M/2 | C5 | ratio Ω(2^K) | ratio 9.9 → 3.5e5 for M 4 → 16 (fixed [1,1]: 9.9 → 107) | n/a | n/a | yes (exponential) | supported (bound fails) | n/a (exact) | Eq. 16 bound violated at all densities |
| Prop. 1, ordering success at N = ratio / 3 × ratio | C1 | guarantee ≥ 1 - 2/t - 2δ | 0.78-0.89 / 0.95-0.97 | n/a | n/a | collapses in N/ratio | supported | 2,000 Haar pairs | guarantee vacuous at these N |
| Eq. 144, GPSR gradient variance | C1 | (2R^2+1)/6 · Var_samp/N | measured / predicted = 1.02 | 1.02 | yes | yes | supported | 200 Haar meshes | |
| Eq. 20, NN term count (M = 3, L = 3) | C10 | C(2^L+M, M) = 165 | 165 non-zero terms, degree 8 | 1.00 | exact | yes | supported | n/a (exact) | 35/70 without biases |
| Fig. 7, GPSR, 10,000 shots per shifted loss | C11 | M = 20: loss -0.134 → -0.1425 | M = 8: improvement 0.160 ± 0.106 (shots) vs 0.160 ± 0.105 (exact); M = 12: 1.32 ± 0.10 vs 1.69 ± 0.03 | n/a | n/a (different observable, size) | yes (smooth decrease) | supported at reduced size | 3 | partial RED-tier run; `results/fig7_m*_training_*.png` |
| Fig. 9, parameter shift, 1000 samples per gradient | C12 | PT-2: loss -2 → -19 | 16-mode substitute: 30.5 ± 4.5 (shots) vs 31.5 ± 4.1 (exact) | n/a | n/a | yes | partially supported | 3 | substitute; `results/fig9_substitute_training_*.png` |

### Notes on the main results

**Decay of the circuit variance (C3).** On the paper's M range, a least-squares
exponential fits the decaying branch better than a power law for all seven
Fig. 3a PNMs and for n^8 and n^10 in Fig. 4 (AIC difference -37 to -69). That is
pre-asymptotic. Extending the exact computation to M = 2000 (labelled
extension; `results/extended_*`) shows the local slope d log Var_circ / d log M
settling to a constant:

| Input | n^2 | [0,0,6] | n^10 | [0,1,5] | [2,2,2] |
|---|---|---|---|---|---|
| nu = 0.5, slope at M = 1400-2000 | -0.996 | -0.973 | -0.933 | -0.968 | -0.965 |
| nu = 1, slope at M = 1400-2000 | -2.995 | -2.966 | -2.918 | -2.963 | -2.961 |

So Var_circ ~ M^-1 for the alternating input and ~ M^-3 for the uniform input:
polynomial, as conjectured, with an input-dependent exponent the paper does not
report. Var_samp reaches 90-96% of mu1(2 alpha) at M = 2000.

**Fig. 5c,d gap.** The paper does not say how the 5M monomials are drawn.
Drawing monomials uniformly almost never selects single-mode powers such as
n^4 at large M, and those terms dominate both variances. With zero-mean
coefficients the expected variances decompose exactly into per-pattern terms
(Eqs. 18-19 hold with equality). That decomposition predicted, and full runs
then confirmed under a predeclared factor-1.5 test, that drawing the exponent
pattern uniformly reproduces the paper's magnitudes. The known-answer check
(exact against Monte Carlo on an instance) ruled out a code error. Details are
in `LOG.md`, "Gap Diagnosis: Fig. 5c,d".

**Eq. 16.** The lower bound Var_samp ≥ (2^K - 1) nu^{2K} for order-growing
monomials does not hold at any density tested (e.g. nu = 2, M = 16: bound
1.7e7, exact 5.6e4). It is derived from a fixed-K, M → ∞ limit. The
exponential growth of the ratio, which is the paper's conclusion, still holds.

**Eq. 23.** The paper writes `P2 = Θ(n + n/M + 1)`, where its source (Mhiri et
al.) gives `M/n + 1` in the saturated regime. Exact Eqs. 21-22 are unaffected.

**Ambiguity.** The paper's "N samples per gradient estimate" can mean per
shifted loss or per gradient component. Under the second reading (625 and 250
shots per shifted loss, same learning rates), shot-based training still matches
exact-gradient training:

- Fig. 7, M = 8: 0.161 ± 0.107 (shots) vs 0.160 ± 0.105 (exact).
- Fig. 9 substitute: 29.2 ± 6.9 (shots) vs 31.5 ± 4.1 (exact).

**Training losses.** Absolute training losses are not comparable with the paper
(different random observables, a rescaled NN output, different sizes). The
reproduced claim is the qualitative one.

### Learning-rate selection sweeps

Three coordinated sweeps (`outdir/sweeps/lrsel2_*`, coordinator
`utils/run_lr_sweep.py`) selected the training learning rates. The full
candidate table is `results/lr_selection_summary.csv`; it is generated from
each candidate's `metrics.json`.

- Rule: lowest final exact loss of `gpsr_exact` on the validation seed 100;
  direction minimize; checkpoint last iteration; one repetition per candidate;
  ties within 1e-12 go to the smaller rate.
- Candidates: {0.01, 0.03, 0.1, 0.3}. Every sweep is a `COMPLETE SWEEP` with 4/4
  runs, and no ties occurred.

| Sweep | lr 0.01 | lr 0.03 | lr 0.1 | lr 0.3 | Selected |
|---|---:|---:|---:|---:|---|
| Fig. 7, M = 8 | -0.0305 | -0.0360 | **-1.6625** | -1.5684 | 0.1 |
| Fig. 7, M = 12 | 0.0628 | 0.0470 | 0.0339 | **-0.1188** | 0.3 (grid edge) |
| Fig. 9 substitute | -38.92 | **-40.13** | -12.06 | 4.77 | 0.03 |

With one repetition per candidate, the selections are tuning choices, not
evidence that one rate is statistically better. At M = 12 the largest rate won,
so a larger one might train faster; the grid was not extended after the fact.

## Fair Baselines

Only one claim is quantum versus classical: C9 (App. D.3-D.4). For balanced-sign
fixed-order PNPs, the paper says quantum sampling needs O(|A|) shots to
estimate the loss to additive error eps, while term-wise Lim-Oh estimation
needs O(|A|^2) samples. The paper argues this analytically and runs no
comparison (baseline state `MISSING_OR_UNFAIR`), so two fair baselines planned
in Phase 01 were implemented:

- **B1:** the Lim-Oh estimator (their Theorem 1), applied term by term with the
  variance-optimal (Neyman) sample split.
- **B2:** exact classical evaluation from permanents.

Both use the same Haar interferometers, Fock input and polynomials as the
quantum side, and the same N = Var / eps^2 rule. The protocol was frozen in
`LOG.md` ("Phase 5 Comparison Plan") before any run.

Scope: M = 16, n = 8, PNPs drawn from the 152 PNMs of order <= 2,
|A| = 10-150, 3 Haar unitaries × 5 draws, simulator, Apple M2 CPU.

| |A| (balanced) | Quantum shots | B1 Lim-Oh samples | B1 classical time (projected) | B2 exact time (measured) |
|---:|---:|---:|---:|---:|
| 10 | 342 | 1.8e7 | 6.9 s | 3.3 ms |
| 40 | 1,190 | 3.3e8 | 129 s | 12 ms |
| 150 | 5,480 | 4.2e9 | 1,620 s | 46 ms |

| Coefficients | Slope of quantum shots vs |A| | Slope of Lim-Oh samples vs |A| | Predeclared outcome vs Lim-Oh |
|---|---:|---:|---|
| balanced, U[-1, 1] | 1.07 ± 0.27 | 2.02 ± 0.07 | separation supported |
| positive, U[0, 1] | 0.53 ± 0.08 | 2.05 ± 0.12 | separation supported |

The tables show means over 15 cases at eps = 0.1 and slopes as mean ± std over
3 unitaries. The B1 time is projected: measured seconds per Lim-Oh sample
(3.9e-7 s) times the samples required. Full tables are in `results/comparison_c9*.csv`; the figure is
`results/comparison_c9_estimator_scaling.png`. Lim-Oh is unbiased against exact
expectations (max |z| 3.3), and B2 matches the exact loss to 2e-14.

**C9 outcome: partially supported.**

- The ~|A| versus ~|A|^2 separation from Lim-Oh is real. Its large constant
  gap comes mostly from truncated per-term norms (sum_k k^4 = 8,772 for n^2 at
  n = 8), which the paper treats as O(1).
- The stated mechanism, sign cancellation, does not drive it. Positive
  coefficients scale better (|A|^0.53), because photon-number conservation
  already suppresses their variance.
- Exact classical evaluation returns the exact loss about 3.5e4 times faster
  than Lim-Oh, at a cost linear in |A| for fixed order. So there is no speed-up
  over the best classical method for fixed-order PNPs; the paper notes these
  are classically efficient.
- The marginal-correlator emulators (B3, Gaussian-input methods) and the NN
  regime with exponentially many expanded terms were not tested.

**Training controls.** C11 and C12 have no comparator in the paper. Against the
Phase 01-planned controls (exact-gradient GPSR and equal-shot SPSA; table in
`results/comparison_training.csv`):

- Shot-based GPSR reaches 1.00× (M = 8) and 0.78× (M = 12) of the exact-gradient
  improvement.
- It beats SPSA: 1.32 vs 0.23 at M = 12 and 30.5 vs 4.8 for the Fig. 9
  substitute. At M = 8 the gap (0.160 vs 0.148) is within one std.

**No quantum-advantage claim follows from these results.**

## Photonic Assessment and MerLin Extension

Status: **`MERLIN_IMPLEMENTATION`**, an adaptation and not part of the
reproduction. The paper is natively photonic, and the reproduction already uses
MerLin (`QuantumLayer` over the Clements mesh) as the forward model of every
training experiment. The adaptation tests what the paper leaves open: do the
trainability conclusions survive realistic photonic hardware? It keeps the
paper's model and adds one effect at a time:

- partial photon distinguishability (`NoiseModel(indistinguishability=0.9, 0.7)`);
- uniform loss (`transmittance = 0.9, 0.7`), with and without post-selection on
  n photons;
- threshold detection (`pcvl.Detector.threshold()` via `pcvl.Experiment`).

Considered and rejected: `ComputationSpace.UNBUNCHED` as a stand-in for
threshold detection, because it drops bunched events rather than modelling
detectors. Not modelled: PT-2's heralded squeezed-light input. Plan, amendment,
commands and per-observable factors are in `LOG.md` (Phase 4 sections).

Results (reduced-scope validation, simulator; `results/merlin_hardware_hardware_variance_ratio.png`):

- The ideal MerLin pipeline reproduces the exact Haar moments (36 z-scores,
  max |z| 2.21).
- Variance ratio relative to ideal for n0^2 at M = 4 / 6 / 8:

  | Condition | Ratio relative to ideal |
  |---|---|
  | indistinguishability 0.7 | 1.01 / 0.99 / 0.97 |
  | post-selected loss 0.7 | exactly 1 |
  | unpost-selected loss 0.7 | 1.68 / 1.54 / 1.46 |
  | threshold detection | 1.03 / 1.15 / 1.06 |

  Post-selected loss gives exactly the ideal statistics because uniform loss
  commutes with linear optics; the cost is the t^n accepted-shot fraction. In
  the figure, the ideal and both post-selected-loss curves coincide. Results
  for n0 n1 and a random PNP are in `LOG.md`.
- Threshold detection bounds the observable (Var_samp of n0^2 at M = 8 drops
  from 3.28 to 0.22), but it shrinks the circuit variance by a similar factor.
  The variance ratio, and with it the trainability diagnosis, barely changes.
- Every condition met the predeclared criterion: the ratio stays within a
  factor 3 of ideal, and its M = 4 → 8 growth stays within a factor 2.
- Training at M = 6 over 3 seeds: shot-based GPSR reaches 0.994-0.999 of the
  exact-gradient improvement under every condition.

## Hardware-Aware Settings (when MerLin was executed)

| Field | Variance study (`configs/merlin_variance_hardware.json`) | Training (`configs/merlin_training_hardware_*.json`) |
|---|---|---|
| Computation space | `FOCK` (needed for PNR observables and source noise) | `FOCK` |
| Detector model | PNR; threshold in the `threshold_detection` condition | same |
| Photon number | n = M/2 (2, 3, 4); fewer detected under unpost-selected loss | 3 |
| Number of modes | 4, 6, 8 | 6 |
| Input state | `|1,0,1,0,...>` | `|1,0,1,0,1,0>` |
| Encoding | none; Haar-random mesh phases as `input_parameters` | none; trained mesh phases |
| Measurement strategy | `MeasurementStrategy.probs(FOCK)` | same |
| Post-selection | n detected photons in `loss_*_postselected`; otherwise none | none |
| Simulator / QPU | MerLin 0.4.1 CPU simulator, analytic | same; shots by multinomial sampling |
| Shot count | n/a (exact distributions, Monte Carlo over 2,000 Haar draws) | 10,000 per shifted loss; 1.8e6 per iteration |
| Gradients | n/a | GPSR (R = 3), exact under each hardware effect (tested against autograd) |
| Wall-clock | 7 s for the whole study | 4-5 s per run including start-up (1-2 s of experiment time) |
| Seeds | Haar draws 41, observable 3 | 0, 1, 2 |

## Limitations

- Training demonstrations run at reduced size (Fig. 7: M = 8 and 12 instead of
  20 and 26) and as a simulated substitute (Fig. 9: 16 modes, Fock input, no
  hardware). MerLin's full-distribution simulation limits the size.
- Two unstated details were resolved by diagnosis and cannot be confirmed: the
  Fig. 5c,d term-selection scheme (two readings fit) and the nu = 1 input of
  Fig. 3c,d.
- E7 (order-growing monomials), E6 (projectors, bins) and E10 (sample
  complexity) use small systems (M ≤ 16), so asymptotic statements rest on
  trends.
- The MerLin hardware study uses representative noise values at M ≤ 8. It
  omits phase noise, g2, squeezed-light inputs and pseudo-PNR detection.
- C9 was tested at one size (M = 16, order ≤ 2) against Lim-Oh and exact
  evaluation only. Emulators and the NN regime were not tested.
- C13 (Theorem 2) is untested.
- The Fig. 7 M = 12 learning rate sits at the edge of the predeclared grid.

## Tests

```bash
cd papers/photonic_trainability
pytest -q
```

87 tests cover:

- exact identities and explicit-matrix checks of the moment code;
- the agreement between Mhiri's P2 formula and summed projector second moments;
- Perceval, Ryser and MerLin probability agreement;
- the Clements decomposition;
- GPSR against autograd and finite differences, including under hardware
  noise;
- the fast PNP paths;
- the Lim-Oh estimator's unbiasedness and variance bound;
- Neyman allocation;
- tiny end-to-end runs of every experiment type.

## Citation and License

```bibtex
@misc{makarovskiy2026trainability,
  title         = {The trainability of photonic quantum circuits},
  author        = {Makarovskiy, Alexander and Taylor, Adam and Li, Zhenghao and
                   Hanks, Michael and Clark, Aubrey and Kim, M. S. and
                   Walmsley, Ian and Clements, William},
  year          = {2026},
  eprint        = {2607.21544},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph}
}
```

Code in this folder is released under the repository license (MIT). Mhiri et
al.'s code is not copied here; it is cached locally under its own MIT license
for cross-checking only.
