# Insights — QNN hyperparameter study (Herbst et al., QCE 2024)

Durable findings for future reproductions. The evidence is in `README.md` and
`LOG.md`.

## Scientific findings

- **Check that a hyperparameter is actually applied before testing its effect.**
  The paper's initialization arms never reach the optimizer: the
  `assign_parameters` return value is discarded and `initial_point` is unset.
  Any "init effect" in such code comes from whatever else differs between
  arms.
- **Input range dominates angle-encoded classifiers.** Feeding raw PCA/LDA
  scores (range about ±6) as rotation angles wraps them several times over.
  MinMax scaling to [0, 1] gives +0.31 (Rice) and +0.11 to +0.16 (Glass) on the
  qubit model, and +0.37 / +0.13 to +0.17 on a photonic reuploading model. An
  applied initial point gives nothing in either. Treat input scaling as a
  first-class hyperparameter in every angle-encoding study.
- **Equivalence tests expose a confound quickly.** Paired TOST (±0.02) across
  the six released init arms split them into exactly two classes. The split
  matched preprocessing, not the init distribution, and pointed to the bug
  before any re-training.
- **Unequal budgets make optimizer rankings mostly about cost.** COBYLA "being
  faster" here comes from its tolerance stopping it after 138-163 evaluations,
  while SPSA always spends 351. Compare optimizers at equal evaluation budgets
  if the claim is about efficiency.
- **No advantage at matched capacity.** On Rice, a 28-parameter MLP beats the
  28-parameter QNN by 0.03 test accuracy on identical features. On Glass, with
  43 test rows, nothing is separable.

## Implementation pitfalls

- **scikit-learn >= 1.5 changed PCA's sign convention** (two Rice components
  flipped against 1.2.2). In angle encoding, a flipped sign changes the angle,
  not just a label. A `pca_signs: u_based` option restores the old convention;
  without it, the MinMax-arm agreement gap was +0.05.
- **SciPy 1.16 replaced COBYLA's Fortran code with PRIMA.** Pin `scipy<1.16` to
  match older COBYLA results.
- **Qiskit's Nelder-Mead caps `maxfev` at 1000** by default, whatever `maxiter`
  says.
- **Unseeded noisy samplers** (Aer `SamplerV2` without a seed) move
  single-configuration accuracy by up to about 0.1 between identical runs.
  Aggregate agreement stays within 0.015. Seed the simulator when bit
  reproducibility matters.
- **Released "selected on validation" numbers are optimistic.** The paper
  selects and reports on the same 250 rows. A QNN chosen on validation lost
  0.02-0.12 on test here.

## Photonic translation

- **Translate the scientific question, not the circuit.** One trainable MZI
  mesh is universal on its modes, so qubit ansatz-family and entanglement
  questions have no photonic counterpart. Photon number and reuploading depth
  are the natural substitutes. Here neither mattered (within ±0.02).
- **A useful MerLin pattern:** `2d` modes for `d` features, phase `pi * x_i` on
  even modes between trainable meshes, and `mode_expectations` (UNBUNCHED,
  threshold detection) into a linear head, trained with Adam. It trains in
  40-130 s per run on an L4 with 8 concurrent runs.
- **MerLin 0.4.1 on CUDA:** move `layer.measurement_mapping.mask` to the device
  after building a `mode_expectations` layer. Pin `perceval-quandela<1.3`.
- **CUDA runs are not bit-reproducible:** identical seeds gave identical
  initial phases, but effects moved by up to 0.03 over 200 Adam steps. Cite
  significance patterns, not third-decimal effect sizes. Set match tolerances
  from a measured run-to-run floor.

## Baseline fairness and statistics

- **Check the paper's control against the claim's confounds:** here the beta
  arm changed two things at once. The fair control varies one factor with the
  other held fixed (E-B).
- **Validation selection needs a tie rule.** With 43 validation rows, one sample
  is 0.023 accuracy, and several configurations tie or nearly tie. Without a
  tie rule, four of six system comparisons depended on which tied
  configuration was picked. Re-score comparisons over all candidates within one
  validation sample of the best.
- **Small datasets:** Glass (214 rows) cannot separate systems at 3 seeds. Plan
  more seeds, or report it as characterization only.

## Reproducibility and operations

- **Provenance:** launch evidence runs only from a clean commit (or a clean
  worktree of it). Fixing provenance afterwards cost a full rerun of every
  sweep.
- **Re-analysis first:** when a paper releases raw results, re-analysing them
  reproduces its tables cheaply and exposes analysis or code defects before any
  re-training is planned.
- **An independent recompute script** that rebuilds every reported table from
  raw per-run metrics is cheap. It gives an exact consistency check (6.7e-16
  here) that audits can rerun.

## Open directions

- An equal-evaluation optimizer comparison.
- A capacity-matched photonic model against the same classical baselines.
- A shot-based, lossy photonic evaluation, to test whether the scaling
  mechanism survives realistic hardware assumptions.
