# Action Required: Send to MerLin Team

## Summary

MerLin could express post-selected (heralded) conditional states exactly, and it
agreed with an independent permanent-based simulator to 4e-15. Three things
would make this kind of work easier:

- documenting the `PartialMeasurement` branch API;
- returning conditional density matrices of the unmeasured modes when source
  noise is on, instead of only full Fock probabilities;
- a kernel path for discrete, outcome-indexed feature maps.

None of these is a confirmed bug.

## Context

The paper is Hoch et al., "Quantum machine learning with Adaptive Boson Sampling
via post-selection", Nat. Commun. 16, 902 (2025), arXiv:2502.20305. It builds SVM
kernels from the conditional state of residual photons, after post-selecting a
detection pattern on some modes, with one adaptive circuit per pattern. The
setups use 2-3 photons on 6-8 modes, and one includes bunched (pseudo-PNR)
outcomes.

The MerLin adaptation is `papers/adaptive_boson_sampling/lib/abs_merlin.py`, with
tests in `tests/test_abs_merlin.py` (Phase 04). It ran on the CPU simulator with
MerLin 0.5.0.dev0, a development checkout, and Perceval 1.2.4. The PyPI release
0.4.2 was not tested. Heralding and post-selection are central to photonic QML,
so these points matter beyond this paper.

## Items to Send

### Medium — Measurement API: document `PartialMeasurement` branches for heralded states

- Type: missing documentation.
- What happened:
  - `MeasurementStrategy.partial(modes, ComputationSpace.FOCK)` does exactly
    what heralded protocols need. Each entry of `.branches` carries `.outcome`,
    `.probability` and `.amplitudes`: a `StateVector` of the unmeasured modes,
    with `.basis` and `.tensor`.
  - This was found only by reading signatures and docstrings with `inspect`. The
    cookbook used for the work (written for 0.4.1) describes `partial` but not the
    branch objects.
  - Two points needed a manual check against our own basis: whether the
    amplitudes are normalised, and the basis order of the residual modes. The
    code now raises if the order differs.
- Evidence:
  - `papers/adaptive_boson_sampling/lib/abs_merlin.py`, `merlin_state_table`
    (basis check near the `unexpected residual basis order` error).
  - `LOG.md`, "Phase 4 Adaptation Record / APIs, sources inspected".
  - The kernels equal the independent simulator to at most 3.8e-15 for all four
    platforms (`results/merlin_comparison.json`). So once understood, the API is
    correct.
- Why it matters: heralded single photons, post-selected gates and adaptive
  (feed-forward) circuits all need the conditional state, not the marginal
  probabilities. Without documentation, users fall back to computing full
  distributions and conditioning by hand.
- Suggested action:
  - Add a cookbook example "conditional state after heralding", with a
    two-photon HOM-type herald.
  - State the normalisation of `.amplitudes` and the residual basis ordering.
  - Say which MerLin release first exposes the branch API.

### Medium — Noise models: conditional density matrices under source noise

- Type: improvement.
- What happened:
  - With `noise=pcvl.NoiseModel(indistinguishability, g2)`, the adaptation could
    only use `MeasurementStrategy.probs(ComputationSpace.FOCK)`.
  - To get the mixed state of the residual photon, the code appends tomography
    settings to the circuit and post-selects the detected pattern from
    `layer.output_keys`. It then rebuilds the density matrix by linear inversion
    with a PSD projection: 3 settings for a qubit, 7 for a qutrit, one
    `QuantumLayer` per setting and outcome.
  - About 150 lines of `abs_merlin.py` (lines 125-278) exist only to recover a quantity the
    simulator already has internally.
  - Uncertainty: we did not establish whether `partial` with noise is
    unsupported by design or only undocumented. We relied on the FOCK requirement
    for noise and did not try other spaces.
- Evidence:
  - `lib/abs_merlin.py`, `noisy_density_table`, `tomography_settings` and
    `reconstruct_density_matrix`.
  - `LOG.md`, "Mappings considered": "Only `probs()` supports noise, so density
    matrices need the tomography step".
  - Runs `outdir/run_20261105-095357` .. `100508`, which take 50-89 s per kernel
    table against 3-33 s for the exact path.
- Why it matters:
  - Realistic hardware models (partial distinguishability, multiphoton
    emission) for heralded or adaptive protocols currently need hand-written
    tomography.
  - Linear inversion also adds reconstruction artefacts unrelated to the
    physics.
- Suggested action: let `partial(...)` return a branch density matrix (or a
  mixture of `StateVector`s with weights) for the unmeasured modes when a source
  noise model is set.

### Low — Kernels: support outcome-indexed (discrete) feature maps

- Type: improvement.
- What happened: `merlin.FeatureMap` and `FidelityKernel` assume continuous
  inputs encoded into phases. In ABS each data point is a discrete detection
  outcome that selects a whole programmed circuit. The kernel was therefore
  built by hand from the per-outcome states, and the built-in kernel classes were
  not used.
- Evidence: `LOG.md`, "Mappings considered" (`FeatureMap` / `FidelityKernel`
  rejected), and `lib/kernels.py`, `quantum_kernel`.
- Why it matters: post-selection-based and adaptive feature maps are a growing
  family (ABS, heralded encodings). A kernel class that takes a precomputed table
  of states (pure or mixed) indexed by outcome would let them reuse MerLin's
  kernel and SVM utilities.
- Suggested action: allow `FidelityKernel`, or a sibling class, to accept a
  state table or a per-input circuit factory, with Uhlmann fidelity for density
  matrices.

### Low — Validation: reusable cross-checks for FOCK post-selection and source noise

- Type: other (validation evidence the team may reuse).
- What happened: two independent cross-checks passed.
  - The exact MerLin path matched a separate numpy permanent simulator for 2-3
    photon post-selection, including bunched B3 outcomes in FOCK space, to at
    most 3.8e-15.
  - Perceval's `NoiseModel(indistinguishability=0.83)` through `QuantumLayer`
    reproduced an independently written partial-distinguishability model
    exactly.
- Evidence: `results/merlin_comparison.json`, `tests/test_abs_merlin.py` (9
  tests), and the README section "Photonic Assessment and MerLin Extension".
- Why it matters: these are small, deterministic, paper-anchored regression
  cases for heralded FOCK simulation and source noise. Platform A reproduces the
  paper's printed kernel 0.298 / 0.966 / 0.331.
- Suggested action: consider adopting the platform-A and B3 cases as regression
  tests; the test file is self-contained apart from `lib/`.
