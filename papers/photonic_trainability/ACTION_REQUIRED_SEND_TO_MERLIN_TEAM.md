# Action Required: Send to MerLin Team

## Summary

- `ComputationSpace.UNBUNCHED` is easy to mistake for threshold detection. It
  returns the collision-free outcomes renormalized, which is post-selection and
  can discard most of the probability mass without any signal.
- Shot-based training is limited by MerLin always computing full output
  distributions. A sampling path that supports gradients (for example
  parameter-shift with shots) would be needed at hardware scale.
- Threshold detectors trigger a PyTorch sparse-tensor `UserWarning`.
- Four behaviours were verified with known answers and would make useful
  documentation examples (noise models, threshold detectors, batched
  `input_parameters`, parameter ordering).

## Context

This reproduction of *The trainability of photonic quantum circuits*
(arXiv:2607.21544) studies when passive linear-optical circuits can be trained
with finite shots. MerLin 0.4.1 (Perceval 1.2.4, PyTorch 2.14.1, Python 3.13,
macOS arm64, CPU) was used in three ways:

- as the forward model of shot-based generalized parameter-shift training:
  `QuantumLayer` over Clements meshes and loop circuits;
- for Monte Carlo variance estimates over Haar-random meshes;
- for a hardware-aware study adding partial distinguishability, loss and
  threshold detection.

Photon-number observables need the full occupation pattern, so detector
semantics directly change the physics that is reported.

## Items to Send

### Medium — ComputationSpace.UNBUNCHED: post-selection semantics are easy to misread as threshold detection

- Type: missing documentation, with a hardware-model concern.
- What happened: `MeasurementStrategy.probs(ComputationSpace.UNBUNCHED)`
  returns the FOCK distribution restricted to collision-free outcomes and
  renormalized to sum 1. Nothing tells the user how much probability was
  discarded. With bunching this changes the physics:
  - Hong-Ou-Mandel input `|1,1>` on a 50:50 beam splitter: UNBUNCHED returns
    `(1,1)` with probability 1.0, although the physical coincidence probability
    is 0.
  - True threshold detectors (`pcvl.Detector.threshold()` through
    `pcvl.Experiment`) return `(1,0)` and `(0,1)` with 0.5 each, the correct
    click statistics.

  Guidance in circulation (including the cookbook this reproduction followed)
  describes UNBUNCHED as the "threshold detection" default.
- Evidence (MerLin 0.4.1, reproduced deterministically):
  - HOM check: UNBUNCHED keys `[(1, 1)]`, probabilities `[1.0]`. Threshold
    detector keys `[(1,0), (1,1), (0,1)]`, probabilities `[0.5, 4.9e-32, 0.5]`.
  - Generic 4-mode Haar circuit with input `|1,1,0,0>`: the collision-free mass
    in FOCK is 0.4416. UNBUNCHED sums to 1.0 and equals `FOCK / 0.4416` exactly
    (max difference 0.0); the maximum difference from raw FOCK is 0.167.
  - Threshold detection equals PNR probabilities aggregated into click
    patterns, to 1e-12 (`papers/photonic_trainability/tests/test_merlin_hardware.py::test_threshold_detection_aggregates_pnr_probabilities_into_clicks`).
- Why it matters: a user who picks UNBUNCHED as "hardware-realistic threshold
  detection" silently gets a post-selected, renormalized distribution. In the
  saturated regime (n comparable to M) that can drop most of the probability
  mass. Variances, losses and trainability estimates then describe a different
  experiment, and the post-selection shot cost (the accepted fraction) is
  invisible.
- Suggested action:
  - Document UNBUNCHED explicitly as collision-free post-selection with
    renormalization.
  - Point users to `pcvl.Detector.threshold()` for threshold detection.
  - Optionally expose the discarded probability (or the acceptance rate) when
    UNBUNCHED is used, so the cost of the post-selection is visible.

### Medium — QuantumLayer / sampling: no gradient-capable path beyond full-distribution simulation

- Type: improvement (integration blocker at hardware scale).
- What happened: shot-based parameter-shift training needs many shifted
  circuit evaluations, each sampled. MerLin computes the full output
  distribution, so training is bounded by the Fock dimension. At the paper's
  hardware scale (ORCA PT-2: 48 modes, 6 photons, dimension C(53, 6) ≈ 2.3e7),
  full distributions per shifted circuit were not practical. Training was
  therefore run on 16-mode substitutes (dimension 54,264). The documented
  sampling route (`MerlinProcessor` with a local Perceval processor) is
  inference-only.
- Evidence:
  - Perceval `CliffordClifford2017` sampled about 2,000 samples/s at 48 modes and
    6 photons (local probe on Apple M2). The paper's protocol (94 parameters x 4
    shifts x 1,000 shots per iteration) would take about 190 s per iteration,
    or about 5 hours per training run.
  - Configs fail fast above `max_fock_dimension`
    (`papers/photonic_trainability/configs/fig9_loop_training_substitute.json`).

  Uncertainty: `MerlinProcessor` throughput was not benchmarked; the
  bottleneck may be partly Perceval's sampler.
- Why it matters: shot-based training of photonic circuits at hardware sizes
  (the regime where trainability claims matter) is out of reach in MerLin
  without hand-written sampling plus parameter-shift code.
- Suggested action:
  - A sampling forward mode that returns outcome counts for a batch of
    parameter vectors without materializing the distribution.
  - Optionally, a built-in parameter-shift helper with shots on top of it;
    for phase parameters with n photons, frequency bound R ≤ n suffices.

### Low — Detectors: sparse-tensor UserWarning from the detector transform

- Type: bug (cosmetic).
- What happened: every layer built with threshold detectors emits PyTorch's
  "Sparse invariant checks are implicitly disabled" `UserWarning` from
  `merlin/measurement/detectors.py:400` (`torch.sparse_coo_tensor(...)`).
- Evidence: the pytest output of
  `papers/photonic_trainability/tests/test_merlin_hardware.py` (MerLin 0.4.1,
  PyTorch 2.14.1); it appears on every run that uses threshold detectors.
- Why it matters: noise in user logs and test output, and it hides real
  warnings.
- Suggested action: pass `check_invariants=` explicitly, or wrap the
  construction in a scoped warning filter.

### Low — Documentation: verified behaviours worth adding as examples

- Type: improvement (documentation).
- What happened: these behaviours, which users need, are easy to get wrong, and
  were each verified against a known answer in this reproduction:
  1. `NoiseModel(indistinguishability=V)` on HOM gives `P(1,1) = (1 - V) / 2`
     exactly (V = 1, 0.9, 0.7, 0 tested), and stays differentiable.
  2. `NoiseModel(transmittance=t)` adds outcomes with fewer photons. The mean
     detected photon number is exactly `t * n`. Post-selecting on n photons
     returns exactly the lossless distribution, because uniform loss commutes
     with linear optics.
  3. A batch of parameter vectors passed through `input_parameters=["phi"]`
     evaluates all parameter-shift shifts in one call. This made shot-based
     training of 132-parameter meshes practical.
  4. `input_parameters` values are assigned in circuit order, not by sorted
     parameter name: a 12-parameter circuit with unpadded names `phi_0 ...
     phi_11` matched the index-order unitary to 2e-16, while a lexicographic
     assignment differed by 0.51.
- Evidence: `papers/photonic_trainability/tests/test_merlin_hardware.py` (HOM,
  loss, threshold, GPSR under noise) and `tests/test_training.py` (MerLin versus
  unitary distributions; GPSR versus autograd). Probes for item 4 were recorded
  in this reproduction's Phase 06 MerLin feedback check.
- Why it matters: these are exactly the questions a user asks when modelling
  realistic photonic experiments; short documented examples would prevent
  silent modelling errors.
- Suggested action: add these as short examples or tests in the noise-model,
  detector and `QuantumLayer` documentation, and state the parameter-ordering
  rule explicitly.
