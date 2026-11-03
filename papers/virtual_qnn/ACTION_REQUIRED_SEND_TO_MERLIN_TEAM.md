# Action Required: Send to MerLin Team

## Summary

Building a photonic classifier with MerLin 0.4.1's high-level API
(`CircuitBuilder`, `QuantumLayer`, `pcvl.NoiseModel`) worked without reading
MerLin source code. Two modelling pitfalls are not covered by current
documentation or examples, and each silently produced a scientifically
meaningless model:

1. Readouts that are linear in mean mode occupations are blind to photon
   distinguishability and to multi-photon interference.
2. Encoding features on *all* modes loses information, because a passive
   interferometer is invariant under a global phase.

A third, lower-priority point: partial-distinguishability noise cannot be
combined with threshold detection.

## Context

The reproduced paper is *Virtual quantum neural networks* (Zhao et al.,
arXiv:2609.11327), on quasi-probability combinations of trainable circuits.
In Phase 4 the paper's two-branch "virtual classifier" was adapted to
photonics:
- 2 photons in 4 modes per branch: trainable `add_entangling_layer()` meshes
  around an `add_angle_encoding` stage;
- `MeasurementStrategy.probs`;
- `pcvl.NoiseModel(indistinguishability=0.9)`;
- a 2D binary classification task, 10 repetitions.

Environment: merlin 0.4.1, perceval-quandela 1.2.4, torch 2.12.1, Python
3.13.15, macOS (Apple M2, CPU). The final adaptation reached 0.92–0.96 test
accuracy and reproduced the paper's qualitative trend
(`papers/virtual_qnn/results/merlin_comparison.md`). Both pitfalls below were
hit first, and both look reasonable from the API alone.

## Items to Send

### Medium — Documentation / measurement strategies: readouts linear in mean mode occupations ignore distinguishability and multi-photon interference

- Type: missing documentation
- What happened:
  - The first branch readout was the occupation contrast
    z = Σ_k p_k (n0 + n1 − n2 − n3)/2. That is a linear function of the mean
    photon numbers ⟨n_j⟩, the same information `mode_expectations()` returns.
  - With `NoiseModel(indistinguishability=η)` the output probabilities do
    change (max |Δp| ≈ 0.18 at η = 0.3, fixed seed), but this readout did not
    change at all, for any η.
  - The physics is expected: ⟨n_j⟩ = Σ_i |U_ji|² ⟨n_i⟩ does not depend on
    photon distinguishability or on two-photon interference. A model trained
    on such a readout is therefore equivalent to independent single photons,
    and the noise model has no effect on it.
  - A correlation-sensitive readout (photon-number parity (−1)^(n0+n1) with
    PNR detection) fixed it.
- Evidence:
  - Test `papers/virtual_qnn/tests/test_vqc_merlin.py::test_linear_occupation_readout_is_blind_to_distinguishability`:
    the occupation readouts of a noisy (η = 0.3) and a noise-free layer agree
    to 1e-10. Its companion `test_noise_setting_reaches_the_layer` fails with
    a linear readout and passes with the parity readout.
  - `papers/virtual_qnn/LOG.md`, "Phase 4 … Amendment" (dated
    2026-10-13T20:19Z), records the diagnosis.
  - Reproduced deterministically (any seed, any η).
- Why it matters:
  - `mode_expectations()` and linear groupings of `probs()` are natural
    default readouts for regressors and classifiers.
  - Users comparing noise levels, or claiming multi-photon effects, can get
    models that are classically trivial without any warning. The policy that
    a meaningful photonic model needs ≥ 2 photons is silently defeated by the
    readout choice.
- Suggested action:
  - Add a note to the measurement-strategy docs and the cookbook: first-moment
    readouts are invariant to distinguishability and equal the single-photon
    result.
  - Point to correlation-sensitive alternatives: parity, coincidence or
    bunching probabilities, or a trainable readout on the full `probs()`
    vector.
  - Optionally, emit a one-time warning when `mode_expectations()` is
    requested together with a source-noise model. Today that combination
    raises "the probabilities measurement strategy must be used", so the
    warning would matter mainly for linear post-processing of `probs()`.

### Medium — Documentation / `CircuitBuilder.add_angle_encoding`: encoding on all modes makes the model depend only on feature differences

- Type: missing documentation
- What happened:
  - The first photonic encoding placed (x1, x2, x1, x2)·π on all four modes
    via `add_angle_encoding(modes=[0, 1, 2, 3])`.
  - Photon-number conservation makes the output distribution invariant under
    a common phase on all modes. Every accessible phase difference was
    therefore a multiple of x1 − x2.
  - The model could not represent the task's radial decision boundary and
    trained to majority-class accuracy at every setting.
  - Encoding (x1, x2) on modes 0–1 only, with modes 2–3 as a phase-free
    reference, fixed it: test accuracy rose from 0.62 to 0.92 at the same
    settings.
  - More generally, `add_angle_encoding()` on all modes (as in cookbook
    Pattern B, `add_angle_encoding()  # one phase per mode`) yields outputs
    invariant to adding a constant to every feature. One degree of freedom is
    lost, and repeated or symmetric feature layouts can lose many more.
- Evidence:
  - Four points with x1 − x2 = 0.2 gave identical predictions to 10 digits
    (inline probe recorded in `papers/virtual_qnn/LOG.md`, "Amendment 2",
    2026-10-13T20:24Z).
  - Superseded sweep `outdir/sweeps/e4_vqc_merlin_budget`, 50/50 runs: flat
    accuracy 0.640 train / 0.617 test at all settings. Its full table is in
    `papers/virtual_qnn/results/sweep_tables.md`.
  - Guard test
    `tests/test_vqc_merlin.py::test_predictions_are_not_restricted_to_x1_minus_x2`.
- Why it matters: the failure is silent. The model trains, the loss decreases
  slightly, and the result looks like "the photonic model is weak" rather
  than "the encoding is degenerate". It affects any angle-encoded MerLin
  classifier or regressor that encodes on every mode.
- Suggested action:
  - Document the global-phase invariance in the `add_angle_encoding`
    docstring and in the cookbook encoding patterns.
  - Recommend keeping at least one unencoded reference mode, or adding a
    trainable or fixed phase offset per mode.
  - Optionally, warn when the encoded mode set equals all modes and the
    encoding has no per-mode offset.

### Low — Noise modelling: partial distinguishability cannot be combined with threshold detection

- Type: improvement (hardware-model gap)
- What happened:
  - Requesting `ComputationSpace.UNBUNCHED` (threshold detection) together
    with `NoiseModel(indistinguishability=0.9)` returns the full FOCK
    distribution: 10 outcomes, including bunched patterns such as (2,0,0,0).
  - It warns: "Noisy simulations with source noise currently use
    ComputationSpace.FOCK. Other computation spaces are not yet supported."
  - The behaviour is therefore documented and warned about. But the
    hardware-realistic combination, threshold detectors with imperfect
    photon indistinguishability, cannot be simulated directly. Our adaptation
    had to assume photon-number-resolving detection.
- Evidence: inline probe on merlin 0.4.1. With noise: 10 outputs plus the
  warning above. Without noise: 6 outputs, no warning. The FOCK/PNR
  assumption is recorded in `papers/virtual_qnn/README.md`, "Hardware-Aware
  Settings".
- Why it matters: many current photonic platforms use threshold detectors,
  and partial distinguishability is a dominant noise source. Users must
  either assume PNR detectors or coarse-grain FOCK outcomes into threshold
  clicks themselves, which is easy to get wrong.
- Suggested action: support threshold-detection outcomes under source noise,
  e.g. by mapping FOCK keys to click patterns inside the measurement
  strategy. Until then, document the manual coarse-graining recipe in the
  noisy-simulation guide.
