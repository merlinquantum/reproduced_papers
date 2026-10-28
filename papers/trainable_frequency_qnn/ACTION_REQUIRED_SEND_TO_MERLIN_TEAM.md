# Action Required: Send to MerLin Team

## Summary

MerLin carried a photonic adaptation of trainable-frequency QNNs end to end. The
adaptation covered toy Fourier regression and a physics-informed Navier–Stokes solve with
third-order input derivatives, with no framework bug blocking the work. The main friction
was the readout. The default `UNBUNCHED` computation space and the built-in one-body
observables silently change which Fourier frequencies a circuit can express, and there is
no built-in threshold-detector (click) measurement without post-selection. We also found
an undocumented semantics question (`add_angle_encoding`), a version-string mismatch, and
a physics limit worth stating in Fourier-expressivity examples: multi-photon frequency
coefficients are capped.

## Context

Paper: Jaderberg et al., *Let Quantum Neural Networks Choose Their Own Frequencies*,
Phys. Rev. A 109, 042421 (2024). It compares fixed-frequency (FF) and trainable-frequency
(TF) encodings. The MerLin adaptation (`papers/trainable_frequency_qnn/lib/qnn_merlin.py`)
uses a trainable MZI mesh → phase-shifter encoding `φ = γ·w·x` → mesh. `w` is fixed (FF),
trainable per mode (TF) or one global scale (FF+S). 3 photons in 6 modes or 4 in 8 modes;
320 simulated runs. Environment: `merlinquantum` 0.4.2 (reports 0.4.1), `perceval-quandela`
1.3.1, torch 2.12.1, Python 3.12, macOS arm64 and a Linux NVIDIA L4 host. The output
spectrum is the scientific object of the paper. Any readout that alters the spectrum
changes the conclusions, so these points matter to every Fourier-series or
data-re-uploading user of MerLin.

## Items to Send

### Medium — MeasurementStrategy / ComputationSpace: readout choice changes the accessible Fourier spectrum, undocumented

- Type: missing documentation
- What happened: with a phase-shifter encoding, the frequencies a layer can output depend
  on the readout, not only on the encoding:
  - `MeasurementStrategy.mode_expectations` (one-body ⟨n_k⟩) shows only ω ≤ 1, whatever
    the photon number.
  - `MeasurementStrategy.probs(ComputationSpace.UNBUNCHED)` (the default) post-selects
    collision-free outcomes and renormalises. The output becomes a ratio of trigonometric
    polynomials, with spectral leakage at ω = 4…8 for a circuit whose spectrum is {1, 2, 3}.
  - Only `FOCK` probabilities (no post-selection) keep the finite integer spectrum.
- Evidence: DFT probes on a 6-mode, 3-photon layer (`LOG.md`, Phase 4 Adaptation Plan).
  `tests/test_qnn_merlin.py::test_ff_threshold_spectrum_is_integer_up_to_photon_number`
  checks that FOCK power at ω ≥ 4 is ≤ 2e-5;
  `::test_unbunched_readout_leaks_above_circuit_spectrum` checks the leakage. The readout
  ablation `m2_fig2b` (10 seeds each) gives different results: FF train MSE 0.101 (FOCK)
  vs 0.066 (UNBUNCHED); TF 0.057 vs 0.037 (`results/m2_fig2b_results.md`).
- Why it matters: users comparing photonic Fourier models with gate-model theory (Schuld
  et al.; Gan et al.) get a spectrum that differs from the theory without any warning. The
  default space post-selects, which also has a hardware cost that is not obvious from the
  API.
- Suggested action: document in `MeasurementStrategy` / `ComputationSpace` and in the
  Fourier-series examples that post-selection renormalisation breaks the finite spectrum
  and that one-body expectations carry only ω ≤ 1. Recommend `FOCK` for spectrum studies.

### Medium — MeasurementStrategy: no built-in threshold-detector (click) readout without post-selection

- Type: improvement
- What happened: realistic threshold detection without post-selection had to be built by
  hand. We requested `FOCK` probabilities, read `layer.output_keys`, grouped Fock states by
  click pattern, and applied a fixed observable vector
  (`lib/qnn_merlin.py`, `click_observable` buffer). This needs the full Fock space even
  though only click patterns are physically measured.
- Evidence: `lib/qnn_merlin.py` (lines ~160–260);
  `tests/test_qnn_merlin.py::test_click_observable_counts_threshold_clicks`.
- Why it matters: threshold detectors are the common hardware case. A hand-built grouping
  is easy to get wrong and slower than a native strategy. Its absence pushes users towards
  `UNBUNCHED`, which changes the model (item above).
- Suggested action: add a threshold-detection measurement (click-pattern probabilities, no
  post-selection), e.g. `MeasurementStrategy.probs(ComputationSpace.THRESHOLD)` or a
  detector option on `QuantumLayer`.

### Low — CircuitBuilder.add_angle_encoding: encoding semantics not stated in the docs

- Type: missing documentation
- What happened: we had to read `merlin/algorithms/layer_utils.py::apply_angle_encoding`
  to confirm the encoding is a phase shifter `PS(φ = scale·x)` with no other mapping (no
  rescaling to [0, π], no arccos). That was needed to know the frequencies it generates
  (photon-number differences weighted by `scale`).
- Evidence: `LOG.md`, Phase 4 Adaptation Plan (source inspection). The first spectrum probe
  showed unexpected harmonics, which turned out to come from `UNBUNCHED`, not from the
  encoding.
- Why it matters: Fourier-spectrum reasoning, and any trainable-frequency (data-scaling)
  model, depends on the exact map from `x` to phase.
- Suggested action: state the phase map and the role of `scale` in the
  `add_angle_encoding` docstring and docs, with the resulting frequency set for n photons.

### Low — Fourier-expressivity examples: multi-photon frequency coefficients are capped

- Type: other (physics observation for documentation and examples)
- What happened: a "simple" encoding (γ = 1 on 3 modes, 3 photons) has the spectrum
  {1, 2, 3}. Its ω = 3 coefficient, however, stays near 0.05, while the target and the
  qubit model reach 0.33. That term needs all photons in the encoding modes coherently
  superposed with none there (a NOON-like state), which a linear mesh on single-photon
  inputs prepares only with small amplitude. Carrying the same frequency on one photon
  through a mode with γ = ω (tower / exponential) avoids the cap.
- Evidence: `m1_fig2a` FF median train MSE 0.095 vs 6.4e-3 for the qubit model. A
  single-seed diagnostic (20 000 iterations) gives amplitudes 0.337 / 0.208 / 0.053 at
  ω = 1, 2, 3. In `m5_fig3` the tower map (mean 0.024) beats the simple map (0.061) and TF
  (0.059) (`results/m1_fig2a_results.md`, `results/m5_fig3_results.md`). Uncertainty: one
  architecture (mesh–encode–mesh, spread input state `[1,0,1,0,…]`) was tested; other
  input states or deeper re-uploading may lift the cap.
- Why it matters: "more photons ⇒ larger spectrum" (Gan et al.) is true for the frequency
  set but not for the reachable coefficients. Examples built on it may mislead users
  about expressivity.
- Suggested action: note the coefficient limit in the Fourier / Fock-expressivity examples,
  and show multi-mode encodings with different γ as the practical way to reach high
  frequencies.

### Low — Packaging: `merlin.__version__` disagrees with the installed distribution

- Type: bug
- What happened: `merlinquantum` 0.4.2 reports version 0.4.1.
- Evidence:
  `python -c "import merlin, importlib.metadata as m; print(merlin.__version__, m.version('merlinquantum'))"`
  → `0.4.1 0.4.2` (env `trainable_freq`, 2026-10-28).
- Why it matters: run logs and provenance records use `__version__`, so results are
  attributed to the wrong release.
- Suggested action: derive `__version__` from the package metadata or bump it in the release
  process.

### Low — QuantumLayer: higher-order input derivatives work; cost note for PINN workloads

- Type: other (positive confirmation, performance observation)
- What happened: third-order derivatives of the layer output with respect to inputs
  back-propagate to mesh phases and encoding scales. Autograd agrees with finite
  differences to 3e-11. A photonic physics-informed Navier–Stokes iteration (2 layers,
  batch 600, 3 photons / 6 modes) costs 1.6–1.8 s on the L4 host (4 workers, mostly CPU)
  and ≈ 0.9–1 s locally. Our hand-written 6-qubit statevector model costs 0.24 s for the
  same workload.
- Evidence: `tests/test_qnn_merlin.py::test_navier_stokes_layout_and_input_derivatives`;
  `metrics.json` `seconds_per_iteration` of `m6_table1_*` vs `e6_table1` runs
  (`results/p5_comparison.md`, Training cost). Uncertainty: different model classes and
  parameter counts, so this is not a like-for-like benchmark.
- Why it matters: it confirms MerLin supports PINN-style losses, a use case not shown in
  the examples. The per-iteration cost is the practical limit for such studies.
- Suggested action: consider a PINN example. Profile the nested-autograd path, e.g.
  whether the unitary is rebuilt for every derivative order.
