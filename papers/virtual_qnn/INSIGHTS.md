# INSIGHTS — Virtual Quantum Neural Networks (arXiv:2609.11327)

Durable observations for future reproductions of quasi-probability, virtual
or error-mitigation QML methods, and for photonic adaptations.

## Scientific insights

- **Judge quasi-probability methods at equal shot budgets.**
  - A signed combination with overhead γ = ‖c‖₁ is unbiased, but its
    per-shot variance is γ² − ⟨P⟩² instead of 1 − ⟨P⟩².
  - Comparisons at exact expectation values, as in the paper, hide this cost
    completely.
  - Here the advantages appeared only above roughly 10² shots per prediction
    (classifier) or 10³ shots per energy estimate (VVQE), and reversed below.
  - Exact closed forms make the equal-shot evaluation cheap: Var = (γ² −
    ⟨P⟩²)/S, plus binomial accuracy. Evaluate the trained models; no
    retraining is needed.
- **"Noise robustness" needs a noiseless control.**
  - The virtual classifier's gain was identical with and without gate noise,
    in both the gate model and the photonic adaptation.
  - A noiseless conventional classifier scored the same as the noisy one.
  - The gain comes from the enlarged readout family (a signed combination of
    two circuit outputs can rescale and reshape the decision function), not
    from noise compensation.
- **Check what "exact" means in a VQE paper.** The paper's "error-free"
  curve was the Hartree–Fock energy. Measured against exact diagonalization,
  even the best virtual budget stayed ≥ 0.04 Ha off.
- **Add a noiseless-ansatz ceiling to every noisy-VQE claim.** The 6-qubit
  reduced-state ansatz stopped 0.02–0.05 Ha above the exact energy even
  without noise. A mitigation method can at best recover that ceiling.
- **Learned quasi-probability inverses do reach the SDP optimum.** VQEM with
  N = 2 branches matched the App. D minimum overhead within about 1%. Under
  approximate inversion it can dip slightly *below* the optimum (HS² ≈ 1e-6).
  Our SDP showed the paper's Table I mislabels exactly this case.
- **Closed forms useful for checks:**
  - minimum inversion overhead for depolarizing D_p: (1 + p/2)/(1 − p)
    (matches the SDP);
  - for amplitude damping A_q: (1 + q)/(1 − q), which matches the SDP at
    q = 0.1, 0.2, 0.3 (checked only there, not proven);
  - unmitigated squared HS error of D_p on half a Bell pair: 0.75 p².

## Implementation pitfalls

- **Inspect released code for figure data arrays early.** Exact plotted
  values turned digitized comparisons into exact ones. They also corrected our own
  plot-digitization misreading and revealed mislabelled reference curves, axis units, and
  swapped table rows.
- **PennyLane `qml.qchem.Molecule` uses bohr by default.** A geometry at
  (0, 0, ±d) means a separation of 2d bohr.
- **Squared vs plain HS distance.** Papers may call Tr[(σ − Φ)²] an "HS norm".
  Confirm with a closed-form point.
- **The reported statistic is often unstated** (single run, best-of-N or
  mean). Predeclare best-of-repetitions as a secondary view. Here it halved
  the E5 gap but did not close it: the paper still beat all 10 of our seeds at
  some points, which points to initializer or landscape differences.
- **Noisy multi-qubit VQE landscapes are multimodal.** Seeds converged to
  energies 0.01–0.04 Ha apart, and doubling the iterations barely moved
  them. Use more seeds rather than more iterations.
- **Config deep merges leak map-valued defaults into overlays.** Keep
  dict-valued defaults empty.

## Photonic (MerLin) translation

- **Readouts must be correlation-sensitive.** Any readout linear in the mean
  mode occupations, such as `mode_expectations()` or occupation contrasts,
  ignores photon distinguishability and multi-photon interference, which
  makes the model equivalent to single photons. Photon-number parity or
  coincidence-type readouts work.
- **Do not encode on all modes.** A passive interferometer is invariant under
  a common phase. Repeating features across all modes made the model depend
  only on x1 − x2, which collapsed it to majority-class accuracy. Keep
  reference modes unencoded.
- **Two cheap sanity tests catch both pitfalls:**
  1. do outputs change when the noise level changes?
  2. are outputs non-constant along feature-difference directions?
- **Distinguishability noise in MerLin 0.4.1 forces PNR detection.** MerLin
  warns and uses FOCK; threshold detection is not available with source
  noise.
- **The VQNN mechanism ports to photonics unchanged.** Two branches, signed
  coefficients, the same estimator. Error-mitigation-style Choi-state
  constructions do not port: passive optics cannot implement arbitrary
  multi-qubit channels on dual-rail qubits.

## Baseline fairness

- **The paper's own baselines** (γ = 1 versions of the same circuit) match
  neither parameters (two branches double them) nor shots. Add:
  - a parameter-matched single circuit;
  - a noiseless control;
  - an equal-shot evaluation;
  - a simple classical model.
- **The circle task is classically easy:** a 17-parameter MLP reached 0.92
  test accuracy against 0.74 for the best gate-model virtual classifier. Do
  not read quantum advantage into toy 2D benchmarks.

## Open directions

- Train under finite shots, not only evaluate.
- Photonic-native error mitigation: learned quasi-probability inverses of
  distinguishability noise.
- Dual-rail photonic VVQE.
- Emulate QuAIRKit's initializer to test whether the remaining Fig. 7 gap is
  initialization.
