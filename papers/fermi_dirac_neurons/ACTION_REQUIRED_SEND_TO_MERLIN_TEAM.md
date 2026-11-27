# Action Required: Send to MerLin Team

## Summary

A MerLin adaptation of a quantized-neuron paper exposed four MerLin
documentation and usability points. None is a defect: every MerLin
computation we checked was exact.

1. `add_entangling_layer` creates m/2 trailing phase parameters that cannot
   affect photon-counting outputs. Their gradient is identically zero, and
   this is not documented.
2. There is no public accessor for a layer's single-photon unitary, and the
   convention linking it to the mode transformation is undocumented. We had
   to find it by trial.
3. Plain gradient descent at a typical small learning rate badly
   under-trains mesh phases where Adam succeeds. A cookbook note would save
   users from wrong conclusions.
4. The adaptation itself, a "Fermi–Dirac neuron" Tr[tanh(H/T)ρ] for a
   passive Hamiltonian, is an exact interferometer + photon-number-resolving
   (PNR) readout. It could make a compact MerLin example.

## Context

The reproduced paper is arXiv:2605.24386 (He, Liu, Wilde), "Fermi–Dirac
machines as quantizations of neurons". It applies an activation φ to a
Hamiltonian, f(ρ) = Tr[φ(H(θ))ρ].

For a passive (photon-number-conserving) quadratic Hamiltonian
H = Σ h_ij a_i†a_j with h = U† diag(ε) U, the N-photon eigenstates are Fock
states of the normal modes, so

  f(ρ) = Σ_n p_U(n) tanh(ε·n / T).

That is one trainable `CircuitBuilder.add_entangling_layer()` mesh, a
`QuantumLayer` with `MeasurementStrategy.probs(computation_space=FOCK)`
(float64, state-vector amplitude input) and a classical readout of the
outcome keys.

Environment: merlinquantum 0.4.1, perceval-quandela 1.2.4, torch 2.14.1,
Python 3.13, macOS (Apple M2), CPU simulation only.

Confirmed working as documented (no action needed, useful as evidence):
- **Amplitude (state-vector) input in FOCK space** gives the same
  distribution as the equivalent `input_state`, to 3.3e-16.
- **`output_keys` ordering** matches reverse-lexicographic Fock ordering
  for every size tested.
- **UNBUNCHED layer** probabilities equal the collision-free FOCK
  probabilities renormalized to 1, to 1e-12.
- **The passive-Hamiltonian identity** above holds to 1e-12 against dense
  Fock-space diagonalization at (m, N) = (4, 2), (4, 3), (5, 2).

Evidence (paths relative to the repository root):
- tests: `papers/fermi_dirac_neurons/tests/test_photonic.py`;
- implementation: `papers/fermi_dirac_neurons/lib/photonic.py`.

## Items to Send

### Medium — CircuitBuilder.add_entangling_layer: trailing phases are inert for photon counting

- **Type:** missing documentation.
- **What happened.** Each MZI that `add_entangling_layer` builds is
  BS → internal phase shifter (`…_li*`) → BS → output phase shifter
  (`…_lo*`) on the MZI's upper mode. When that upper mode meets no later
  beam splitter, the `lo` phase only multiplies one output mode by a phase.
  It cannot change any photon-number distribution, so its gradient is
  identically zero under `probs(...)`.
  - There are m/2 such parameters out of m(m−1), at m = 4, 6, 8.
  - In a gradient-variance study we picked "the last phase of the mesh" as
    a representative parameter and measured a variance of ≈ 1e-34
    (round-off). We drew wrong conclusions from it before finding the
    cause.
- **Evidence.**
  - Zero gradient in 5/5 random trials (random phases, random energies,
    random Haar states) for parameter indices 7, 11 at m = 4; 21, 27, 29 at
    m = 6; 43, 51, 53, 55 at m = 8 (vector order of the layer's single
    trainable tensor `el`).
  - Circuit listing at m = 4 (`layer.computation_process.circuit`): the
    inert parameters are `el_0_lo3` (MZI on modes 0, 1 in the last column
    that touches mode 0) and `el_0_lo5` (MZI on modes 1, 2, last column).
  - Recorded in `papers/fermi_dirac_neurons/LOG.md`, "Correction to the
    E13 entry above".
- **Why it matters.**
  - Users count parameters, compare models at equal parameter count, or
    study per-parameter gradients (barren-plateau analyses). Redundant
    zero-gradient parameters bias these silently: our photonic model had
    m² nominal but m² − m/2 effective parameters.
  - The parameters are harmless for training.
- **Suggested action.** Document that, under counting measurements, the
  trailing output phases of `add_entangling_layer` are gauge parameters.
  Optionally offer a flag that omits them, or expose which parameters are
  measurement-inert for a given measurement strategy.

### Medium — QuantumLayer: no public accessor for the single-photon unitary; mode-transformation convention undocumented

- **Type:** API confusion / missing documentation.
- **What happened.**
  - **No public accessor.** To check the photonic model against
    independent dense Fock-space numerics we needed the mesh's m × m
    unitary U. The only route we found is the internal
    `layer.computation_process.converter.to_tensor(*trainable_parameters)`.
    It is not in the user guide or cookbook.
  - **Undocumented convention.** How U acts on modes (rows or columns;
    U, U†, Uᵀ or conj U) is not stated. We tried all four orderings. Only
    h = U† diag(ε) U makes the identity hold, with the returned
    distribution matching dense diagonalization to 3.3e-16; the other three
    are off by 0.10–0.16 on a random 2-photon state.
- **Evidence.**
  - `papers/fermi_dirac_neurons/lib/photonic.py::MerlinPassiveNeuron.unitary`;
  - `tests/test_photonic.py::test_merlin_neuron_equals_dense_activation_observable`;
  - `LOG.md`, "Phase 4 Adaptation Plan" (exploratory check of the four
    orderings).
- **Why it matters.** Any user who validates MerLin against an independent
  simulation, or maps a Hamiltonian or physical model onto a mesh, needs
  the unitary and its convention. Guessing the convention silently gives
  wrong physics: the wrong orderings are not small errors.
- **Suggested action.**
  - Add a public method, e.g. `QuantumLayer.unitary()`, returning the
    current single-photon unitary.
  - Document the convention: under the paper's model, an outcome
    occupation n has probability |⟨n|Φ(U)|ψ⟩|², with a_j† → Σ_i U_ij a_i†,
    to be confirmed by the team.

### Low — Training guidance: plain gradient descent under-trains mesh phases

- **Type:** improvement (documentation / cookbook guidance).
- **What happened.**
  - **Plain GD, paper protocol.** We trained the photonic neuron with the
    paper's protocol (full-batch gradient descent, lr 0.1, 2000 epochs).
    At (m, N) = (8, 4) it removed only 46 % of the target variance, although
    the model contains the target exactly. A classical baseline on
    one-photon correlations appeared to beat it by up to 1.1 decades.
  - **Adam, one exploratory seed.** With identical data and
    initialization, Adam (lr 0.02) reached a test MSE of 2.7e-4, against
    1.4e-2 for plain GD.
  - **Adam for every model, tuned.** With every model on Adam and the
    learning rate tuned per model (10 seeds), the photonic neuron beat that
    classical baseline at all sizes (+0.16, +0.65, +1.04 decades, 10/10
    seeds), the reverse of the plain-GD conclusion.
- **Evidence.**
  - `papers/fermi_dirac_neurons/LOG.md`: "P4-L results", "Post-hoc
    diagnostic", "Phase 5 results";
  - `results/p4_learning_summary.json`, `results/p5_comparison.json`.
- **Uncertainty.** This is general optimization behaviour of
  periodic-phase landscapes, not a MerLin defect. The Adam-vs-GD probe in
  Phase 4 is a single exploratory seed; the 10-seed confirmation is the
  Phase 5 tuned comparison.
- **Why it matters.** Users reproducing papers that prescribe plain GD can
  wrongly conclude that the photonic model is weaker than a classical one.
- **Suggested action.** A short cookbook note: default to an adaptive
  optimizer for mesh phases, and tune the learning rate before comparing
  models.

### Low — Example opportunity: quantized "Fermi–Dirac neuron" as interferometer + PNR readout

- **Type:** improvement (example).
- **What happened.** For any passive Hamiltonian, the activation
  observable Tr[φ(H)ρ] is evaluated exactly by one mesh, PNR detection and
  a classical function of the photon-number energies ε·n. Each shot yields
  a bounded sample φ(ε·n). We measured the hardware-relevant costs:
  - threshold detection costs up to 0.1 decade of MSE at (8, 4);
  - collision-free postselection costs ≈ 0.12 decade;
  - 1000 PNR shots per state are indistinguishable from exact.
- **Evidence.**
  - `papers/fermi_dirac_neurons/lib/photonic.py`;
  - `tests/test_photonic.py`;
  - `results/p4_learning_summary.json` (`readout_ablations`);
  - README section "Photonic Assessment and MerLin Extension".
- **Why it matters.** It is a compact, physically exact example combining
  FOCK measurement, amplitude input and a nonlinear classical readout. It
  also shows clearly why photon-number resolution matters.
- **Suggested action.** Consider a short example or tutorial based on
  `MerlinPassiveNeuron`, with attribution to the paper.
