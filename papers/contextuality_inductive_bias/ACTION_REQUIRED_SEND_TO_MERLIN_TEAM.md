# Action Required: Send to MerLin Team

## Summary

MerLin 0.4.1 worked without defects for this reproduction. No MerLin bug
was found. Four documentation and API items came up:

1. The renormalising postselection in `ComputationSpace.UNBUNCHED` is not
   documented, and it decides whether a constraint survives
   postselection.
2. There is no accessor for the postselection success (keep) rate. The
   rate fell to 7-63% in trained models.
3. `QuantumLayer` initial phases come from the global torch RNG, which the
   docs do not state.
4. Photon-number conservation as an exact constraint is a reusable modelling
   pattern worth adding to the cookbook.

## Context

The paper is Bowles et al., "Contextuality and inductive bias in quantum
machine learning" (arXiv:2302.01365). It encodes a zero-sum constraint,
E[Y1|x] + E[Y2|x] + E[Y3|x] = 0, into a 3-qubit model as an inductive bias.

The reproduction added a MerLin photonic adaptation
(`lib/photonic_merlin.py`):
- 6 modes, N = 2 or 3 photons, three trainable `add_entangling_layer`
  meshes, 90 parameters;
- angle-encoded inputs re-uploaded twice.

Because passive linear optics conserves photon number, a readout that is
linear in mode occupations makes the constraint hold exactly. Three readouts
were compared: vertex, group (provably noncontextual) and unbiased. Each ran
with threshold detection (`UNBUNCHED`) and photon-number-resolving detection
(`FOCK`), 10 seeds each, on an exact CPU simulation with no shots.

Results:
- The biased photonic models match the best qubit model.
- With PNR detection, the photonic group readout reaches test KL
  4.06e-3 ± 0.21e-3.

The items below affect anyone who builds constraint-preserving readouts or
reasons about threshold-detector hardware with MerLin.

## Items to Send

### Medium — MeasurementStrategy / ComputationSpace: document that `UNBUNCHED` renormalises over collision-free outcomes

- Type: missing documentation
- What happened:
  - The cookbook's computation-space table and the docstrings do not say
    whether `MeasurementStrategy.probs(computation_space=ComputationSpace.UNBUNCHED)`
    returns raw collision-free probabilities, which sum to less than 1, or
    renormalised ones.
  - A numerical check showed it renormalises. It equals the FOCK
    distribution restricted to outcomes with at most one photon per mode,
    divided by their total mass. The maximum absolute difference was 0.0.
  - In other words, UNBUNCHED is threshold detection with collision-free
    postselection.
- Evidence:
  - `papers/contextuality_inductive_bias/LOG.md`, "MerLin semantics
    checked (2026-10-16, merlin 0.4.1)": the FOCK-vs-UNBUNCHED comparison on
    6 modes, N = 2, random parameters.
  - The collision-free success rate was 0.82-0.88 at initialisation.
  - The renormalisation check was a one-off numerical comparison, recorded
    in LOG.md and not kept as a test.
  - Test `tests/test_photonic_merlin.py::test_unbunched_space_and_collision_free_outcomes_are_used`
    confirms only that UNBUNCHED output keys are the C(6, N)
    collision-free states.
  - Environment: merlinquantum 0.4.1, perceval-quandela 1.2.4, torch
    2.12.1, macOS / Apple M2 CPU.
- Why it matters:
  - Whether a linear constraint on mode occupations survives depends on
    this. With renormalisation, the occupation readout stays exactly
    zero-sum: max |sum_k E[Y_k]| is 1.4e-15 under UNBUNCHED.
  - Users reasoning about hardware also need to know that UNBUNCHED results
    are conditional on postselection, not raw detector statistics.
- Suggested action:
  - Add one line to the computation-space documentation and to the
    `MeasurementStrategy.probs` docstring: "UNBUNCHED returns probabilities
    conditioned on collision-free outcomes (renormalised)".
  - State the same for any other postselecting space, such as `DUAL_RAIL`.
    Its behaviour was not checked here.

### Medium — QuantumLayer / MeasurementStrategy: expose the postselection success probability

- Type: improvement
- What happened:
  - MerLin offers no accessor for the probability mass kept by a
    postselecting computation space. The reproduction had to build a second
    `QuantumLayer` with `ComputationSpace.FOCK` and the same `state_dict`,
    then sum the collision-free outcomes via `output_keys`.
  - In training, nothing penalises bunching, and the trained models drift
    towards it. Mean keep rates on the test set after training:
    - N = 2: 0.63 (vertex), 0.57 (group) and 0.44 (unbiased), with the
      worst samples at 0.02-0.12;
    - N = 3: 0.07-0.22, with the worst samples at 0.001-0.012.
- Evidence:
  - `papers/contextuality_inductive_bias/lib/photonic_merlin.py`,
    `PhotonicZeroSumModel.postselection_success`: the FOCK-twin workaround.
  - `papers/contextuality_inductive_bias/results/merlin_hardware_aware.md`:
    keep rates for every configuration, 10 seeds each.
  - Sweeps `fig8_photonic_merlin` (60/60 runs) and
    `fig8_photonic_merlin_fock` (20/20 runs).
  - Test `tests/test_photonic_merlin.py::test_postselection_success_is_a_probability`.
- Why it matters:
  - A model can look good under UNBUNCHED while discarding most shots on
    hardware.
  - The FOCK twin is costly: its output space is larger, and the layer has
    to be rebuilt.
  - Without an accessor, keep-rate reporting and keep-rate regularisation
    are easy to skip.
- Suggested action:
  - Return or expose the pre-renormalisation mass, for example a
    `layer.postselection_probability(x)` method or an optional
    `return_success=True` output.
  - Mention in the cookbook that training under UNBUNCHED can drift
    towards bunched states, and suggest monitoring or penalising the keep
    rate.

### Low — QuantumLayer: document that initial phases use the global torch RNG

- Type: missing documentation
- What happened:
  - Trainable phases of `add_entangling_layer` meshes are initialised from
    the global torch RNG at `QuantumLayer` construction.
  - The constructor takes no `generator` or seed argument, so call
    `torch.manual_seed(...)` before building the layer to reproduce an
    initialisation.
  - This was found by testing, not documentation.
- Evidence:
  - `papers/contextuality_inductive_bias/LOG.md`, "MerLin API notes".
  - Test `tests/test_photonic_merlin.py::test_seeding_is_deterministic`:
    with the same `torch.manual_seed` before construction, two models give
    identical outputs, and a different seed gives different outputs.
  - merlinquantum 0.4.1.
- Why it matters:
  - Multi-seed sweeps need deterministic initialisation.
  - Code that builds several layers, or other torch modules, before the
    MerLin layer silently changes its initialisation.
- Suggested action:
  - State the RNG source in the `QuantumLayer` docstring and the cookbook.
  - Optionally accept a `torch.Generator`.

### Low — Cookbook: photon-number conservation as an exact linear-constraint pattern

- Type: improvement
- What happened:
  - Mode occupations of a passive linear-optical circuit sum exactly to N,
    and `mode_expectations()` sums to N exactly.
  - A readout E_k = sum_m c_{k,m} <n_m>, with sum_k c_{k,m} equal to the
    same constant for every mode m, therefore enforces sum_k E_k = const
    for every input and every parameter value. No penalty or projection is
    needed.
  - This is the photonic counterpart of qubit "generators commute with a
    conserved observable" constructions.
  - A grouped readout (each task owns a set of modes) also makes the model
    provably noncontextual. This gave a clean control for
    contextuality-based claims.
- Evidence:
  - `papers/contextuality_inductive_bias/lib/photonic_merlin.py`: the
    vertex and group readouts.
  - Constraint violation: max |sum_k E[Y_k]| is 1.4e-15 under UNBUNCHED and
    1.0e-14 under FOCK, against 0.51 for the unbiased readout
    (`results/merlin_hardware_aware.md`).
  - The biased readouts beat the unbiased one by 1.5e-3 test KL
    (p = 6e-4, 10 seeds).
- Why it matters:
  - Many QML papers encode conserved quantities or symmetry constraints.
  - This pattern gives MerLin users an exact, parameter-free way to build
    such models, and a natural hardware story: photon number is conserved
    without loss.
  - With loss, the constraint holds only after postselection on N photons.
    This was not simulated here.
- Suggested action: add a short cookbook pattern, "Exact linear constraints
  via photon-number conservation", with an occupation-readout example and
  the postselection and loss caveat.
