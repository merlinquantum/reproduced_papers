# Action Required: Send to MerLin Team

## Summary

The MerLin 0.4.2 high-level API was sufficient to build a photonic Q-function
for deep Q-learning, with exact gradients and agreement with an independent
Perceval SLOS simulation to about 6e-9. Six items would help future users:
the most important is that `ComputationSpace.UNBUNCHED` silently post-selects
and renormalises, which the cookbook presents as "threshold detectors";
threshold detection without post-selection needs `FOCK` with
`occupancy_readout=True`, which the cookbook does not mention. The other items
are a `StateVector` / phase-input restriction, undocumented parameter-ordering
semantics, an undocumented batching pattern, a version string mismatch and a
CPU-versus-GPU observation.

## Context

Reproduction of "Quantum agents in the Gym" (Skolik, Jerbi, Dunjko 2022): a
DQN whose Q-function is a 4-qubit circuit, adapted to linear optics with
MerLin (6 modes, 3 photons, per-action click probabilities as Q-values;
`papers/quantum_agents_deep_q_learning/`). The adaptation uses the circuit
builder through Perceval circuits with all phases passed as batched
`input_parameters`, so that many independent agents run in one call. All
observations below are on `merlinquantum==0.4.2` (Python 3.12, torch 2.4.1
locally and 2.12.1 on the remote machine).

## Items to Send

### High — Measurement strategy: `UNBUNCHED` post-selects and renormalises; threshold readout is undocumented

- Type: missing documentation / API confusion
- What happened: the cookbook (section 7, "Picking a computation space")
  lists `UNBUNCHED` as the default for "threshold detectors, hardware-aligned".
  With 6 modes and 3 photons, `MeasurementStrategy.mode_expectations(UNBUNCHED)`
  returns per-mode values that sum to exactly 3.0000 for every input, i.e.
  only collision-free outcomes are kept and renormalised (a post-selection on
  3 distinct clicks). A real threshold detector without post-selection sees
  bunched events: the expected number of clicks was 2.16-2.27 for random
  phases. The readout that does match threshold detection is
  `MeasurementStrategy.probs(ComputationSpace.FOCK, occupancy_readout=True)`
  followed by a click matrix built from `layer.output_keys`; we found it only
  by reading the docstring of `MeasurementStrategy.probs`. `mode_expectations`
  in `FOCK` returns the mean photon number (sum 3.0), not a click probability.
- Evidence: `papers/quantum_agents_deep_q_learning/LOG.md`, section "Amendment
  (computation space and detector model)"; test
  `tests/test_merlin_model.py::test_unbunched_readout_postselects_on_all_photons`
  (8 random inputs: unbunched click sum 3.0 exactly, threshold click sum
  below 3); observed in every probe, 0 counterexamples; MerLin 0.4.2.
- Why it matters: any reproduction that follows the cookbook default and calls
  the result "threshold detection" is post-selected, which changes both the
  physics and the scientific conclusion (the planned readout in this study
  would have post-selected on all-photon events). It also affects hardware
  claims.
- Suggested action: state in the cookbook and the `ComputationSpace` docs that
  `UNBUNCHED` is a collision-free subspace with renormalisation (post-selected),
  document the `FOCK` + `occupancy_readout=True` pattern for threshold
  detection, and consider a built-in per-mode click-probability strategy so
  users do not need to build the click matrix by hand.

### Medium — QuantumLayer: a per-sample input `StateVector` cannot be combined with phase inputs

- Type: improvement / integration blocker (workaround found)
- What happened: for basis-state encodings (a per-sample Fock input state,
  such as dual-rail input of 4 bits on 8 modes) combined with trainable
  per-sample phase inputs, `layer(state_vector, phases)` raises `TypeError:
  Cannot mix torch.Tensor and StateVector inputs in the same forward() call.
  Use either tensor inputs (angle encoding) or StateVector (amplitude
  encoding).` (the call with the arguments swapped gives a longer message
  recommending `set_input_state()`, which is not per-sample).
- Evidence: probe on 8 modes, 4 photons, `StateVector.from_tensor(one_hot,
  encoding=EncodingSpace.DUAL_RAIL, n_modes=8, n_photons=4)` with a circuit
  with 56 phase inputs; reproduced 2/2 argument orders on MerLin 0.4.2.
  Workaround in `papers/quantum_agents_deep_q_learning/lib/qfunction_merlin.py`
  (`dual_rail_input`): prepare each state exactly from a fixed input with one
  MZI per mode pair and a per-sample phase (checked against Perceval SLOS,
  `tests/test_merlin_model.py::test_dual_rail_preparation_gives_the_basis_fock_state`).
- Why it matters: classical-data-to-Fock-state encodings are common in
  reproductions; the workaround adds interferometers that are not part of the
  intended model.
- Suggested action: support a batched input state together with batched
  circuit parameters, or document the preparation-circuit workaround.

### Medium — QuantumLayer: ordering of `input_parameters` groups is undocumented

- Type: missing documentation
- What happened: with `trainable_parameters=[]` and `input_parameters=["w",
  "x"]` we pass two tensors, one per prefix group, and rely on each tensor's
  columns being consumed in lexicographic order of the Perceval parameter
  names within the group (we zero-pad names, `w00000`, `w00001`, ..., so that
  lexicographic order equals creation order). Nothing in the cookbook states
  the prefix-group semantics or the intra-group order.
- Evidence: `lib/qfunction_merlin.py` (`build_circuit`); agreement between
  `QuantumLayer` and an independent Perceval SLOS simulation to about 6e-9 for
  all tested circuits (`tests/test_merlin_model.py::test_click_probabilities_match_perceval_slos`,
  7 parametrisations). Uncertainty: we always used zero-padded names, so a
  failure for unpadded names (for example `w10` before `w2`) was not tested.
- Why it matters: a wrong assumption would silently permute phases; the
  cross-check against Perceval was necessary to trust the layer.
- Suggested action: document the prefix-group rule and the within-group order,
  and show the zero-padding convention in the cookbook.

### Low — QuantumLayer: batched per-sample circuit parameters work and deserve a documented pattern

- Type: missing documentation
- What happened: passing every circuit phase as a batched `input_parameters`
  tensor of shape `(batch, n_params)` worked, with exact gradients to those
  tensors, and let 10 independent agents (each with its own phase vector)
  share one forward call. The cookbook describes only encoding-style inputs.
- Evidence: `lib/qfunction_merlin.py::MerlinQFunction.click_probabilities`;
  `tests/test_merlin_model.py::test_gradients_reach_every_trainable_tensor`
  and `test_agents_are_independent_and_seeded`; 15 + 3 + 3 runs completed.
- Why it matters: vectorising ensembles, agents or hyperparameter replicas is
  common in RL and tuning studies; the pattern is easy to miss.
- Suggested action: add the pattern (and its gradient behaviour) to the
  cookbook.

### Low — Packaging: `merlin.__version__` does not match the installed release

- Type: bug (minor)
- What happened: after `pip install merlinquantum==0.4.2`, `merlin.__version__`
  prints `0.4.1`; `pip freeze` shows `merlinquantum==0.4.2`.
- Evidence: laptop `.venv` and the remote machine (the environment build log
  prints "merlin 0.4.1" while the freeze lists 0.4.2); reproduced 2/2
  environments.
- Why it matters: reproduction logs record the version; a mismatched string
  causes confusion when auditing environments.
- Suggested action: align the version string with the package metadata.

### Low — Performance observation: GPU slower than CPU for small Fock spaces

- Type: other (observation, cause not profiled)
- What happened: for 6 modes and 3 photons (56 Fock states), batched forward
  and backward passes ran faster on CPU than on an NVIDIA L4: 40.5 ms (CPU, 8
  threads) vs 68.9 ms (cuda:0) per training step with 10 agents on Cart Pole,
  and 95.1 ms vs 158.3 ms for the 15-layer Frozen Lake circuit. We suspect
  kernel-launch overhead but did not profile it.
- Evidence: `papers/quantum_agents_deep_q_learning/outdir/benchmark/remote1_merlin_{cpu,cuda0}_*.json`
  (10 agents, 20 episodes, float32); one measurement per configuration.
- Why it matters: users may expect the GPU to help for small circuits;
  guidance on where the crossover lies would save time.
- Suggested action: document the expected crossover (number of Fock states or
  batch size) or profile the small-circuit path.
