# Action Required: Send to MerLin Team

## Summary

Four MerLin-specific items surfaced while building a photonic adaptation of a
quantum-reservoir-computing paper on MerLin 0.4.0:

1. **High** — `CircuitBuilder.add_entangling_layer(name=...)` silently merges
   layers whose names differ only by a trailing integer into a single parameter
   tensor. Layers named `mesh_0`, `mesh_1`, ... become one tensor `mesh`, so
   per-layer parameter access fails without any warning.
2. **High** — `add_entangling_layer(trainable=False)` produces an *identity*
   mesh with zero parameters, not a frozen random one. There is no high-level way
   to get a fixed non-trivial interferometer, which is precisely the primitive a
   reservoir-computing user needs.
3. **Medium** — `MeasurementStrategy.partial` exposes a branch decomposition but
   there is no way to re-inject the resulting mixed state as a `QuantumLayer`
   input, so recurrent architectures whose memory is a partial trace cannot be
   expressed in MerLin.
4. **Low** — `MeasurementStrategy.mode_expectations` is the natural analogue of a
   per-qubit `<Z_j>` readout and was the single most useful primitive for this
   reproduction, but it appears in neither the repository cookbook nor the
   patterns most reproductions copy from.

## Context

Reproduction: *Quantum Reservoir Computing for Realized Volatility Forecasting*
(arXiv:2505.13933), `papers/qrc_volatility`. The paper's model is a 10-qubit
transverse-field Ising reservoir with 7 feature-encoding qubits and 3 memory
qubits: encode, evolve, trace out the input qubits, repeat for three lags, then
read `<Z_j>` on all qubits and train only a ridge readout.

The photonic adaptation is a 10-mode, 3-photon frozen Haar mesh with three
sequential angle-encoding blocks and a per-mode photon-number readout. The
requirements it placed on MerLin — a *fixed, untrained, non-trivial*
interferometer, per-layer parameter addressing, per-mode expectation readout, and
a mixed-state memory carried between timesteps — are exactly the requirements any
reservoir-computing or recurrent reproduction will place on it, so the items below
are likely to recur.

Environment: `merlinquantum` 0.4.0 (`merlin.__version__` reports `0.4.0`),
torch 2.12.1+cpu, Python 3.12, CPU-only container.

## Items to Send

### High — `CircuitBuilder.add_entangling_layer(name=...)`: names differing only by a trailing integer are silently merged

- Type: bug (or at minimum a silent, undocumented convention)
- What happened: giving two entangling layers the names `mesh_0` and `mesh_1`
  produced a **single** parameter tensor named `mesh` with the concatenated size.
  Letter suffixes (`mesh_a` / `mesh_b`, `meshA` / `meshB`) produce two separate
  tensors as expected. Nothing is logged or raised.
- Evidence: on MerLin 0.4.0, 6 modes, one `add_angle_encoding` between two
  `add_entangling_layer` calls, reading `QuantumLayer.named_parameters()`:

  ```text
  ['m0',      'm1']      -> [('m',      (60,))]
  ['mesh_0',  'mesh_1']  -> [('mesh',   (60,))]
  ['mesh_a',  'mesh_b']  -> [('mesh_a', (30,)), ('mesh_b', (30,))]
  ['meshA',   'meshB']   -> [('meshA',  (30,)), ('meshB',  (30,))]
  ```

  Reproduced in 4/4 attempts. In `papers/qrc_volatility/lib/photonic.py` this
  caused a copy loop keyed on `name != "mesh_3"` to become a no-op, so a
  two-reservoir ensemble that was supposed to share three of four meshes and
  redraw the last silently became two *identical* reservoirs. It was caught only
  because a downstream assertion compared the two readout blocks and found them
  bit-identical.
- Why it matters: numeric layer names (`layer_0`, `layer_1`, `block_0`, ...) are
  the obvious naming choice, and the failure is silent and semantic — the circuit
  still runs and still trains, it just has tied parameters the user did not ask
  for. Any user freezing, copying, or separately regularising individual layers
  will hit this.
- Suggested action: either document the trailing-integer grouping convention
  prominently on `add_entangling_layer` / `add_rotations`, or warn when a supplied
  `name` is coerced into an existing parameter group, or key grouping on an
  explicit `group=` argument instead of on the name's spelling.

### High — no high-level way to build a fixed, non-trivial interferometer (`trainable=False` yields identity)

- Type: pain point / API gap
- What happened: `add_entangling_layer(trainable=False)` yields a
  `QuantumLayer` with **zero** parameters whose output equals the input state,
  i.e. an identity mesh. Two different `torch.manual_seed` values give bit-identical
  outputs, so `trainable=False` cannot be used to instantiate distinct fixed
  reservoirs.
- Evidence: 6 modes, 2 photons, `input_state = [1,0,1,0,0,0]`,
  `mode_expectations` readout:

  ```text
  trainable=True   named params [('meshA',(30,)), ('meshB',(30,))]  out[0,:3] = [0.2128 0.6840 0.4925]
  trainable=False  named params []                                  out[0,:3] = [1. 0. 1.]
  seed 1 vs seed 2 with trainable=False: max abs difference = 0.0
  ```

  Working around it requires `trainable=True` plus
  `torch.manual_seed(instance_seed)` before construction and
  `parameter.requires_grad_(False)` afterwards
  (`papers/qrc_volatility/lib/photonic.py::PhotonicReservoir`).
- Why it matters: "fixed random interferometer, train only the readout" is the
  definition of a photonic reservoir and of quantum extreme learning machines. It
  is also how `merlin.ReservoirClassifier` describes itself ("initialized once
  from a Haar-random interferometer and kept frozen"). The `CircuitBuilder` path
  offers no equivalent, so every reservoir reproduction reinvents the same
  seed-then-freeze workaround, and a user who reads `trainable=False` as "frozen at
  its random initialisation" gets a silently trivial circuit.
- Suggested action: add something like
  `add_entangling_layer(trainable=False, init="haar", seed=...)`, or expose the
  `ReservoirClassifier` frozen-mesh construction as a builder-level primitive, and
  document what `trainable=False` currently means.

### Medium — `MeasurementStrategy.partial` cannot feed its mixed state back into a `QuantumLayer`

- Type: integration blocker
- What happened: the paper's memory mechanism is a partial trace — measure and
  discard the input register, keep the reduced (mixed) hidden-register state,
  inject fresh photons into the input modes, repeat. `MeasurementStrategy.partial`
  (with `return_object=True`) returns a `PartialMeasurement` describing the
  classical mixture over measured-mode outcomes with per-branch hidden amplitudes,
  but `QuantumLayer` accepts only a pure Fock `input_state`, so that mixture cannot
  be re-entered as the next step's input.
- Evidence: `papers/qrc_volatility/lib/photonic.py` module docstring records the
  assessment; `inspect.signature(ml.QuantumLayer.__init__)` shows
  `input_state: StateVector | pcvl.StateVector | pcvl.BasicState | list | tuple | None`
  with no density-matrix or mixture option. The reproduction was therefore recorded
  as `PARTIAL_MERLIN_TRANSLATION` rather than a full translation. Uncertainty: we
  did not exhaustively search for an undocumented path; the conclusion is "not
  expressible through the public API", not "impossible in principle".
- Why it matters: partial-trace memory is the standard construction in the quantum
  reservoir computing literature (Fujii & Nakajima and everything downstream), and
  it is also what a physical implementation with a delay line would do. Without
  mixed-state input, these architectures have to be reimplemented on low-level
  Perceval with a custom Fock-space density-matrix simulator, which defeats the
  purpose of using MerLin and makes cross-paper comparability harder.
- Suggested action: allow a `QuantumLayer` input to be a mixture / density matrix
  (even if only in the analytic, `shots=0` path), or provide a documented recipe
  for chaining `PartialMeasurement` branches across timesteps with photon-number
  heralding on the discarded modes.

### Low — `MeasurementStrategy.mode_expectations` is the key primitive for porting `<Z_j>` readouts but is undocumented in the patterns users copy

- Type: missing documentation
- What happened: the paper's readout is the per-qubit Pauli-Z expectation vector.
  Its exact photonic analogue is the per-mode expected photon number, which
  `MeasurementStrategy.mode_expectations(computation_space=...)` provides directly
  and which makes the photonic readout *iso-dimensional* with the qubit readout
  (10 features for 10 modes) — essential for a fair comparison. It was found only
  by enumerating `dir(merlin.MeasurementStrategy)`.
- Evidence: `mode_expectations` and `occupancy_readout` do not appear in the
  repository's `MERLIN_COOKBOOK.md` (which documents `PROBABILITIES`,
  `partial`, `LexGrouping`, `ModGrouping`); its docstring documents
  `computation_space` but not the `keys` argument that `ModeExpectations` itself
  takes. Verified working: 10 modes / 3 photons / UNBUNCHED gives an
  `output_size` of 10 whose rows sum to the photon number (3.0).
- Why it matters: reproductions of gate-model papers routinely need "one feature
  per wire". Without this primitive the obvious choice is the full outcome
  distribution (`probs`, 120 features here), which changes the readout dimension
  and silently makes any iso-parameter comparison unfair. In this reproduction the
  120-feature readout was also *worse* (test MSE 0.185 vs 0.134 at matched
  encoding scale), so the default choice actively hurts.
- Suggested action: add a short "porting a per-qubit expectation readout" entry to
  the user guide and to the repository cookbook, and complete the
  `ModeExpectations` / `mode_expectations` docstrings.

## Additional observation (no action requested)

Phase encoding is `2*pi`-periodic while a qubit `RY(pi x)` rotation is
`4*pi`-periodic in its argument, so copying a gate-model encoding scale into a
photonic circuit maps `x = -1` and `x = +1` to the same phase and silently folds
half the feature range. Reducing the scale from `pi` to `pi/8` improved mean test
MSE from 0.2070 to 0.1482 (PQR1) and 0.1765 to 0.1136 (PQR2) over 25 mesh seeds
each. A sentence to this effect in the encoding documentation would save future
reproductions a debugging cycle; the repository cookbook's "encoding scale" pitfall
note is the right place, but it does not currently explain *why* the gate-model
value is wrong.
