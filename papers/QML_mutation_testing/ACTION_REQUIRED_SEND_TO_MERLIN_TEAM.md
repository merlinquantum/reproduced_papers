# Feedback for the MerLin team — from the QML mutation-testing reproduction

Context: photonic adaptation of *Efficient Mutation Testing of Quantum Machine
Learning Models* (arXiv:2605.00107v1) with `merlinquantum` 0.4.1 and
`perceval-quandela` 1.2.4, Python 3.12.3. The reproduction needed to build a
layered photonic ansatz, then *edit* it component by component, which exercised
parts of the API that a normal training script never touches.

Items are ordered by how much they cost us.

---

## 1. BUG / footgun: `add_superpositions()` changes the splitter ratio depending on whether `targets` is given

`CircuitBuilder.add_superpositions()` has two code paths, and they produce
**physically different circuits**:

- **no `targets`** -> falls back to `EntanglingBlock(trainable=False)`, which
  compiles to a bare `pcvl.BS()`. Perceval's default is `theta = pi/2`, i.e. a
  balanced 50:50 splitter.
- **with `targets`** -> emits `BeamSplitter` components with the signature
  default `theta = 0.785398` (`pi/4`). In Perceval's convention reflectivity is
  `cos^2(theta/2)`, so `pi/4` is an **85:15** splitter, not 50:50.

Reproducer (verified on 0.4.1):

```python
import merlin as ml, numpy as np, torch


def probs(explicit):
    b = ml.CircuitBuilder(n_modes=4)
    if explicit:
        b.add_superpositions(targets=[(0, 1), (1, 2), (2, 3)], trainable=False)
    else:
        b.add_superpositions(trainable=False)
    q = ml.QuantumLayer(
        circuit=b.to_pcvl_circuit(),
        input_state=[1, 0, 1, 0],
        n_photons=2,
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.UNBUNCHED
        ),
    )
    with torch.no_grad():
        return q().numpy()


print(np.round(probs(True), 4))  # [0.1336 0.6644 0.114  0.0668 0.0115 0.0098]
print(np.round(probs(False), 4))  # [0.4    0.2    0.2    0.     0.     0.2   ]
```

Why it bites: the docstring says "`theta`: Baseline mixing angle for fixed beam
splitters" with default `0.785398`. `pi/4` *is* the balanced angle in the
common `cos(theta) / sin(theta)` parameterisation, so `pi/4` reads as
"50:50" to anyone who has not internalised Perceval's `cos(theta/2)`
convention. We only found the discrepancy because we were comparing an
explicit component list against the block fallback and the output
distributions disagreed.

Suggested fixes, any one of which would be enough:

- make the default `theta` for `add_superpositions` the balanced angle
  (`pi/2`), matching the no-`targets` path and `pcvl.BS()`;
- or state the reflectivity in the docstring ("default `pi/4` gives
  `cos^2(pi/8) = 85%` reflectivity; use `pi/2` for 50:50");
- or accept `reflectivity=` / `balanced=True` as an alternative to a raw angle.

## 2. MISSING CAPABILITY: `CircuitBuilder` is write-only, so structural work has to leave the high-level API

Anything that needs to *read back or rewrite* a circuit — mutation testing,
architecture search, layer-wise ablations, pruning — has no high-level path.
`builder.build()` returns a `merlin.core.circuit.Circuit` that does expose
`.components`, `.add`, and `.clear`, which looks like the right handle, but:

- there is no documented way to feed a *modified* component list back into a
  `QuantumLayer` (the constructor takes `circuit=` as a `pcvl.Circuit`);
- `EntanglingBlock` and `GenericInterferometer` are opaque composites, so
  "delete one beam splitter inside the mesh" is not expressible at all;
- consequently we had to re-implement the component -> Perceval compilation in
  our own module, duplicating the logic of
  `CircuitBuilder.to_pcvl_circuit` for `Rotation` and `BeamSplitter`.

What would have saved us a day: either
`QuantumLayer(circuit=merlin_core_circuit)` accepting the MerLin-level circuit
directly, or a documented `Circuit.to_pcvl()` plus an `expand()` that lowers
composite blocks (`EntanglingBlock`, `GenericInterferometer`) into primitive
`Rotation` / `BeamSplitter` components. The latter is the smaller change and
unlocks all structural use cases.

## 3. DOCUMENTATION: the flat `theta` tensor's ordering is implicit

`QuantumLayer(circuit=..., trainable_parameters=["theta"])` collects every
Perceval parameter whose name starts with the prefix into a single flat tensor.
Nothing documents the ordering. With names like `theta_1`, `theta_2`, ...,
`theta_10`, a lexicographic collection would put `theta_10` before `theta_2`
and silently permute the learned weights — a failure that would look like bad
training, not like a bug.

We defended against it by zero-padding every generated name
(`theta0000`, `theta0001`, ...) and asserting, in a test, that our circuit
reproduces the `CircuitBuilder` circuit bit-for-bit for a known weight vector.
Please document the guarantee (or the absence of one) in the
`QuantumLayer` reference, and ideally expose the resolved ordering as
`layer.trainable_parameter_names`.

## 4. DOCUMENTATION: `LexGrouping` silently discards outcomes

`LexGrouping(input_size, output_size)` drops the remainder when
`output_size` does not divide `input_size`. The cookbook warns about it; the
API does not. For a classifier this shows up as a mysterious accuracy ceiling.
A `ValueError` (or at minimum a `UserWarning`) when
`input_size % output_size != 0` would turn a silent scientific error into an
obvious one.

## 5. SMALL: sampling is silently ignored under autograd

`layer(x, shots=n)` returns the analytic result with only a `UserWarning` when
the layer is in training mode or gradients are enabled. For a mutation-testing
harness, where the shot budget is part of the protocol being reproduced, a
silently analytic result would have invalidated the whole experiment. We
guarded with explicit `model.eval()` + `torch.no_grad()`. Consider making this
an error when `shots` is passed explicitly, or returning the shot count
actually used so callers can assert on it.

---

## What worked well, for balance

- `QuantumLayer(circuit=<pcvl.Circuit>, trainable_parameters=[...],
  input_parameters=[...])` is an excellent escape hatch: once we had our own
  component IR, wiring it into MerLin was five lines and gradients worked
  immediately with no parameter-shift plumbing.
- `MeasurementStrategy.probs(computation_space=...)` plus
  `ComputationSpace.coerce` made the detector model a one-line configuration
  choice, which is exactly what a hardware-aware results table needs.
- Building ~1,000 distinct 49-component circuits and evaluating each on 20
  samples at 1,024 shots cost about 0.18 s per circuit on CPU, entirely
  dominated by layer construction rather than simulation. That is fast enough
  that exhaustive structural studies are practical.
