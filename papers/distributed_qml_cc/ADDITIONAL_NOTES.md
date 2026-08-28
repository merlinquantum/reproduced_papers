# Additional notes — distributed_qml_cc

Detailed design notes and exploration pointers that don't belong in the
top-level `README.md`. The `README.md` aims to be a quick start with
"what we did and what we got"; this file captures the *why* and the
material a future contributor will want when extending the work.

## 1. Strategy and key decisions (gate-model side)

1. **Direct PyTorch state-vector simulator.** PennyLane is not
   installed in this container; rebuilding a 4–8 qubit pure-state
   simulator on top of `torch.einsum` (`lib/simulator.py`) is faster
   and gives full autograd support out of the box. With deferred
   measurement (next point), density matrices or branching code paths
   are not needed.
2. **Deferred-measurement pooling.** The QCNN pooling block
   (mid-circuit measurement + outcome-dependent `RZ`/`RX`
   feedforward) is implemented as the equivalent two-qubit controlled
   unitary
   ```
   |0><0|_control ⊗ U_0_target  +  |1><1|_control ⊗ U_1_target
   ```
   with the "measured" qubit simply omitted from the final readout
   (marginalised by tracing it out). This gives identical output
   marginals on the unmeasured qubits and is fully differentiable.
   See `lib/circuit.py::pooling_layer`.
3. **Cross-QPU communication red block.** The paper does not specify
   the exact gate decomposition; we match the parameter counts in
   Fig. 3c:
   - **NC:** no cross-QPU gate.
   - **CC:** one trainable controlled-RX per sub-layer between the
     boundary qubits (control on QPU 0, target on QPU 1). Under
     deferred measurement this is mathematically equivalent to the
     paper's measure-then-feedforward red block.
   - **QC:** controlled-RX + controlled-RZ per sub-layer, modelling
     the larger subgroup of two-qubit operations available with
     quantum communication.
   This choice reproduces the parameter ordering
   `NC (8L+28) < CC (10L+28) < QC (12L+28)` visible in Fig. 3c
   within ~10%.
4. **Brick-wall convolutional sub-layers.** Per sub-layer:
   one `RX(θ_i)` per qubit (so parameter count = qubits per
   sub-layer, as the paper states) followed by a parity-dependent
   CNOT pattern. Even sub-layer pairs `(0,1)(2,3)`; odd sub-layer
   pairs `(1,2)(3,0)` for the cyclic brick-wall.
5. **Embedding.** Havlicek-style ZZ feature map: `H` on each qubit,
   `RZ(x_i)` on each qubit, then cyclic
   `exp(-i x_i x_{i+1}/2 Z⊗Z)` couplings. The non-DQML scheme repeats
   the 4-attribute embedding twice on the same 4 qubits to cover all
   8 attributes; two-QPU schemes split the 8 attributes 4/4 across
   the two QPUs.
6. **Reduced statistical power.** 3 seeds rather than the paper's 10
   trials. Variance is reported, but should not be over-interpreted.

## 2. Quantitative deviations from the paper (gate model)

Our absolute numbers are systematically a few percentage points higher
than the paper's. The likely causes (most → least probable):

- Our Havlicek-style cyclic ZZ embedding appears mildly more
  expressive than the paper's — we did not have access to the exact
  embedding implementation.
- The deferred-measurement pooling block is exactly equivalent to the
  paper's measure-then-feedforward block under the same gate
  parameterisation, but the paper does not specify the rotation gates
  used inside it. We chose RZ then RX, both with outcome-dependent
  angles.
- Our cross-QPU red block uses a single CRX (CC) and CRX+CRZ (QC).
  Fig. 3c suggests a slightly larger number of free parameters per
  cross-edge, which would let our QC saturate marginally lower than
  the paper's.

The qualitative trends and the central claim of the paper
(`CC ≈ QC ≫ NC`) are not affected by these implementation choices.

## 3. Photonic translation — design narrative

The photonic side did **not** work on the first attempt; the final
design is the third iteration. It is useful to keep the dead-end
attempts on record so the same mistakes are not repeated.

### Iteration 1 — angle encoding, `scale = π`, no input normalisation

- Geometry: `m = 8 / n = 4`, angle encoding on all 8 modes with
  `scale = π`, four trainable MZI meshes, no input normalisation.
- Result: **~60% validation accuracy** (close to random).
- Diagnosis: phase wrap-around. The dataset values live in
  `[-π/4, π/4]`; multiplying by `scale = π` gives phases in
  `[-π²/4, π²/4]` ≈ `[-2.5, 2.5]` rad, which is past the linear
  region of the interferometer response. The chip cannot
  discriminate the inputs cleanly.

### Iteration 2 — amplitude encoding, `m = 6 / n = 3`, z-score inputs

- Geometry: amplitude encoding into the `C(6, 3) = 20`-dim Hilbert
  subspace, 4 trainable entangling layers, z-score input
  normalisation.
- Result: **84% final / 85% best**, just at the user-requested 85%
  target.
- The user asked us to stay with angle encoding instead — amplitude
  encoding was rejected as the path forward, even though it worked.

### Iteration 3 (final) — angle encoding, `scale = 1.0`, raw inputs, spread photons

- Geometry: `m = 8 / n = 3`, angle encoding on all 8 modes with
  `scale = 1.0`, **one** trainable MZI mesh before encoding and one
  after (one mesh is photonically universal — no need to stack
  more), trainable single-mode rotations between the encoding and
  the second mesh.
- Input state: photons placed at **evenly-spread** positions
  `[1, 0, 1, 0, 0, 1, 0, 0]` rather than left-filled
  `[1, 1, 1, 0, 0, 0, 0, 0]`. Clustering photons at the start leaves
  some MZI parameters outside the photon "light-cone" with no effect
  on the readout; spreading them keeps every trainable parameter
  active.
- No input normalisation (raw inputs are already in a sensible phase
  range; combined with `scale = 1.0`, phases are bounded by `π/4`
  and the chip operates in the linear regime).
- Adam(lr=0.05), batch 256, 800 iterations, 3 seeds.
- Result: **89.4 ± 3.3% final / 90.2 ± 2.7% best.** Above target.

### Two photonic gotchas worth their own bullets

- **Angle-encoding scale depends on the input range, not on the
  tutorial default.** A scale sweep on this dataset shows
  `scale ∈ {1.0, π/2, π/4}` all hit ≥ 95% on the simple-chip
  baseline; `scale = π` and `scale = 2π` collapse to ~50%. Any
  future photonic reproduction of an unfamiliar dataset should run
  a quick scale ablation before trusting tutorial defaults.
- **Photon placement matters.** The light-cone argument has nothing
  to do with the dataset; it is purely about which trainable
  parameters can affect the readout given the photon state. Spread
  photons evenly across the chip whenever possible.

## 4. Photonic schemes — recipe details

### Non-DQML (one chip)

- `lib/merlin_model.py::PhotonicSingleChip`.
- 1 × `m = 8 / n = 3`, angle encoding all 8 attributes one per mode,
  tutorial recipe (mesh + encoding + rotations + mesh).
- Readout via `LexGrouping(56, 2)` then fixed `[+1, -1]` head — final
  scalar = `logit(+1) - logit(-1)`.

### NC-DQML (two chips, no inter-chip op)

- `lib/merlin_distributed.py::MerLinDistributedDQML(scheme="nc")`.
- 2 × `m = 8 / n = 3`, one feature-half per chip (chip 0 gets
  `x[0:4]`, chip 1 gets `x[4:8]`).
- Each chip's 56-dim UNBUNCHED probability is reduced to a 2-bin
  *trainable* soft distribution by `Softmax(Linear(56, 2))`
  (`LearnedBitHead`). This is the photonic analogue of the gate
  model's QCNN pooling tree.
- The two soft bits feed a learnable 4-element interpret function
  initialised to parity `[+1, -1, -1, +1]`.

### CC-DQML (NC + classical feedforward)

- Same two chips. Chip 1 has **one extra mode** whose angle encoding
  takes one of two trainable feedforward phases `(φ_0, φ_1)` as an
  additional input.
- Forward pass: compute chip 0's soft bit `p_chip0 ∈ R^2`; run chip
  1 twice, once with `φ_0` and once with `φ_1`; weight the two
  chip-1 soft bits by chip 0's soft bit:
  ```
  P_CC[b0, b1] = p_chip0[b0] * p_chip1[b1 | feedforward = φ_{b0}]
  ```
- Equivalent to the standard "measure-then-conditional-rotation"
  mid-circuit feedforward with the measurement deferred to the
  trainable soft-bit head.
- Adds only **~90 trainable parameters** over NC (one `Linear(57, 2)`
  + 2 feedforward phases) yet lifts accuracy by ~7 pp.

### QC-DQML (one larger coherent chip)

- 1 × `m = 16 / n = 6`. "QC is just one bigger chip" — full quantum
  coherence between the two halves. Angle encoding loads all 8
  attributes on the first 8 modes; modes 8–15 carry photons but no
  encoded data.
- Threshold readout sums `C(16, 6) = 8008` outcomes; reduced to 2
  bins via the same `Softmax(Linear(8008, 2))` head, then a fixed
  `[+1, -1]` classifier.
- The `Linear(8008, 2)` head dominates the parameter count
  (~16,000 of the 16,514 total). The photonic chip itself only has
  ~500 parameters.

### Why `LearnedBitHead` instead of plain `LexGrouping`

A first attempt at the distributed schemes used `LexGrouping(56, 2)`
as the chip-bit extractor — i.e. a fixed partition of the 56 UNBUNCHED
outcomes into two groups of 28. Both NC and CC stuck at ~52%
(essentially random). The fixed lex partition forced the chip to
align discriminative information with a specific outcome ordering,
which optimisation couldn't manage in the available budget.

Replacing the bit head with a trainable
`Softmax(Linear(output_size, 2))` immediately unlocked the
distributed schemes:
- NC went from 52% → 88%.
- CC went from 52% → 95%.

This is exactly the role the QCNN pooling tree plays in the paper's
gate model: a *trainable* reduction from a many-outcome distribution
to a single readout bit. A fixed `LexGrouping` is a non-trainable
sibling of that idea — it cannot adapt to the data and is too
restrictive.

## 5. Photonic NC < photonic non-DQML — what's going on?

In the paper's gate model, NC-DQML > non-DQML (~10 pp at L = 9):
running two QPUs is strictly more capacity than one even without any
inter-QPU operation, because each QPU sees its own embedding of half
the features and the joint readout has 4 outcomes vs 2.

In our photonic reproduction, this ordering inverts slightly: NC
(88.2%) sits *below* non-DQML (89.4%). The likely cause is the
photonic geometry choice rather than a flaw in the scheme:

- **Non-DQML photonic** uses a single `m = 8 / n = 3` chip that
  angle-encodes *all 8 attributes* one-per-mode and entangles them
  through MZI meshes. Every feature contributes coherently to the
  readout.
- **NC-DQML photonic** uses two `m = 8 / n = 3` chips, each
  angle-encoding only 4 of the 8 attributes (the other 4 modes carry
  no encoded data — they are "ancillae" of sorts). Each chip is
  therefore working on *half* the information, and the two chips
  cannot mix it.

Empirically a tiny MLP on the first 4 features of this dataset only
reaches ~70% (see `lib/classical_model.py` and the dataset's
"low Pearson correlation by design"). The fact that photonic NC
reaches 88% — not 70% — confirms each chip is recovering most of
its half's information; but combining two half-rich readouts
*without communication* simply cannot beat a single chip that sees
both halves at once.

This is consistent with the paper's narrative: the case for
distributed QML is that *one* QPU is small (4 qubits ≪ 8) and
*splitting the data is forced*. In the gate model the per-QPU width
matches the per-half feature count (4 qubits = 4 features), so two
QPUs are strictly better than one. In our photonic translation we
made each chip the same size as the non-DQML chip (`m = 8`,
following the user's `m = 8 / n = 3` baseline geometry), which gives
the non-DQML side a fairer chance. The CC photonic scheme then
adds back the communication channel and recovers the gain.

## 6. Future work and exploration ideas

### Within-paper extensions

- **Table II (interpret function vs parity readout).** Add a
  `readout: parity | interpret` config field and rerun NC/CC/QC at
  L ∈ {3, 5, 7, 9}. Easy and high-value (one of the paper's claims
  we marked out-of-scope).
- **Higher L (15, 20).** The gate-model sweep can extend to L = 20 in
  a few hours. Worth verifying that the gap-closing trend in Fig. 4d
  continues monotonically.
- **Fig. 3c effective dimension.** Rank-of-Fisher-information for
  500 Haar random states is expensive but tractable; would give us a
  capacity-based confirmation of the same ordering.

### Photonic extensions

- **Make each NC-DQML chip smaller.** Use `m = 4 / n = 2` per chip so
  the photonic non-DQML and NC-DQML have the *same total
  hardware* (8 modes, 4 photons). This is closer to the paper's
  fairness condition and should recover the NC > non-DQML ordering.
- **CC photonic with a non-scalar feedforward channel.** Currently
  the feedforward is a global scalar `(φ_0, φ_1)`. A natural upgrade
  is to let chip 0 emit a small *vector* (e.g. `Linear(56, k)` of
  outputs) that feeds `k` extra angle-encoded modes on chip 1; this
  raises the inter-chip channel capacity from 1 bit to `k` bits.
- **QC photonic at hardware-realistic chip sizes.** `m = 16 / n = 6`
  is large for current photonic chips. A reduced QC variant with
  `m = 12 / n = 4` and partial encoding would be a hardware-feasibility
  study worth running.

### Open question

The paper's CC/QC red blocks are described as "controlled one-qubit
operation" and "arbitrary two-qubit operation" respectively, without
specifying their gate decomposition. We picked CRX (CC) and CRX + CRZ
(QC) so the parameter counts roughly match Fig. 3c. A clarification
from the authors (or a re-implementation against the original code,
if it is ever released) would let us tighten the quantitative
agreement.

## 7. Pointers to the implementation

| File | What it contains |
| --- | --- |
| `lib/data.py` | Synthetic 8D dataset generator (Appendix B). |
| `lib/simulator.py` | Batched PyTorch state-vector simulator (single/two-qubit gate kernels). |
| `lib/circuit.py` | Embedding, conv sub-layer, pooling, cross-QPU edge — gate-model side. |
| `lib/model.py` | `DQMLModel` covering all four gate-model schemes. |
| `lib/training.py` | Adam training loop with eval hooks. |
| `lib/classical_model.py` | `TinyMLP` iso-parameter classical baseline. |
| `lib/merlin_model.py` | `PhotonicSingleChip` — non-DQML photonic baseline. |
| `lib/merlin_distributed.py` | `MerLinDistributedDQML` (NC / CC / QC) + `LearnedBitHead`. |
| `utils/run_sweep.py` | Gate-model grid sweep over (scheme, L, seed). |
| `utils/plot_results.py` | Gate-model figures (Fig. 4c / Fig. 4d / Table I). |
| `utils/plot_photonic_results.py` | Photonic figures and table. |
