# Insights — Quantum Kitchen Sinks (arXiv:1806.08321)

Distilled, durable notes worth keeping after the reproduction.

## Implementation pitfalls

- The appendix Quil uses **`RX`** rotations, not `RY`.  Substituting `RY`
  gives identical *single-qubit* measurement probabilities — the two differ
  only by a phase on the ``|1>`` component — so the mistake does not announce
  itself; it shows up only as a mis-scaled optimal σ.  The phase does matter
  once an entangler acts on the state.
- A 4-qubit Quil snippet `cnot4` (Fig. 6) reads in the order ``CNOT 0 2; CNOT
  1 3; CNOT 0 1; CNOT 2 3``.  Be careful with multiplication order when
  composing CNOT matrices.  Fig. 2(c) appears to draw the two pairs in the
  opposite order, and they do not commute; the appendix Quil is the authority
  we follow.
- **Gate order is the whole story for Fig. 2(b).**  Its CZ comes *before* the
  encoding rotations; putting it after makes it a no-op.  See the CZ section
  below.

## Sigma scaling with qubit count

The optimal σ depends on the **per-qubit fan-in of the encoding** ``Ω`` —
i.e. the number of non-zero entries per row, ``r = p / q``.  In tile encoding
``θ ~ N(0, r · σ²) + U(0, 2π)``, so larger tiles (smaller ``q``) effectively
amplify σ.  For (3,5)-MNIST we found:

| Ansatz | Tile size r | Best σ (E = 1000) | Test error |
|--------|------------:|-------------------:|-----------:|
| 1q     | 784         | 0.05               | 1.9%       |
| 2q     | 392         | 0.10               | 4.3%       |
| 4q     | 196         | 0.10               | 4.0%       |

**Heuristic.** When increasing ``q`` for the tile encoding, scale σ so that
``σ · sqrt(r)`` stays roughly constant.

## CZ ansatz "no discrimination" claim — reproduced, and why it is easy to miss

The paper states that the Fig. 2(b) ansatz has implicit kernel ``k(u,v) = 1/2``
and "leads to classifiers that are no better than random".  We reproduce this:
across a 3 x 3 ``(sigma, E)`` sweep on picture frames, 27 runs, test accuracy is
**48.51 ± 2.26%** — chance.

The mechanism is entirely in the **gate order**, which the figure shows and the
running text does not spell out:

    Fig. 2(a):  |0> -- RX(theta_i) -- CNOT -- measure
    Fig. 2(b):  |0> -- H -- CZ -- RX(theta_i) -- measure

In (b) the CZ acts *first*, on ``|++>``.  ``CZ|++>`` is maximally entangled, so
each qubit's reduced state is maximally mixed; the subsequent ``RX(theta_i)`` is
a local rotation of a maximally mixed state and cannot make any single-qubit
marginal depend on ``theta``.  Every feature bit is an exactly fair coin, hence
``k(u,v) = 1/2``.

Three things worth keeping:

1. **The input dependence is not destroyed, it is made unreachable.**  The
   *joint* two-bit distribution still depends on ``theta``; only the marginals
   are flat.  Because the QKS feature vector is the raw bits and the classifier
   on top is linear (the paper's Linear Baseline rule), nothing downstream can
   use that correlation.  A non-linear classifier on the same features would
   not be chance-level — which is exactly why the LB rule is load-bearing.
2. **The ordering is load-bearing and fails silently if reversed.**  A CZ is
   diagonal, so applied *after* the rotations it cannot change Z-basis
   marginals at all; the ansatz would collapse to two independent one-qubit
   QKS circuits and score ~98.5% on picture frames — a plausible-looking number
   that contradicts the paper for no physical reason.  ``tests/test_kernel.py``
   pins both orderings against the paper's closed-form kernels.
3. **Train accuracy hides it.**  At ``E = 5000`` with 1600 training points the
   logistic regression fits the random features to ~99% train accuracy while
   test stays at chance.  A train-only check would have missed the effect
   entirely.

## Linear baseline matters

QKS is only meaningful relative to a fair linear classifier (the Linear
Baseline Rule).  Our LR baseline on picture frames gives 49.25% (≈ chance)
and on (3,5)-MNIST gives 95.60% (4.40% error) against the paper's 50% and 4.1%.
Reproducing these baselines is **the first checkpoint**: if the LR baseline
already classifies the data, QKS has no lift to demonstrate.

## Photonic adaptation: the circuit, not the platform, is what matters

**A dual-rail photonic qubit reproduces the gate ansatz exactly.**  Any
single-qubit gate is deterministic in dual rail, so there is no reason for a
photonic QKS to underperform the gate model at one qubit — and it does not, once
the circuit is right.  A balanced 50:50 splitter, the encoding phase ``θ``, and a
second balanced splitter form a Mach-Zehnder whose even-rail click probability is

    P(even rail) = sin²(θ / 2)

which *is* the paper's ``RX(θ)`` ansatz read out in the computational basis.
Measured agreement with the gate-model code path is 2e-07 (float32 precision) for
1, 2 and 3 qubits, and the per-feature signal-to-noise matches to four decimals.
This is pinned by `tests/test_photonic_gate_equivalence.py`.

**The pitfall that hid it: MerLin parameterises a beam splitter by
``R = cos²(θ / 2)``.**  A balanced splitter is therefore ``θ = π/2``; the library
default ``θ = π/4`` is an **85:15** splitter.  Fringe visibility as a function of
the splitter:

| splitter | R | MZI fringe visibility |
|----------|--:|----------------------:|
| random mesh (`add_entangling_layer`) | — | **0.18** |
| `add_superpositions` default, θ = π/4 | 0.854 | 0.50 |
| balanced, θ = π/2 | 0.500 | **1.00** |

Every one of these runs without error and returns plausible accuracies.  Only an
equivalence check against a known-good reference separates them.

### Results on (3,5)-MNIST

All rows share one 4 000/1 000 subsample, so the comparisons are paired.

| Variant | Test error (3 seeds) |
|---------|---------------------:|
| LR baseline (raw pixels) | 4.40% |
| Photonic, random mesh, 4 modes / 2 photons, UNBUNCHED, E=2000 | 7.80 ± 0.08% |
| Photonic, random mesh, 6 modes / 3 photons, DUAL_RAIL, E=10000 | 4.43 ± 0.34% |
| **Photonic, dual-rail MZI, 1 qubit, E=5000** | **1.73 ± 0.17%** |
| **Photonic, dual-rail MZI, 2 qubits, E=2500** | **1.43 ± 0.24%** |
| Gate QKS 1q, E=5000 | 1.87 ± 0.09% |
| Gate QKS 2q, E=5000 | 1.77 ± 0.24% |
| SVM-RBF (non-linear reference) | 0.90% |

The `random_mesh` architecture never beats the linear baseline; the dual-rail MZI
beats it by ~3 percentage points, about 30 of 1 000 test images.  **The earlier
conclusion that photonic QKS shows no lift on MNIST was a statement about one
circuit choice, not about photonic QKS.**

### Why the random mesh loses

The useful quantity is the per-feature signal-to-noise of a single-shot click,

    SNR = Var_u[ p(u) ] / E_u[ p(u)(1 - p(u)) ]

— how much the click probability moves with the input, against the Bernoulli
noise of the shot that reads it.  It sets the episode budget: halving SNR roughly
doubles the episodes needed.  Measured on 784-dimensional standardised inputs:

| Feature map | SNR |
|-------------|----:|
| gate 1q, σ=0.05 | 0.886 |
| photonic dual-rail MZI, 1 qubit | **0.886** (identical, as it must be) |
| photonic random mesh, 6 modes / 3 photons | 0.12 |

For the random mesh this is a **ceiling, not a tuning problem**: across
σ ∈ [0.05, 0.8], `angle_scale` ∈ {1, 2, 4} and `n_layers` ∈ {1, 2, 3} it saturates
at ≈ 0.123, while locality collapses (0.94 → 0.55 → 0.16 → 0.00) once σ passes
≈ 0.1 — features decorrelating faster than the data varies, the classic
random-features bandwidth failure, reached without ever gaining signal.  Adding
photons makes it worse per feature (1 photon 0.28, 2 → 0.15, 3 → 0.12, 4 → 0.10),
though per *episode* it roughly cancels because larger chips emit more bits.

Two things follow.  Scaling a low-visibility architecture up — more modes, more
photons, more episodes — buys very little; that is why `8m4k` never beat `6m3k`.
And per-feature SNR falling with circuit size is *not* photonic-specific: the gate
model does it too (0.89 → 0.59 → 0.33 for 1, 2, 4 qubits) because entanglement
flattens single-qubit marginals.  What distinguishes the architectures is the
level, and the level is set by fringe visibility.

### Two-qubit photonics: KLM works, and is not needed

The paper's Fig. 2(a) CNOT *is* reproducible photonically. A dual-rail CNOT is
``H_B · CZ · H_B`` with a post-selected KLM CZ: three reflectivity-1/3 beam
splitters, two of which pair a logical-|0> rail with a vacuum ancilla. Verified
exact against the gate `cnot2` ansatz to **1.3e-07**, with success probability
exactly **1/9**, independent of the input.

Three practical notes for anyone rebuilding it:

1. **Use MerLin's explicit-circuit interface** (`QuantumLayer(circuit=…)`), not
   the `CircuitBuilder` shorthand. The gadget needs specific splitter
   conventions and non-default reflectivities.
2. **Perceval requires beam splitters on consecutive modes.** A layout that
   works: `[anc, |0>_A, |1>_A, |1>_B, |0>_B, anc]` — the two logical-|1> rails
   are adjacent for the central splitter, the two logical-|0> rails sit next to
   their ancillas, and photons enter on the *outer* rails so that
   `P(|1>) = sin²(θ/2)` on each qubit.
3. **Perceval's `compute_unitary()` uses the transposed index convention**
   relative to "amplitude from input mode i to output mode k". Getting this
   wrong reproduces the *no-entangler* distribution exactly — a very convincing
   wrong answer, since the gate silently drops out.

The construction actually yields `Z_A · CNOT`; the spurious `Z` on the control
is diagonal and therefore invisible to a computational-basis measurement.

**But it does not pay for itself.** On (3,5)-MNIST:

| two-qubit photonic circuit | modes | herald | test error |
|---|---|---|---|
| dual-rail MZI, no entangler | 4 | deterministic | **1.43 ± 0.24%** |
| dual-rail MZI + KLM CNOT | 6 (+2 ancillas) | 1/9 | 1.87 ± 0.40% |
| *gate model `cnot2` for reference* | — | — | *1.77 ± 0.24%* |

The entangler costs 9× the shots and buys nothing here. That is consistent with
the gate model itself, where 1q (1.87%) and 2q (1.77%) are within each other's
error bars — on this task the CNOT is not what produces the lift.

### Do we need post-selection at all? A native 4-mode entangler

A bare central splitter between the two logical-|1> rails — no ancillas, four
modes, two photons — gives the post-selected map `diag(1, a, a, x)` with
`a = √(1-η)` and `x` set by the reflectivity `η`:

| η | normalised diagonal | herald on \|11> |
|---|---|---|
| 1/3 | (1, 0.577, 0.577, **−1/3**) | 1/9 |
| 1/2 | (1, 0.707, 0.707, 0) | 0 |
| 0.9 | (1, 0.949, 0.949, 0.8) | 0.64 |

At `η = 1/3` this is `diag(1, a, a, −a²)`, i.e. **CZ up to local amplitude
damping on each qubit** — an entangling gate, ancilla-free. The two balancing
splitters in the full KLM gadget exist only to undo that local damping and make
the map exactly unitary-on-the-subspace.

For QKS the damping is not obviously fatal, since the features are measured
marginals fed to a linear model. Measured per-feature SNR of the native
four-mode circuit (gate `cnot2` reference: 0.590):

| native 4-mode circuit | SNR | locality ρ | herald |
|---|---|---|---|
| no entangler | **0.971** | +0.89 | 1.00 |
| central splitter η=1/3 | 0.516 | +0.83 | 0.43 (input-dependent) |
| central splitter η=2/3 | 0.762 | +0.88 | 0.60 (input-dependent) |

So a native four-mode entangler is available and roughly matches the gate
model's feature quality — but it is strictly worse than using no entangler at
all, and its herald rate depends on the input (std 0.16), which means the
post-selection leaks data-dependent information rather than merely heralding a
gate. **The honest photonic translation of QKS on this task is the
deterministic, ancilla-free, four-mode circuit.** KLM is the right tool only
when the goal is bit-exact reproduction of the paper's circuit rather than
equal performance.

## Where QKS is most useful

- Tiny circuits, small qubit counts, NISQ noise tolerance — no variational
  training avoids barren plateaus.
- Low-dimensional inputs (synthetic / engineered features).
- Datasets where the linear baseline is *bad* and the QKS has a large lift
  to demonstrate.

## When QKS is *less* compelling

- Data already separable by a linear classifier (no lift to demonstrate).
- Any encoder whose transfer function has low fringe visibility: the signal per
  feature is then small no matter how many episodes, modes or photons are added
  (the `random_mesh` photonic architecture is the concrete example here).
- Whenever a non-linear classical baseline (SVM-RBF) is cheap and already
  strong.

## Photonic-vs-gate qualitative comparison

| Aspect | Gate QKS (Rigetti QVM) | Photonic QKS (MerLin) |
|--------|------------------------|----------------------|
| Entangling primitive | CNOT/CZ between qubits | MZI mesh between modes; a dual-rail CNOT needs post-selection (KLM, 1/9) |
| Bits per episode | ``n_qubits`` | ``n_modes`` occupancy bits after single-shot sampling |
| Output cardinality | ``2^n`` outcomes | ``C(n_modes, n_photons)`` in UNBUNCHED, ``2^(n_modes/2)`` in DUAL_RAIL |
| Best σ on picture frames | 1–4 | 2–3 |
| Per-feature SNR (784-dim inputs) | **0.886** (1q, σ=0.05) | **0.886** with a dual-rail MZI (identical by construction); **0.12** with a random mesh |
| Wall clock for E=1000 | ~1 s | ~10 s |
| MNIST lift over LR? | Yes (1.77% vs 4.40%) | Yes with a dual-rail MZI (1.43%); none with a random mesh (4.43%) |
