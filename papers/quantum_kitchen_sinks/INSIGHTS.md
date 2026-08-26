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
and on (3,5)-MNIST gives 96.2% — closely matching the paper's 50% and 95.9%.
Reproducing these baselines is **the first checkpoint**: if the LR baseline
already classifies the data, QKS has no lift to demonstrate.

## Photonic adaptation

The QKS recipe ports cleanly to MerLin on the **picture-frames** task
(99.5% test accuracy with 4 modes / 2 photons / σ=3 / E=2000).

On **(3,5)-MNIST** the photonic story depends strongly on the computation
space, episode budget, and geometry:

| Variant | Best test error | Notes |
|---------|---------------:|-------|
| Gate QKS 1q (paper-aligned) | **1.87 ± 0.09%** | σ=0.05, E=5000 |
| Photonic MerLin QKS (4 modes, 2 photons, UNBUNCHED) | 7.80 ± 0.08% | σ=0.05, E=2000 |
| Photonic MerLin QKS (6 modes, 3 photons, UNBUNCHED) | 4.73 ± 0.25% | σ=0.05, E=10000 |
| Photonic MerLin QKS (6 modes, 3 photons, DUAL_RAIL) | *withdrawn, re-running* | σ=0.07, E=10000 |
| Photonic MerLin QKS (8 modes, 4 photons, DUAL_RAIL) | *withdrawn, re-running* | σ=0.07, E=5000 |
| LR baseline on raw pixels | 3.8% | n/a |

The `DUAL_RAIL` rows predate the fix to the dual-rail outcome→click-pattern
mapping in `lib/photonic_qks.py` and are being regenerated.  `UNBUNCHED` rows
are unaffected.

The small UNBUNCHED photonic setting does **not** beat the LR baseline on
MNIST. The likely cause there is the mismatch between MNIST's high-dimensional
input (784) and the *small* photonic Hilbert subspace
``C(n_modes, n_photons) = 6`` sampled per episode.

Three experimental facts define the picture:

1. The input state must place its photons on the modes the encoding drives.
  With `(m=4, k=2)` on MNIST, aligning the input state with `input_modes` is
  worth about half a point of test error and most of the seed-to-seed variance.
2. Episode budget matters a lot for larger photonic spaces, but not uniformly.
  In UNBUNCHED mode `(m=6, k=3)` goes from **14.40 ± 0.51%** at `E=1000` to
  **4.73 ± 0.25%** at `E=10000`, whereas `(m=8, k=4)` moves only from
  **19.23 ± 0.97%** to a plateau near **8.27 ± 0.29%** at `E=5000`.
3. Whether the logical `DUAL_RAIL` subspace helps is open: those runs are being
  regenerated after the outcome-mapping fix.  What is already established is
  that no `UNBUNCHED` setting measured so far beats the 3.80% LR baseline.

Taken together, these results show that the weak performance is specific to an
underpowered UNBUNCHED setting rather than to photonic QKS as a whole.

In UNBUNCHED, where the measurements are trustworthy, `8m4k` is *worse* than
`6m3k` at every episode budget tried: simply enlarging the Hilbert space does
not buy a better QKS feature map when the episode budget cannot exploit it.

## The photonic bottleneck is feature contrast, and it is a ceiling

The useful quantity is the per-feature signal-to-noise of a single-shot click,

    SNR = Var_u[ p(u) ] / E_u[ p(u)(1 - p(u)) ]

— how much the click probability actually moves with the input, against the
Bernoulli noise of the single shot that reads it.  It sets how many episodes are
needed: halving SNR roughly doubles the episodes required for the same signal.
Measured on 784-dimensional standardised inputs with the tile encoding:

| Feature map | SNR |
|-------------|----:|
| gate 1q, σ=0.05 | **0.88** |
| photonic 6 modes / 3 photons, DUAL_RAIL | **0.12** |

The gate model's `p = sin²(θ/2)` sweeps the full `[0, 1]`; the photonic click
probability only wanders about ±0.17 around 0.5, because the random entangling
meshes wash out the interference fringe.

**This is a ceiling, not a tuning problem.**  Across σ ∈ [0.05, 0.8],
`angle_scale` ∈ {1, 2, 4} and `n_layers` ∈ {1, 2, 3}, SNR saturates at ≈ 0.123.
Past σ ≈ 0.1 the extra σ buys nothing and costs locality: the correlation
between the features of a point and of a small perturbation of it falls
0.94 → 0.55 → 0.16 → 0.00, i.e. the features decorrelate faster than the data
varies and the model stops generalising.  That is the classical random-features
bandwidth failure, reached without ever gaining signal.

**The architecture is the lever.**  Replacing the random meshes around the
encoder with fixed 50:50 splitters on each rail pair — a proper dual-rail MZI —
improves both axes at once (at σ=0.2: SNR 0.123 → 0.151 and locality
0.55 → 0.69; dropping the trailing mesh as well reaches 0.200 / 0.734).  So the
trailing random mesh is measurably part of the problem.  Even so the best
variant tried is ~4× short of the gate model, so the practical implications are:

- a lift on (3,5)-MNIST needs an episode budget of order `E ≈ 50 000–100 000`,
  not 10 000; or
- a higher-visibility encoder; or
- a task with more headroom.  MNIST 3-vs-5 gives the linear baseline 3.80%
  error on 1000 test points, so the whole prize is 3.8 points and fractions of
  it must be resolved against ±0.59 pp of binomial noise.  Picture frames,
  where LR is at chance, is where the photonic model already shows a 50-point
  gap.

## Where QKS is most useful

- Tiny circuits, small qubit counts, NISQ noise tolerance — no variational
  training avoids barren plateaus.
- Low-dimensional inputs (synthetic / engineered features).
- Datasets where the linear baseline is *bad* and the QKS has a large lift
  to demonstrate.

## When QKS is *less* compelling

- Data already separable by a linear classifier (no lift to demonstrate).
- High-dimensional inputs paired with a *small* Hilbert subspace per episode
  (the photonic UNBUNCHED `(m=4, k=2)` setting on MNIST is a clear example).
- Larger photonic meshes when the episode budget is still too small to exploit
  them effectively (the `8m4k` plateau is the concrete example here).
- Whenever a non-linear classical baseline (SVM-RBF) is cheap and already
  strong.

## Photonic-vs-gate qualitative comparison

| Aspect | Gate QKS (Rigetti QVM) | Photonic QKS (MerLin) |
|--------|------------------------|----------------------|
| Entangling primitive | CNOT/CZ between qubits | MZI mesh between modes |
| Bits per episode | ``n_qubits`` | ``n_modes`` occupancy bits after single-shot sampling |
| Output cardinality | ``2^n`` outcomes | ``C(n_modes, n_photons)`` in UNBUNCHED, ``2^(n_modes/2)`` in DUAL_RAIL |
| Best σ on picture frames | 1–4 | 2–3 |
| Per-feature SNR (784-dim inputs) | **0.88** (1q, σ=0.05) | **0.12**, saturating — see the bottleneck note above |
| Wall clock for E=1000 | ~1 s | ~10 s |
| MNIST lift over LR? | Yes (1.8% error vs 3.8%) | No lift in any UNBUNCHED setting; DUAL_RAIL re-running |
