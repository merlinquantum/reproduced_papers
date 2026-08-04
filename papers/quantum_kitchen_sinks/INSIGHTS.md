# Insights — Quantum Kitchen Sinks (arXiv:1806.08321)

Distilled, durable notes worth keeping after the reproduction.

## Implementation pitfalls

- The Quil snippets in the appendix use **`RX`** rotations, not `RY`.  We
  initially wrote the simulator with `RY` and got correct shapes but
  ill-tuned σ.  Switching to `RX` does not change the probabilities for the
  CZ ansatz (since CZ is diagonal) but matters for the CNOT case.
- A 4-qubit Quil snippet `cnot4` (Fig. 6) reads in the order ``CNOT 0 2; CNOT
  1 3; CNOT 0 1; CNOT 2 3``.  Be careful with multiplication order when
  composing CNOT matrices.

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

## CZ ansatz "no discrimination" claim

The paper claims the CZ ansatz (Fig. 2b) "leads to classifiers that are no
better than random" because its **implicit kernel is the constant 1/2**.  Our
finite-``E`` reproduction sees this is misleading: at ``E = 5000`` the CZ
ansatz still achieves ~98.5% test accuracy on picture frames.  The argument
that "constant implicit kernel ⇒ no discrimination" is an *asymptotic*
statement about the average over Ω draws; for any finite Ω budget the
per-episode features remain non-trivial single-coordinate non-linear
functions of ``u_i``.  For datasets whose decision boundary is expressible as
``f(|u_0|, |u_1|)`` with a separable ``f`` (as in picture frames: ``max(|u_0|,
|u_1|) ≈ r``), the CZ ansatz is **not** uninformative in practice.

Test a "harder" 2-D dataset (e.g., XOR-on-rings) if you want to actually
demonstrate the CZ failure mode.

## Linear baseline matters

QKS is only meaningful relative to a fair linear classifier (the Linear
Baseline Rule).  Our LR baseline on picture frames gives 49.25% (≈ chance)
and on (3,5)-MNIST gives 96.2% — closely matching the paper's 50% and 95.9%.
Reproducing these baselines is **the first checkpoint**: if the LR baseline
already classifies the data, QKS has no lift to demonstrate.

## Photonic adaptation

The QKS recipe ports cleanly to MerLin on the **picture-frames** task
(99.7% test accuracy with 4 modes / 2 photons / σ=3 / E=2000).

On **(3,5)-MNIST** the photonic story depends strongly on the computation
space, episode budget, and geometry:

| Variant | Best test error | Notes |
|---------|---------------:|-------|
| Gate QKS 1q (paper-aligned) | **1.87 ± 0.09%** | σ=0.05, E=5000 |
| Photonic MerLin QKS (4 modes, 2 photons, UNBUNCHED) | 7.80 ± 0.08% | σ=0.05, E=2000, corrected input-state placement |
| Photonic MerLin QKS (6 modes, 3 photons, UNBUNCHED) | 4.73 ± 0.25% | σ=0.05, E=10000 |
| **Photonic MerLin QKS (6 modes, 3 photons, DUAL_RAIL)** | **3.60 ± 0.42%** | **σ=0.07, E=10000** |
| Photonic MerLin QKS (8 modes, 4 photons, DUAL_RAIL) | 5.40 ± 0.22% | σ=0.07, E=5000 |
| LR baseline on raw pixels | 3.8% | n/a |

The small UNBUNCHED photonic setting does **not** beat the LR baseline on
MNIST. The likely cause there is the mismatch between MNIST's high-dimensional
input (784) and the *small* photonic Hilbert subspace
``C(n_modes, n_photons) = 6`` sampled per episode.

Three experimental facts define the current picture:

1. Fixing the photon placement so that the input state aligns with the encoded
  `input_modes` improved the `(m=4, k=2)` MNIST run from **8.30 ± 0.98%** to
  **7.80 ± 0.08%**.
2. Increasing the episode budget matters a lot for larger photonic spaces, but
  not uniformly. In UNBUNCHED mode, `(m=6, k=3)` improved from **14.40 ± 0.51%**
  at `E=1000` to **4.73 ± 0.25%** at `E=10000`, whereas `(m=8, k=4)` improved
  only from **19.23 ± 0.97%** to a plateau near **8.27 ± 0.29%** at `E=5000`.
3. Moving the enlarged settings into the logical `DUAL_RAIL` subspace helps
  significantly. The best point we found is now `(m=6, k=3, E=10000, σ=0.07)`
  with **3.60 ± 0.42%** test error, essentially matching the LR baseline.

Taken together, these results show that the weak performance is specific to an
underpowered UNBUNCHED setting rather than to photonic QKS as a whole.

The contrast between `6m3k` and `8m4k` is itself informative. Even in
DUAL_RAIL, simply increasing the Hilbert space does not guarantee a better QKS
feature map:

1. `m=6, k=3` benefits from both larger `E` and a move from `σ=0.05` to
  `σ=0.07`, reaching **3.60 ± 0.42%**.
2. `m=8, k=4` also improves in DUAL_RAIL, but only to **5.40 ± 0.22%**, and
  the sigma sweep around `0.05` shows only a shallow plateau between `0.05`
  and `0.07`.
3. The additional modes and parameters in `8m4k` appear harder to exploit than
  the more compact `6m3k` geometry at the available episode budgets.

This is the durable photonic lesson from the reproduction: resource increases
help only when the computation space, encoding support, and episode budget stay
well matched.

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
| Best σ on MNIST (best photonic run) | 0.05–0.10 depending on qubit count | **0.07** for `6m3k` DUAL_RAIL |
| Wall clock for E=1000 | ~1 s | ~10 s |
| MNIST lift over LR? | Yes (≥ 1.8% error vs 3.8%) | Not yet beyond LR, but `6m3k` DUAL_RAIL reaches parity-scale performance |
