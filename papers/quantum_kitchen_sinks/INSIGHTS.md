# Insights — Quantum Kitchen Sinks (arXiv:1806.08321)

Distilled, durable notes worth keeping after the reproduction.

## Implementation pitfalls

- The Quil snippets in the appendix use **`RX`** rotations, not `RY`.  We
  initially wrote the simulator with `RY` and got correct shapes but
  ill-tuned σ.  Switching to `RX` does not change the probabilities for the
  CZ ansatz (since CZ is diagonal) but matters for the CNOT case.
- A 4-qubit Quil snippet `cnot4` (Fig. 6) reads in the order ``CNOT 0 2; CNOT
  1 3; CNOT 0 1; CNOT 2 3``.  Be careful with multiplication order when
  composing CNOT matrices: ``U = CNOT(2,3) @ CNOT(0,1) @ CNOT(1,3) @ CNOT(0,2)``
  matches the sequential application of the Quil program.

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

**Heuristic.** When increasing ``q`` for the tile encoding, scale σ by
``sqrt(p / r_new) / sqrt(p / r_old)`` to keep the variance of the encoded
angles roughly constant.

## CZ ansatz "no discrimination" claim

The paper claims the CZ ansatz (Fig. 2b) "leads to classifiers that are no
better than random" because its **implicit kernel is the constant 1/2**.  Our
finite-``E`` reproduction sees this is misleading: at ``E = 5000`` the CZ
ansatz still achieves ~98.5% test accuracy on picture frames.  The argument
that "constant implicit kernel ⇒ no discrimination" is an *asymptotic*
statement about the average over Ω draws; for any finite Ω budget the
per-episode features remain non-trivial single-coordinate non-linear
features of ``u_i``.  For datasets whose decision boundary is expressible as
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

The QKS recipe ports cleanly to MerLin:

- One ``ml.QuantumLayer`` per episode, with the entangling mesh phases
  re-sampled at construction.
- Freeze all parameters (QKS is open-loop).
- Drive the trainable phases with ``add_angle_encoding`` whose inputs are the
  encoded angles ``θ_e = Ω_e u + β_e``.
- Single-shot sample the output occupation pattern → ``n_modes`` binary
  features per episode.

With 4 modes, 2 photons, UNBUNCHED, ``E = 2000``, ``σ = 3`` we reach 99.7%
test accuracy on the picture frames dataset — matching the gate-model QKS
lift.  Larger σ is needed photonically because the angle encoding scale and
the natural phase wrap-around differ from the qubit-model.  This is a
*useful* photonic finding: a generic photonic chip plus a linear classifier
reproduces QKS-style lifts; no exotic photonic feature is required.

## Where QKS is most useful

- Tiny circuits, small qubit counts, NISQ noise tolerance — no variational
  training avoids barren plateaus.
- High-dimensional inputs that compress nicely into tiles.
- Whenever the linear baseline is *bad* — QKS has a large lift to show.

## When QKS is *less* compelling

- Data already separable by a linear classifier (no lift to demonstrate).
- Datasets whose decision boundary is intrinsically high-frequency in input
  space — finite ``E`` may saturate before the right kernel is approximated.
- When a non-linear classical baseline (SVM-RBF, random Fourier features) is
  cheap and already strong — the comparative advantage shrinks.

## Photonic-vs-gate qualitative comparison

| Aspect | Gate QKS (Rigetti QVM) | Photonic QKS (MerLin) |
|--------|------------------------|----------------------|
| Entangling primitive | CNOT/CZ between qubits | MZI mesh between modes |
| Bits per episode | ``n_qubits`` | ``n_modes`` (with occupation 0/1) |
| Output cardinality | ``2^n`` outcomes | ``C(n_modes, n_photons)`` outcomes |
| Best σ on picture frames | 1–4 | 2–3 |
| Wall clock for E=1000 | ~1 s | ~10 s |

Both inherit the open-loop, no-variational-training property — the photonic
adaptation does not need any new MerLin features.
