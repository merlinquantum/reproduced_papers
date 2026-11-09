# Insights — Power of Data in Quantum Machine Learning (arXiv:2011.01938)

Distilled, durable notes worth keeping after the reproduction. Numbers refer to
10 seeds, Fashion-MNIST, n ≤ 20 unless stated.

## Implementation pitfalls

- **Engineered labels need a median threshold, not a sign threshold.** The
  labels come from the top eigenvector `v` of `sqrt(K^Q) U diag(c/(c+λ)²) Uᵀ
  sqrt(K^Q)`, then `y' = sqrt(K^Q) v`. The PQK and RBF kernels are entrywise
  positive, so by Perron–Frobenius `v` has a single sign and `sign(y')` is
  100% one class for every embedding (checked at n = 4, 8). The median gives
  exactly balanced labels, which is also what the paper's 0.5 chance line
  implies. Keep a guard that raises on single-class tasks.
- **Pin the eigenvector sign.** An eigensolver may return `v` or `−v` across
  platforms or versions, which silently flips every engineered label. Fix the
  sign by a deterministic rule before thresholding.
- **The `g_tra < 0.045` rule can have no admissible pair.** At some (embedding,
  n) no (classical kernel, λ) pair meets it. Report the pair with the smallest
  `g_tra` with a `threshold_met: false` flag, and count the flags (zero in the
  final aggregates), rather than failing or silently relaxing the rule.
- **The E2 formula in the text lacks the imaginary unit.** Treat the E2 pair
  coefficient and phase as config values and score them against published
  numbers (coefficient 2 won). Likewise the E3 register size and evolution time
  (n+1 qubits, time (n+1)/3 won).
- **Simulate once, reuse the states.** One (seed, n, embedding) unit simulates
  the 800 states once and feeds `d`/`g`, the QNN datasets, the bound and both
  engineered datasets. E1 is a product state, so its kernels have an analytic
  form and need no simulation.
- **Kernel-SVR with a polynomial kernel is very slow** at the paper's grid size.
  Kernel ridge regression gave the same role at a fraction of the cost, so the
  extension baseline uses KRR for regression.
- **Batch size must not change results.** For the photonic bank, same-batch
  reruns were identical and different batch sizes agreed to ≤ 1e-16; keep that
  as a test when changing batching or adding device paths.

## The effective input scale (largest open question)

- The text implies features enter the circuits at scale 1. At scale 1 the
  effective dimensions are about 10× the published ones, even for the
  parameter-free E1 kernel. A scale of about 0.3 matches `d` and `g` for E1 and
  E2.
- Matching `d_QK(E1)` at each n needs a scale rising from 0.225 (n = 3) to 0.35
  (n ≥ 14), so a single constant is probably not what the authors used. A free
  per-n scale matches by construction, so this shows how the scale would have to
  move, not that any convention is right.
- Fitting label-free quantities (`d`, `g`) to the published values makes the
  geometry agreement partly circular. Report it as fitted, and test claims on
  quantities that were not fitted (task accuracy and MAE).
- Residual gaps after the fit: `g_QK(E3)` is still about 40% off, and the
  stronger classical baseline on (Q, E3) and E2-engineered data persists at
  input scales 0.3, 0.35 and 0.5 and with a periodic QNN boundary (n = 12, 3
  seeds only).

## Baseline fairness

- **Selection protocol changes the picture.** The paper reports the best test
  score over each grid, which is optimistic. Run both that protocol (to compare
  values) and cross-validation (to judge claims). Verdicts should use the
  paired 95% CI across seeds on identical splits.
- **The classical suite had the larger tuning budget** (377/215 configurations
  for regression/classification vs 36/18 for the fidelity kernel and 288/144 for
  the PQK), so any quantum win is conservative.
- **Where quantum wins, the labels favour it.** QNN-labelled data use the same
  embedding that the kernel uses, and engineered labels are built to saturate
  `s_C = g² s_PQ`. On natural labels (Dataset C) the classical suite is better
  than or tied with every quantum kernel. Do not turn the first two into
  advantage statements about natural data.
- **A gap near zero inflates relative error.** Compare quantum-minus-classical
  differences with their CIs, not their ratios to the paper.

## Photonic translation considerations

- **Projected kernel on photons:** with a fixed photon number, one mode's reduced
  state is diagonal in photon number, so the threshold-detector analogue of a
  one-qubit reduced state is the **per-mode no-click probability P(n_k = 0)**.
  It needs no postselection and involves permanents. Mean photon numbers are
  single-particle quantities, so they are classically trivial and make a poor
  analogue.
- **Fidelity kernel on photons:** the Gram matrix of the output amplitude
  vectors, `|⟨s|U(x')†U(x)|s⟩|²`, needs one circuit evaluation per point, not
  per pair.
- **Computation space:** the no-click marginals need the full `FOCK` space, since
  `UNBUNCHED` discards collision outcomes. `UNBUNCHED` equals `FOCK`
  postselected and renormalized (max difference 0), and `FOCK` agrees with
  permanents.
- **Photon number sets the scale of the effect.** k = m/2 photons in m modes
  makes the Fock dimension grow exponentially, which preserves the
  fidelity-kernel concentration. A cap at k = 8 (dimension 1–3 M at 18–21
  modes) still concentrates the hopping-chain kernel (d = 0.99 N at n = 20) and
  keeps the engineered advantage (+0.28), so a moderate cap is a usable way to
  reach larger n. A very small k (2–3) grows only polynomially; untested here.
- **Embedding choice matters.** A bosonic hopping chain (counterpart of the
  Heisenberg chain) concentrates the fidelity kernel strongly (`d` = 0.94 N at
  14 modes, 0.995 N at 18 modes) and gave the largest engineered advantage,
  growing with size (+0.22 / +0.25 / +0.37 at n = 8 / 12 / 16); a phase sandwich
  (counterpart of IQP) concentrates slowly (0.42 N at 18 modes) and behaved like
  E2 (+0.05 rising to +0.10 at n = 18). A dual-rail separable embedding
  is exactly E1, so it is only useful as an equivalence test.
- **Size limits:** 12 modes with 6 photons (12,376 states) take 151 MB for 800
  points; 14 modes with 7 photons (77,520) take about 1 GB; 16 modes is out of
  reach on a laptop. On one NVIDIA L4 the limit is 18 modes with 9 photons
  (3.1 M states, 240–280 s per 800-point bank), set by host RAM for the stored
  states (20 GB), not by the GPU.
- **MerLin device memory is about 22× the output amplitudes.** A `FOCK`
  `QuantumLayer` forward pass peaks at about 175 B per complex64 output
  amplitude, plus about 3 KB per Fock state. Size batches on that peak, not on
  the output tensor: a 200-point batch at 490 k states asked for 18 GB.
- **Keep the stored states on the host and build the fidelity Gram in device
  blocks.** The Gram itself is cheap (≤ 5 s at 3.1 M states); only holding all
  800 states on the device is not.
- **The GPU buys memory, not much speed.** At 77–116 k states the L4 (float32)
  was only about 1.6–1.8× faster than an Apple M5 CPU (float64). Its value was
  extending the reach from 14 to 18 modes. Time is close to linear in the Fock
  dimension (about 75–90 µs per state per 800-point bank).
- **Version:** the code uses the 0.2.x `MeasurementStrategy` enum
  (`AMPLITUDES`, `PROBABILITIES`); the 0.4.x factory API needs a port. MerLin's
  `DetectorTransform` (per-mode detectors) may replace the hand-written
  no-click features; not tested.

## Scaling limitations

- Statevector cost doubles per qubit: n = 20 needed up to 21 GB of GPU memory
  and about 3.7 h per 5 seeds on an L4 (float32). n = 21–22 is about the
  ceiling for one 23 GB GPU; the paper's n = 28–30 needs petaflop-class
  hardware.
- Float32 is adequate: `d` changed by about 1e-3 relative and the labels by
  about 6e-6 against float64.
- Costs from small-n smoke runs underestimate large-n batches (a predicted
  2.5 h became 3.7 h per batch), so measure at the largest n before quoting a
  time limit. A probe at the target sizes fixed this for the photonic runs:
  production simulation times matched the probe within a few percent.

## Negative and partial results

- The QK-engineered claim (no QK advantage at large n) holds for E1 (tie) and
  E3 (the fidelity kernel ties at n = 16 and loses at n = 20, with
  `d_QK` = 795/800) but not for E2, where the QK advantage persists to n = 20.
  The paper's n ≈ 28 statement is untested.
- The paper's absolute prediction errors are not matched on some tasks:
  Dataset (C) errors for the E2/E3 kernels are far below the paper's (for
  example E3 at n = 16: 0.22–0.27 vs 0.38–0.46), and the (Q, E3) classical MAE is
  about half the paper's at n = 8. Both persist under the sensitivity checks
  tried.
- The photonic PQK did not beat the classical suite on the photonic quantum
  dataset, even though it did on engineered labels.

## Debugging heuristics

- **Test label-free quantities first.** `d` and `g` need no training, run in
  seconds, and expose convention errors (scale, E2, E3) before any model fits.
- **Predeclare the selection rule, with a tie tolerance, before launching a
  convention sweep,** and write the full candidate table to a curated file so
  the choice can be re-derived.
- **Unit-test against dense references.** Gate and embedding unit tests against
  `scipy.linalg.expm` and photonic tests against direct permanents caught
  convention errors that would otherwise look like physics.
- **Check a refactor by rerunning one stored unit** and comparing all recorded
  values (max difference 0.0 for the photonic refactor).
- **Bridge across machines.** Before combining results from two machines, rerun
  the same small config on both and compare geometry (matched to 1e-11) and task
  metrics (up to 0.007 for tree models, from library versions).
- **Record the device per run** so that timing tables cannot attribute GPU hours
  to the CPU machine.
- **Probe before a long GPU campaign.** One bank per size, sorted by cost, with
  out-of-memory errors recorded rather than fatal, gives the reachable sizes,
  memory and time in minutes. Initialise CUDA before calling
  `torch.cuda.reset_peak_memory_stats` (it raises "Invalid device argument"
  otherwise).

## Metric ambiguities

- Report the protocol with every number: `paper` (best test score) vs `cv`
  (5-fold). The two differ by about 0.01–0.03 accuracy at N = 600.
- Effective dimension, the `g_gen`/`g_tra` split and the λ rule follow the
  paper's appendices; an implementation that uses a fixed regularizer or drops
  the normalization `Tr K = N` computes a different `g`.
- "Advantage" must be tied to the label type (quantum-generated, engineered, or
  natural) and to the baseline suite.

## Future directions

- Resolve the effective input scale (author contact, or a joint fit across
  embeddings) and see whether the stronger classical baseline disappears.
- Run n = 21–28 on larger GPUs to test the QK-engineered claim and the >20%
  accuracy gap at the paper's scale.
- Add photon loss, distinguishability and shots to the photonic study, and test
  native detector support in MerLin.
- Use `g` as a screening step before training any photonic kernel: it predicted
  where the projected kernel helped (PB > PA, E3 > E2 > E1).
