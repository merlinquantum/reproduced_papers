# Insights — photonic QCBM trained with a classical MMD estimator

Durable observations from reproducing arXiv:2603.08793. Evidence for each point is in
`LOG.md` (sections named in brackets) and `results/`.

## Implementation pitfalls

- **Butterfly meshes need m = 2^k.** The original butterfly indexes modes `i + 2^l` without
  bounds; at m = 100, 84 of 4950 MZIs act on modes ≥ m, and the "unitary" has singular values
  from 0 to 3.6 even at initialization. Training still lowers the loss, which hides the bug.
  Check `‖U†U − I‖` after every training run and refuse to sample above about 1e-8.
  [Phase 03 Results, D3]
- **Only n columns are needed.** ⟨W_k⟩ needs U_S† diag(±1) U_S (m×n), not the m×m product:
  O(mn²) instead of O(m³), identical values, about 8× faster at m = 100. Compiling the whole
  training step (operator sampling + estimator + Adam) with JAX matters as much as the column
  trick. [Phase 02 Implementation Record]
- **Near-identity initialization depends on the mesh convention.** The MerLin
  `CircuitBuilder` mesh is the identity at internal phase π, not 0.
- **Seed-dependent local optima at small m.** On boson-sampling data with m = 12, n = 3, some
  seeds started near identity end 5–8× worse on the paper's estimator (the training objective) and up to 30×
  worse on count-based metrics. Rerunning them for 4000 steps with no early stopping does not
  help, so the cause is a poor local optimum, not the 200-step stopping rule. Parity features
  cannot see modes holding ≥ 2 photons, which widens the gap on count-based metrics. At
  m = 256 the five seeds were tight. Report per-seed values or medians, not only mean ± std.
  [Phase 07 entry in LOG.md; notebook §6]

## Metric and baseline fairness

- **Check the no-collision assumption on the data.** Boson-sampling data at m = n² has
  collisions in about 60% of samples. Raw-count Gaussian MMD, binarized MMD and the parity
  (Walsh) features the QCBM trains on then rank models differently. A binary model (RBM) scores
  7–14× worse on raw counts than on binarized samples just because it cannot emit counts. Report all three representations
  and choose the primary one by what every model can express. [D5, Phase 5 Comparison Plan]
- **Same kernel for every model.** The paper's "QCBM overtakes the RBM" (Fig. 3e) reappears
  only when the QCBM's polynomial-kernel score is compared with the RBM's Gaussian score. With
  one kernel and estimator, the RBM wins under all three kernels. [Gap Diagnosis — Fig. 3e]
- **The distinguishable-photon twin is the right control.** The same circuit and training with
  distinguishable photons gives exact, cheap training (⟨W_k⟩ factorizes) and classical
  sampling. On boson-sampling data it beats both RBMs; the QCBM leads it by only 1.8–2.8×. Because B5
  is trained separately, that lead is an upper bound on the interference contribution: at
  m = 16 the QCBM's circuit sampled with distinguishable photons still beats B5. To isolate
  interference, also evaluate the trained QCBM circuit with distinguishable photons.
  Any "quantum model suits quantum data" claim should be stated against this twin.
  [Phase 5 Results]
- **First-order baselines are strong on real fixed-weight data.** An independent-marginal
  sampler (Gumbel top-n on training marginals) beats the trained QCBM on sushi, movies and
  bioinformatics. On movies the QCBM matches marginals but captures no pairwise
  co-occurrence, and puts 9–11% of samples on collision outcomes the data never has.
  [Phase 03 Results]
- **Tuned baselines can be optimistic on validation.** The tuned RBM's selected validation MMD
  (about 1e-4) was 6–11× below its test MMD after refit, with large seed variance. Selecting
  among many noisy candidates and refit instability (PCD, lr 1e-2, 512 hidden) are likely
  causes. Compare validation and test of the selected candidate before calling a baseline
  competent. [Phase 5 Results]
- **Weighted-Gaussian normalization.** With the paper's sum normalization the kernel is almost
  flat: every model scores at the noise level. Use the code's mean normalization if the kernel
  must discriminate. [D4]

## Photonic and MerLin patterns

- **At paper scale MerLin is a compiler and a sampler, not a SLOS simulator.**
  `CircuitConverter(..., input_specs=["el"]).to_tensor` gives a differentiable unitary at
  m = 256. `photonic_engines.clifford.fock_sample` gives seeded, exact boson samples on GPU.
  SLOS is exact only up to roughly m = 16, n = 4.
- **Perceval's Clifford2017 sampler is not seedable** (`pcvl.random_seed` ignored). Training is
  bitwise reproducible across machines; only sampled metrics move, by 0.3–0.7%.
- **Native meshes train poorly from near identity.** Rectangular-mesh QCBMs started near
  identity end 20–85× worse than a QR/Haar parametrization on the same task, because light
  barely propagates. Use Haar/QR, or an initialization that spreads photons.
- **Noise.** Uniform loss followed by postselection on n photons leaves the distribution
  unchanged; only the event rate drops. With PNR detection the QCBM's median is flat down to V ≈ 0.8 and stays
  below its distinguishable twin at every V, even V = 0, so part of that edge is optimization rather
  than interference. Threshold detection with collision-free postselection keeps 47% of events and
  removes the edge over the marginal sampler. Plan on PNR detectors for this model class.
  [Phase 04 Results, M3]

## Scaling and compute

- Per step at m = 100, n = 10: 0.14 s on an L4 and 0.88 s on an Apple M5, against the
  paper's 3–4 s on a laptop. At m = 256, n = 16: 0.21 s/step on an L4. Most runs stop in
  400–1600 steps, so a 5-seed figure takes GPU-minutes to hours.
- Training is hardware-independent bitwise (CPU vs GPU, two GPUs), so test MMD from different
  machines can be compared; wall-clock cannot.

## Open directions

- Datasets with genuine higher-order structure, where a marginal sampler should fail, would
  test whether the QCBM's inductive bias helps outside boson-sampling data.
- Characterize which trained unitaries are close to Fourier or permutation structure (the
  paper's own simulability caveat).
- A seeded, documented large-m sampler and a public unitary compiler in MerLin would make this
  class of models easy to reproduce.
