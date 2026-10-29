# Insights — QELM and Concentration Barriers (arXiv:2603.13005)

Durable lessons from the reduced-scale reproduction and its MerLin adaptation. Evidence for
each point is in `LOG.md` and `results/`.

## Implementation pitfalls

- **The gate angle factor is not printed in the paper.** It decides the regime. Reading
  each gate as `exp(-i θ P)` (`angle_scale = 2`) reproduces the paper's Fig. 2 MPS bond
  dimensions (8 / 14 / 28 / 50 at 6–12 q). Check any kicked-Ising reproduction against a
  published capacity number before running tasks.
- **Circuit layout matters more than the MPS protocol.** An early bond-dimension mismatch
  came from the circuit (block order, Bell pair offset), not from the truncation settings.
- **The NARMA input map in the paper is inconsistent.** `u ~ Unif(0, 0.5)` with `2u − 1`
  does not give Unif(−1, 1). Use `4u − 1` and keep the literal map as an ablation.
- **The NARMA score range is inconsistent.** The search score sums orders 1..N, although
  the text says 1..N/2; the Fig. 2 stars exceed the 1..N/2 maximum.
- **Statevector memory at 24 qubits.** One complex64 state is 128 MiB. Chunk the samples
  (16 per chunk peaks at 13.7 GB on a 22 GB L4; 32 per chunk runs out of memory).

## Shot noise, input scaling and the Landsat gap

- **The F1 gap at 1e5 shots is shot noise.** With exact probabilities, the 16-qubit
  feature map already reaches the paper's F1 (0.79 w1, 0.82 w2). Always run an
  exact-probability control before blaming the circuit or the pipeline.
- **Input scaling is the dominant unstated factor.** z-score inputs instead of min-max
  [−1, 1] give larger input rotations (|u| up to about 3), more input-dependent signal per
  shot, and +0.05–0.08 F1. When a paper leaves the encoding scale unstated, run both
  readings and declare one primary before seeing results.
- **Depth is not a free lever.** More uploads through more layers concentrate the outputs
  and lose to shot noise; shallower circuits do not help either.
- **Save exact intermediate probabilities.** Storing exact pair POVM probabilities per
  realisation makes shots, cutoffs and readouts cheap to re-sample. It also lets a
  timed-out run be resumed by reuse, guarded by checks on the config and the split.

## Eigentasks and NSR readouts (negative result)

- At N ≤ 24 without hardware noise, the paper's readout engineering gives no gain (paired,
  n = 45, both input scalings):
  - local eigentasks ≈ Pauli features;
  - the NSR cutoff never beats keeping all eigentasks, and F1 rises monotonically with λ;
  - NSR or signal-aware scaling ≈ or below unit variance.
- **The paper's margins are small.** They are 0.01–0.03 F1, from one hardware run on one
  split, while our per-evaluation std is about 0.03. Treat single-run readout orderings as
  unconfirmed until multi-seed.
- **The retained-eigentask fraction depends on the input scaling.** It is 37.5 % with
  min-max (paper 32 %) and 71 % with z-score inputs, so it is not a stable target across
  preprocessing choices.
- **Open question:** eigentask benefits may need hardware noise, which is not shot-like,
  or the larger NSR spread of deeper or larger circuits.

## Baseline fairness

- **Match readout dimension, not raw input size.** The paper's classical models see 36 raw
  features while the QELM's ridge sees 372–930. Against a random-feature map of the same
  dimension, with the same ridge, splits and input scaling, the qubit QELM ties or trails.
- **Beating ridge on raw features shows separability, not advantage.** Every QELM beats
  raw ridge (0.59 vs 0.72–0.81).
- **The strongest random-feature family depends on scaling and dimension.** With z-score
  inputs, tanh features beat random Fourier features from 16 to 72 features (by +0.08 at
  16) and tie them at 120–180. With min-max inputs, Fourier wins at every dimension
  (48–180). Plan the baseline as "best family on validation" rather than a fixed family.
- **Baseline hyperparameters must follow the input scaling.** A Fourier bandwidth set for
  min-max inputs is suboptimal for z-scores, and this can flatter a small-dimension quantum
  model.
- **On NARMA, a QELM is a nonlinear window model.** It never beats degree-2 window ridge or
  an ESN at any size.

## Photonic translation (MerLin)

- **Single-photon correspondence.** For one photon in a mode pair, BS(θ) ↔ RX(θ) and
  PS ↔ RZ. That carries the kicked-Ising single-qubit angles over. The ZZ interaction has
  no linear-optical counterpart, so correlations come only from multi-photon
  interference.
- **Use `FOCK` with threshold detectors for shot-noise studies.** `UNBUNCHED` renormalises
  on collision-free outcomes (only 20 % of the mass at 8 modes / 4 photons), which is a
  hidden postselection.
- **Click features are cheap to estimate but plateau lower.** At 100 shots, photonic F1 is
  0.65 against the qubit's 0.24. At 1e5 shots it is 0.74 against 0.77.
- **Photonic click outputs vary by only about 3–5 % across inputs.** Additive Gaussian
  feature noise at the paper's 10^-1.5 swamps them, while true 1e5-shot noise costs only
  2–3 %. Model readout noise as sampling, not additive Gaussian, for photonic reservoirs.
- **Concentration has a photonic signature.** Haar-random interferometers lose variability
  as modes grow: 3–6x from 8 to 24 modes at every depth (1.2e-3 → 3.8e-4 at 16 layers). A
  structured, weakly mixing brickwall loses only 1.2–1.5x at 4–8 layers, but 2.3x at 16 layers,
  and stays 3–15x above Haar. Unlike the qubit case, variability rises with depth, because deeper circuits
  re-encode the input more often.
- **Validate MerLin outputs independently.** Pair click marginals checked against a
  permanent calculation from the circuit unitary (to 1e-9) also catch input-parameter
  ordering errors.

## Scaling limitations

- Exact simulation stops at about 24 qubits. The paper's 72/124-qubit claims need hardware
  or approximate MPS.
- At 24 modes / 6 photons, the full threshold-pattern distribution makes MerLin Landsat
  runs about 30x slower than at 16 / 4 (44 min vs 1.4 min on CPU).

## Debugging heuristics

- Compare train and test F1 under noise. Both fell together here, which points to
  noise-induced underfitting, not overfitting.
- Check classical baselines against the paper's table first. If they match, the dataset
  and metric are not the cause of a QELM gap.
- Give each parallel run its own output directory; second-resolution run names collide.

## Future directions

- Re-optimise the photonic regime with a photonic capacity measure (an E1 analogue).
- Add photon loss and partial distinguishability to test how robust the photonic regime is.
- Tune random-feature bandwidths on validation to settle the small-dimension photonic
  margins.
- Test eigentask readouts under a realistic, non-shot-like hardware noise model.
