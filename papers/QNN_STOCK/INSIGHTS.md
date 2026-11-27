# Insights — Contextual QNN for Stock Prediction (QNN_STOCK)

Durable lessons from this reproduction, for future reproductions of
superposition-trained, SWAP-test or "contextual" quantum models. Evidence and
run records are in `LOG.md`; results in `README.md`.

## Scientific findings

- **A loss on one superposed input does not constrain per-input behaviour.**
  QBGU trains on $\sum_c \sqrt{P(c)}\lvert c\rangle\lvert 0\rangle$ only. A
  circuit that acts on the context qubits can match the joint target while
  mixing contexts, so single-context inference is at chance. Always check
  context retention, $\lVert(\langle c\rvert\otimes I)\,U\lvert c,0\rangle\rVert^2$,
  and evaluate with the readout the paper uses at prediction time.
- **Context-preserving circuits are classical additive models.** With
  context qubits as controls only, $q(\text{up}\mid c) = \sin^2(a_c/2)$ with
  $a_c$ linear in the bits: logistic regression with a different link. It
  ties B2 at every data size (25 windows to full) and at 10 qubits.
- **The QBGU fidelity of a context-preserving circuit is a Hellinger-type
  classical loss** with the calibrated conditional as optimum. The per-sample
  one-hot fidelity of the paper's Eq. 18–19 is linear in $q$, so it is
  mode-seeking: same accuracy, overconfident probabilities (+0.5 to +0.85
  nats). Eq. 20's "QBGU = full-batch SGD" does not hold (gradient cosine
  about 0).
- **Interference between predicted paths did not help.** A circuit with
  signed amplitudes over multi-step paths was equal or worse than the
  interference-free chain-rule circuit in every cell, including with L-BFGS
  on the exact loss.
- **Photon number sets the polynomial degree, not interference.** For binary
  inputs the degree is capped at $T$ anyway. In the MerLin adaptation, 1 vs 2
  photons and distinguishable vs indistinguishable photons were not
  distinguishable in 10 of 12 cells.
- **Compare losses on a shared learning-rate grid.** The paper's "MSE fails"
  (Fig. 17) used lr $10^{-5}$ for MSE against 0.1 for the fidelity loss. On a
  shared grid both pick 0.1, and MSE then fits the measured distribution
  slightly better (train KL $5 \times 10^{-5}$ against $10^{-3}$). Only the
  fidelity loss fixes the phases ($F$ 0.999 against 0.07), which matters only
  if the state is used coherently afterwards.
- **Raw daily direction is a coin flip** (best test log-loss 0.691 vs
  $\ln 2 = 0.693$). Use a smoothed series (MA5) as the setting where a
  difference could show; ties on raw data carry little information.

## Implementation pitfalls

- **Trained amplitude loaders are phase-blind.** An HEA loader fitted with MSE
  on measured probabilities gets $\lvert\psi_c\rvert^2$ right (TVD 0.002) and
  the phases random (state fidelity 0.00–0.57). It looks harmless with a
  circuit that can undo phases on the context register, and hurts any
  circuit that keeps the context (test log-loss +0.12 to +1.08).
- **Random initialization throws away a near-solution.** The exact input is
  one RY away from fidelity 0.88–0.999, but $\theta \sim U(0, 2\pi)$ starts at
  about $1/2^n$; slow SPSA convergence came from the start point. Small
  positive angles ($U(0, 0.3)$) fix the sign trap of squared-overlap losses.
- **Report training fidelity against a trivial reference.** On raw data a
  50/50 circuit already reaches $F \approx 0.999$.
- **Fig. 16-style loss curves above 0.5** mean the paper plots $1 - F$, not
  the Eq. 14 loss $(1 - F)/2$.
- **Read circuit figures gate by gate.** The QMTL asset block has no CNOT
  (Fig. 7); a full layer adds an unconditional entangler per layer. Found
  only after a first sweep.
- **SPSA with 100 shots per evaluation does not move**; trading shots for
  iterations failed for both objectives.

## Baseline fairness

- The frequency table (B0) is the maximum-likelihood estimate of the target;
  a train-target KL near 0 only says the model fits B0. Compare on a test
  split.
- Matched pairs that made the conclusions clear: 4-parameter context-
  preserving circuit vs 4-parameter logistic regression; QMTL vs 2-layer
  QSTL at equal epochs and parameters per asset.
- For parity claims ("learns", "equals"), a 95 % interval containing 0 is
  not equivalence; set an equivalence margin before the table.

## MerLin patterns

- A trainable interferometer with context bits as phases $\pi c_i$ on modes
  $1..T$, two re-uploads, and "detector on mode 0 clicks" as $q(\text{up})$
  slots into a classical training loop with exact gradients
  (`lib/photonic_merlin.py`).
- A mean-photon-number readout is linear in single-photon probabilities and
  identical for 1 photon, distinguishable and indistinguishable photons; use
  a detection-event readout to make photon statistics matter.
- Source noise (distinguishable photons) forces FOCK; asking for UNBUNCHED
  with a noise model silently returns FOCK outputs.
- Threshold detection with collision-free postselection kept 48–61 % of
  shots and was worse than PNR in 4 of 12 cells.

## Scaling and resources

- At 10 qubits with 10,000 shots, SPSA no longer trains the interfering
  circuit ($F$ 0.03–0.47); shot-free losses fix it in simulation only.
- The target state encodes a table of up to 960 cells for $T = 6$, horizon 4,
  which 3,000–8,000 training windows cannot fill; the frequency table then
  loses to logistic regression, and no circuit beats logistic regression.

## Future directions

- Noise study (Fig. 22) was not run; it would only characterize one model.
- A data-defined superposed input is the method's real requirement and has
  no linear-optics counterpart; a photonic QBGU is not a promising target.
