# Insights — Adaptive Boson Sampling kernels (Hoch et al. 2025)

Durable observations for future reproductions of photonic kernel and
post-selection papers. Evidence for each item is in `LOG.md` and `results/`.

## Implementation pitfalls

- **Fix conventions on the one printed number.** Use platform A's printed kernel
  (0.298 / 0.966 / 0.331) as the gate for the MZI convention
  (B·diag(e^{iθ},1)·B with B = [[1,i],[i,1]]/√2) and the outcome order. Several
  plausible conventions give different kernels. Settle this before simulating
  anything that has only a heatmap target.
- **Index rules can be misread.** The B2/B3 angle rule only gives 15 distinct
  indices with one reading of (first, second) detector. Check that the inferred
  rule produces exactly D distinct outcomes.
- **Basis order of residual modes.** MerLin branch amplitudes and a hand-written
  Fock basis can order states differently. Assert the order at run time rather
  than trusting it.

## Unpublished randomness decides the result

- **Fit the hidden unitary to the paper's own figure.** The paper's random U_0
  is unpublished. Haar U_0 destroys the kernel structure (r ≈ 0.2), while
  gradient descent on an autograd 3×3 permanent recovers it (r = 1.00 on the
  fitted panel). The real test is the **unfitted** experimental panel
  (r = 0.96-0.99). Several fitted U_0 exist; check that downstream quantities
  agree across fits. Here 1D accuracies agree within about 0.03, but the share of
  alternative kernels in range moved from 81-82 % to 75-96 % between fit sets.
- **Discrete encodings hide an assignment choice.** Mapping K-means clusters to
  outcomes is unpublished, and it moves moons accuracy from about 0.56 (random)
  to 0.99 (moon-following). Report this as a declared sensitivity sweep with no
  winner. A rule found after seeing the paper's numbers stays labelled post hoc.

## Metric and protocol ambiguities

- **Tiny test sets amplify everything.** With 15 points and 80/20 splits, each
  test set has 3 points. The partition protocol (own seeds vs sklearn), C, and
  0.01 digitisation error each move 1D accuracy by 0.04-0.10, which is as much as
  the paper-vs-reproduction gap. Quantify these sensitivities before calling a
  gap a discrepancy.
- **Std conventions.** Keep one ddof convention across summary scripts. Here
  0.073 and 0.077 described the same run.

## Baseline fairness

- **Match information, then dimension.** For outcome-indexed quantum kernels, use
  two matched classical kernels:
  - a smooth kernel on the same outcome index, which matches the information;
  - random rank-d kernels |<u_i|u_j>|², which match the feature-space dimension.

  Raw-coordinate SVMs see more information and are only an upper reference.
- **"Higher dimension helps" can be generic.** The qutrit > qubit gain on moons
  (+0.16 to +0.25) is matched by random rank-3 vs rank-2 kernels (+0.24 to
  +0.29). Any qudit-dimension claim needs this control.
- **Know whether the task tests generalisation.** In moons + K-means every
  outcome appears in training, so it is close to a lookup task: any kernel of
  enough rank memorises it. Only the 1D task, where test points are unseen
  outcomes, tests the kernel's inductive bias. There a classical Gaussian outcome
  kernel (0.96) beats every ABS kernel (0.75-0.80).
- **Permutation controls can be too harsh.** Permuting a classical kernel's
  outcome order destroys locality, while ABS re-assignments keep band structure.
  The resulting gap (81-94 % vs 2-8 % in range) shows structural robustness, not
  advantage.

## Photonic translation and MerLin patterns

- **Heralded / post-selected states.**
  `MeasurementStrategy.partial(detected_modes, ComputationSpace.FOCK)` gives one
  branch per outcome, with the residual-mode `StateVector`. This is the direct
  MerLin primitive for ABS-like feature maps. Use FOCK when post-selected
  patterns can be bunched.
- **Source noise needs a workaround.** With `pcvl.NoiseModel`, only `probs(FOCK)`
  was usable. Mixed residual states then come from simulated tomography: append
  measurement settings, linear inversion, PSD projection.
- **Exactness check first.** Before adding noise, verify that the exact MerLin
  path equals an independent simulator (here to 4e-15). Compare classifier
  outputs with a tie-aware criterion, since one SVM fit in 25,000 flipped on a
  1e-15 kernel difference inside libsvm's tolerance.
- **Hardware cost of post-selection.** Each data point is a post-selected outcome
  with probability down to 3e-4. A hardware kernel entry needs about 1/p shots
  per state plus tomography. Report this next to any accuracy result.

## Negative results

- **The B1 kernel (Fig. 3e) is not reproduced** by the published circuit. The
  outcome order, 176 circuit conventions, and partial-distinguishability or
  multiphoton noise were all ruled out (r ≈ -0.1). The likely cause is the chip's
  calibrated phases, which are not published.
- **No accuracy advantage** of ABS kernels over matched classical kernels was
  found on any task the paper uses.

## Future directions

- Obtain the author data: U_0, the moons sample and assignment, and the B1
  calibration.
- Test C6: can any single non-adaptive circuit reproduce the post-selected
  kernel set?
- Look for tasks where outcome-indexed kernels must generalise to unseen
  outcomes and the classical Gaussian outcome kernel is not already near
  ceiling.
