# INSIGHTS — Photonic Quantum-Enhanced Knowledge Distillation (PQKD)

Durable observations from reproducing arXiv:2603.14898 on MNIST (partial RED-tier: 30 epochs, 3 seeds, CPU). Numbers are final-epoch test accuracy, paired over seeds unless noted. Evidence lives in `results/`.

## Baseline fairness

- **"Photonic gain vs z = 0" is not an attribution control.**
  - When a conditioning vector multiplies the weights (vec(M) = A z), z = 0 zeroes every compressed kernel and gives chance accuracy (10.8%).
  - Any working z then shows the same ~88 pp "gain": photonic 87.78, fixed random 87.83.
  - Use a fixed random z, a frozen θ and a classical generator with the same tuning budget instead.
- **Compression claims need an iso-parameter dense student.** The dense KD student at −0.5% parameters beat the photonic-conditioned student on every seed (+0.16 ± 0.05 pp). Comparing only against the larger teacher hides this.
- **Give the photonic and classical generators the same optimiser and tuning budget.** Here that meant the same SPSA grid for each. With exact gradients, photonic and classical 45-parameter generators were indistinguishable (+0.05 ± 0.10 pp).

## Optimizer and trainability behaviour

- **Accuracy follows the stability of z across epochs, not its source.** Spearman ρ = −0.90 over 18 runs. Arms whose z changes during training (random z redrawn per epoch, SPSA-driven classical generator, sampled classical histograms) lose 0.3–1.0 pp. A fixed z of any origin is best.
- **θ = 0 is a stationary point.**
  - Zero beam-splitter angles make the mesh a fixed permutation: z is deterministic and its σ is 0.
  - The exact gradient of θ is zero there. In the gradient-trained arm, the angles never moved in 30 epochs (‖θ‖ = 0.00, 3/3 seeds).
  - SPSA with small steps also stays near it (‖θ‖ ≈ 1–2). Papers that start at θ = 0 never exercise their shot-noise claims.
- **SPSA step size matters a lot and is often unreported.** Large steps with a small perturbation drive every angle to the ±π clip (‖θ‖ = π√45 = 21.07) and give the worst objective. Run a small predeclared grid, and give the classical baseline the same grid.
- **Sampled conditioning can create a hidden deployment dependency.** Re-evaluating with one fresh, unsmoothed z draw:

| Arm | Fresh-z test accuracy, per seed |
|---|---|
| EMA, S = 200 | 99.1 / 21.8 / 13.7 (fails on 2 of 3 seeds) |
| EMA, S = 25 | 10.9 / 10.7 / 15.8 (fails on 3 of 3) |
| no EMA, S = 25 | 14.5 / 33.2 / 96.0 (fails on 2 of 3) |
| no EMA, S = 200 | 99.3 / 98.9 / 94.0 (robust) |

  - Only training on moderate, unsmoothed shot noise produced a student robust to new samples.
  - If z is sampled on hardware at test time, evaluate with fresh samples.

## Photonic translation and MerLin patterns

- **Threshold detection needs `ComputationSpace.FOCK`.** The default `UNBUNCHED` silently post-selects away bunched events. Either threshold manually (clicks = 1[n > 0]) or use `probs(FOCK, occupancy_readout=True)` / `pcvl.Detector.threshold()`.
- **Exact autograd through `QuantumLayer` probabilities is the cheapest strong control in simulation.** It removes shot noise and gradient-free optimisation as excuses: if the photonic module does not help with exact gradients, sampled hardware will not either.
- **`shots=` is ignored while autograd is on.** Implement sampling explicitly when shot noise is part of the claim.
- **Input-independent conditioning needs only one circuit evaluation per θ update,** not one per image. Its cost is negligible next to the CNN at 4 photons.

## Scaling limitations

- **Full-Fock memory grows as C(m + n − 1, n):**

| Modes / photons | Fock states | Memory | Forward + backward |
|---|---|---|---|
| 16 / 4 | 3,876 | ~0.9 GB per process | 5.9 ms |
| 16 / 8 | 490,314 | ~1.8 GB | 0.16–0.19 s |
| 16 / 12 | ~17M | ran out of memory on 7.5 GB | — |

- **Photon number did not change the PQKD result** (4 vs 8 photons: +0.06 ± 0.08 pp). For input-independent conditioning, a smaller photon number is a valid cost reduction.

## Metric ambiguities

- The paper's Table 1 "photonic Δ" (first − final SPSA objective) has undefined units, and it also falls when only the student weights train. With θ frozen, accuracy is unchanged. Treat Δ > 0 as no evidence of θ learning.
- SPSA tunes on the validation split that the paper reports. Use test accuracy as the primary metric.
- The paper's compression factors are reproduced only if each compressed layer counts a per-channel gain plus a bias (R·k² + 2·C_out).

## Negative results worth reusing

- A photonic conditioning vector feeding a dictionary convolution gives no measurable benefit over a fixed random vector on MNIST. This holds:
  - at 8 photons with SPSA;
  - at 4 photons with exact gradients;
  - in a stochastic mixing regime, where it is worse.
- Shot noise and EMA effects reproduce in direction (25 shots −3.93 ± 1.53 pp; EMA +3.32 ± 1.15 pp). They are generic noisy-conditioning effects that a classical redrawn z shares (−1.03 ± 0.36 pp).

## Promising future investigations

- **Fashion-MNIST:** the paper reports PQKD above the teacher there. Test it against the dense iso-parameter student and a fixed random z.
- **Input-dependent conditioning z(θ, x)** via angle encoding: the only route by which a photonic generator could add information beyond a fixed vector.
- **The EMA variance ratio** (Supp. Fig. S1) can be computed from saved feature traces of the stochastic arms at no extra compute.
- **8-photon and 100-epoch budgets on a GPU,** to confirm that the conclusions are not budget artifacts.
