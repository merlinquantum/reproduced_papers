# Insights — Trainable-Frequency QNNs (Jaderberg et al., PRA 2024)

Durable lessons for future reproductions of Fourier-spectrum, data-encoding and
physics-informed QML papers. Evidence: `README.md`, `results/`, `LOG.md`.

## Scientific findings

- **Trainable frequencies help when the data's frequencies are irregular, not merely
  rescaled.** On {1, 1.2, 3} TF beats FF by ~40× in MSE, and neither a global input scale
  nor 32× more FF layers closes the gap. On {1.5, 3, 4.5} (an integer set times 1.5) a
  single global scale suffices, provided it starts near 1.5.
- **Frequency parameters move only locally under gradient descent.** Learned TF weights
  stay within ≈ 0.15 of their initial value (0.89–1.15 from 1.0). A global scale started at
  1 never reaches 1.5. The loss is highly non-convex in a frequency parameter, so
  initialisation decides which spectrum is reachable. Expect this in any trainable-encoding
  paper.
- **Input scaling is the dominant confounder of "trainable encoding" advantages.** On the
  Navier–Stokes DQC task, FF with 3 trainable input scales per network (FF+S) recovers
  most of the TF gain (Table I) or all of it (Table II). Raw physical coordinates leave
  deep FF models on a plateau that more iterations do not fix. Always run an FF+S or
  normalised-input control before attributing gains to per-gate trainable frequencies.
- **Integer-gap FF maps are indistinguishable on non-integer data.** Simple, tower and
  exponential FF maps gave identical errors on `linspace(1, 3, k)` for k ≥ 4. The paper's
  Fig. 3 differences between maps appear only when FF inputs are scaled by the data's
  base frequency, and then exponential FF beats TF by orders of magnitude.
- **"TF contains FF, so it is never worse" holds at initialisation, not after training.**
  At lr 1e-2, TF's worst case on Fig. 3 was worse than every FF map.
- **Check stated cost factors.** The paper's C_f = 1.07 counts 12 frequency parameters
  against 180 ansatz parameters for two networks; counted consistently it is 1.033.

## Photonic translation

- **Phase-shifter encoding is the natural photonic TF map.** `PS(φ = γ·w·x)` gives
  frequencies equal to photon-number differences times γ·w: integer for FF, learnable for TF.
- **The readout determines the spectrum.** One-body ⟨n_k⟩ sees only ω ≤ 1. Collision-free
  post-selection (`UNBUNCHED`) renormalises and leaks power above the circuit's spectrum.
  Threshold clicks from `FOCK` probabilities keep the finite spectrum. Choose and test the
  readout with a DFT before any spectrum claim.
- **Multi-photon frequency coefficients are capped in linear optics.** With γ = 1 on 3
  encoding modes and 3 single photons, the ω = 3 coefficient stays near 0.05 (target 0.33),
  because it needs a NOON-like superposition. Carrying high frequencies on one photon
  through a mode with large γ (tower / exponential) avoids the cap. "More photons ⇒ larger
  spectrum" is true for the frequency set, not for the reachable amplitudes.
- **Consequence:** a TF model initialised as the photonic simple map inherits the cap, and
  tower / exponential FF beat it. In photonics, choose γ deliberately, then make it
  trainable.
- **Photonic PINNs are feasible.** Third-order input derivatives through `QuantumLayer` are
  exact under autograd. Expect ~1–2 s per iteration for 2 networks × 600 points at 3
  photons / 6 modes. Photonic FF with raw coordinates did not train on Navier–Stokes;
  photonic TF mostly learned to shrink or switch off input coordinates.

## Implementation pitfalls and heuristics

- **Simulate R_y-encoded circuits in the Pauli-Y eigenbasis.** Encoding blocks become
  per-sample phase vectors and ansatz layers fuse into dense matrices. This is exact to
  1e-12 against gate-by-gate evaluation and 58× faster, which made 10-seed Navier–Stokes
  runs affordable.
- **Pair seeds by drawing the ansatz before any frequency parameter exists.** FF, FF+S and
  TF then start identical, and per-seed paired tests become possible.
- **MAERM ambiguity:** "relative to the median" must be median(|y|). The signed median of
  v is ≈ 0 and inflates errors above 100 %. The choice was pinned by matching the paper's
  TF v error exactly (30.4 %).
- **Infer missing layer counts from stated parameter totals** (App. C: 804 parameters →
  67 layers) before declaring a paper underspecified.
- **Diagnose gaps with one-component tests on one seed** (longer training, normalised
  inputs, scale control) before multi-seed confirmation; three runs isolated the E7 cause.

## Baseline fairness

- The paper's own baseline (identical circuit, frequencies frozen) is the right control
  for the mechanism. It is not sufficient for an accuracy claim, because input scaling is
  free capacity that FF lacks.
- Plan ≥ 6 paired seeds for any control meant to be decisive: with 5 seeds the minimum
  two-sided Wilcoxon p is 0.0625.
- A small classical MLP PINN beat every quantum model on Navier–Stokes. Report such
  context models even when the paper makes no quantum-vs-classical claim.

## Promising directions

- Photonic TF initialised from tower / exponential γ, and input states that lift the
  multi-photon coefficient cap.
- Trainable frequencies combined with learned input normalisation, to separate the two
  effects cleanly.
- Shot-noise and loss models for the photonic threshold readout.
