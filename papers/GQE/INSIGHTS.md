# Insights — Generative Quantum Data Embeddings (EGAS)

## Scientific
- **The Wasserstein diagnostic is the most robust part of the paper.** Even under a heavily
  reduced search, embedding-sensitivity (IQR of QKSVM accuracy across embeddings) tracks the
  input-space 1-Wasserstein distance: low-W1 datasets (Wine-Quality, MAGIC) show tightly
  clustered accuracies (saturation), higher-W1 datasets show a wider spread. The EGAS *search
  itself* reflects this — the surrogate energy barely decreases on low-W1 datasets, because no
  embedding in the family can pull weakly-separated classes apart. This is a genuinely useful,
  cheap, a-priori screening criterion and reproduces with simple preprocessing.
- **EGAS reliably beats the data-agnostic ZZ feature map; it is competitive with, but does not
  dominate, NQE**, under reduced search. The gap to the paper's stronger EGAS results is
  consistent with using 120 vs 4000 search iterations.
- **Fair-baseline caution:** on these UCI tabular tasks a plain linear SVM on standardized PCA
  features is a strong baseline. Quantum-kernel embeddings only edged it on the higher-W1 dataset
  in our reduced runs. Any "quantum advantage" reading must be made against this classical
  baseline at matched features, not just against ZZ.

## Implementation pitfalls
- **Logit-matching loss (Eq. 10) is numerically fragile.** `w_sum` is an unbounded sum of raw
  GPT logits, so `exp(-γ·w_sum)` overflows and the loss explodes for `γ≈1`. Fix: small `γ`
  (0.1), clamp the exponent arguments, EMA-normalise energies, and gradient-clip the GPT.
- **Pure-state fidelity = one matmul.** Since every embedding yields a pure state, the full N×N
  fidelity/kernel matrix is `|Ψ Ψ†|²` over precomputed statevectors — compute statevectors once
  per embedding, then Gram. This is the single biggest speedup for both the EGAS energy and the
  QKSVM kernel.
- **Autoregressive GPT sampling dominates EGAS wall-clock**, not the quantum simulation: 28
  sequential transformer forwards per candidate are Python-dispatch-bound. When CPU-limited,
  reduce *iterations*, not model size.
- A custom batched torch statevector engine (gates applied to all samples at once) validated to
  machine precision against PennyLane is both faster and differentiable — the latter is needed
  for the continuous bias-refinement stage.

## Photonic translation
- The gate-based fidelity-kernel QKSVM maps naturally to MerLin's `FidelityKernel`; the photonic
  analogue of EGAS's discrete architecture search is a *trainable interferometric mesh* optimised
  with the same pairwise-fidelity surrogate.
- **The photonic translation works**: on MGT, a *trained* photonic mesh (2 photons, 8 modes) reached
  0.733±0.025 vs 0.687 fixed and 0.487 for the data-agnostic ZZ map, approaching the classical
  linear baseline (0.753). The EGAS inductive bias (optimise pairwise fidelity for separability)
  survives the photonic mapping even at minimal photon count.
- **Training a feature-map mesh through `FidelityKernel` is expensive** (SLOS autograd, ~2.5 min per
  split for 12 epochs at 2 photons), which bounds practical photonic experiments to few photons /
  epochs / splits. Forward evaluation is cheap; prefer fixed or lightly-trained meshes when CPU-bound.

## Metric / preprocessing ambiguities
- Table I absolute W1 values are sensitive to the (underspecified) rescaling. Per-feature MinMax
  to [0,2π] caps per-component class separation, so the most-separable datasets (Dry Bean, Wine
  Color) report smaller W1 than the paper, even though the qualitative ordering and the
  low-W1-⇒-saturation conclusion hold.

## Future directions
- Compare EGAS against a *random* circuit-sequence search at matched evaluation budget to isolate
  the GPT's contribution (the paper does not).
- Multi-seed runs and the full 4000-iteration search to test whether EGAS consistently overtakes
  the classical linear baseline on higher-W1 datasets.
