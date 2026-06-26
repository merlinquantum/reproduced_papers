# Action Required: Send to MerLin Team

## Summary
MerLin 0.4.0's `FidelityKernel` is a great fit for reproducing quantum-kernel-SVM papers. Two
items would materially help: (1) the cost/feasibility of *training* a feature-map mesh through
`FidelityKernel` via autograd, and (2) minor API-discoverability friction.

## Context
Reproducing arXiv:2605.30866 ("Generative Quantum Data Embeddings"), a gate-based paper whose
embeddings are scored by a fidelity quantum kernel `K=|⟨ψ(x_i)|ψ(x_j)⟩|²`. The photonic
counterpart uses `merlin.FidelityKernel` over a `CircuitBuilder` angle-encoding + entangling
feature map, ≥2 photons, UNBUNCHED. Forward kernel evaluation worked cleanly; training the mesh
parameters (continuous analogue of the paper's architecture search) was the bottleneck.

## Items to Send

### High — FidelityKernel: training-through-kernel performance
- Type: pain point / performance
- What happened: backpropagating a pairwise-fidelity loss through `FidelityKernel.forward` to
  optimise the feature-map's trainable mesh parameters costs ~10–13 s/epoch for 2–3 photons in
  8 modes with a 30–36-sample batch (so a 40×40 kernel per step). This made full
  trainable-photonic experiments (multiple datasets × splits × tens of epochs) infeasible on CPU
  and forced a reduced scope (2 photons, 25 epochs, 5 splits).
- Evidence: `lib/photonic.py::train_photonic_embedding`; a 40-epoch run did not complete in ~9
  min wall-clock on the container CPU.
- Why it matters: quantum-kernel-alignment / trainable-feature-map papers are a common
  reproduction target; the forward path is fast but the train path dominates. Even forward kernel
  matrices recompute all pairs each call.
- Suggested action: document expected autograd cost vs photon/mode count; if possible expose a
  cached/lower-overhead training path, or guidance on shot-free analytic gradients and batching.

### Low — API discoverability
- Type: API confusion / missing documentation
- What happened: `CircuitBuilder.input_parameter_prefixes` / `trainable_parameter_prefixes` are
  attributes in some paths but callables in others (handled defensively in our code). The Wine
  "color" / multi-target dataset issue is unrelated to MerLin, but the prefix ambiguity cost time.
- Evidence: `lib/photonic.py::build_feature_map` (the `callable(...)` guard).
- Why it matters: small friction when wiring a `FeatureMap` from a builder.
- Suggested action: a one-line `FeatureMap.from_builder(builder, input_size)` convenience, or a
  doc example of the builder→FeatureMap→FidelityKernel path.
