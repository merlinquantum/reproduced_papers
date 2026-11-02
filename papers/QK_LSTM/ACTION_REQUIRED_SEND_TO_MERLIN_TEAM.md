# Action Required: Send to MerLin Team

## Summary

- **Main finding.** `merlin.FidelityKernel` with `computation_space=UNBUNCHED` does not return the fidelity |⟨s|U†(x₂)U(x₁)|s⟩|² that its docstring describes. Both the Gram path `k(x)` and the cross-kernel path `k(x1, x2)` return values that deviate from the true fidelity (computed with `computation_space=FOCK`) by up to 0.20 in absolute value, up to 2.4× as a ratio. This is consistent with renormalisation by the collision-free probability mass. The cross-kernel path is additionally asymmetric (k(a, b) ≠ k(b, a)). FOCK gives the correct fidelity on both paths. This is either a bug or undocumented postselection semantics; both deserve attention, because UNBUNCHED is the recommended default for threshold-detector realism.
- **Documentation gaps.** Two smaller items: using `FidelityKernel` as a differentiable cross-kernel inside a model, and building fixed, seeded meshes.
- **Dependency note.** An old merlinquantum/perceval combination fails under NumPy 2.

## Context

- **What was done.** We reproduced arXiv:2412.08851, a QK-LSTM whose LSTM gates are weighted sums of 4-qubit fidelity-kernel values. As a Phase 4 adaptation, we replaced the qubit kernel with a 2-photon, 4-mode MerLin photonic fidelity kernel (`papers/QK_LSTM/lib/qk_lstm_merlin.py`). It is used as a differentiable cross-kernel `k(z_t, z_ref)` inside backpropagation through time.
- **Why it matters.** Kernel symmetry and correctness matter for anyone using `FidelityKernel` in SVMs (Gram matrices), kernel alignment, or as a learned similarity inside a model. Users following the "start with UNBUNCHED" guidance get a quantity that is not a fidelity kernel without being warned.
- **Environment:** merlin 0.4.1 (`merlinquantum` 0.4.2), perceval-quandela 1.3.1, torch 2.12.1, Python 3.12.4, macOS (Apple M4).

## Items to Send

### High — FidelityKernel: `UNBUNCHED` computation space returns a renormalised kernel (asymmetric in the cross-kernel path)

- Type: bug (probable; possibly intended postselection that is undocumented)
- What happened: for the feature map `fixed MZI mesh (seed 7) → angle encoding on modes 0–3 → fixed MZI mesh (seed 8)`, input state `[1, 0, 1, 0]`, `force_psd=False`:
  - Cross-kernel path: `FidelityKernel(..., computation_space=UNBUNCHED)(z, z)` on 4 random points has `max |K − Kᵀ| = 0.037`. The diagonal is 1.
  - Gram path: `FidelityKernel(..., UNBUNCHED)(z)` is symmetric, but its entries differ from the FOCK fidelity by up to 0.20 (absolute). So the renormalisation affects both paths; only the cross path also loses symmetry.
  - FOCK is consistent: the Gram and cross paths agree to 1.3e-6, the matrix is symmetric to 2.98e-8, and the diagonal is 1.
  - Element-wise, `K_unbunched(cross) / K_fock` ranges from 1.0 to 2.4 and depends on the first argument. This is consistent with dividing the return probability by the collision-free probability mass of `U†(x₂)U(x₁)|s⟩`, i.e. a postselected conditional probability P(s | no bunching). We have not inspected the implementation, so this mechanism is inferred.
  - For a collision-free input state, a threshold detector's probability of observing the click pattern of |s⟩ is exactly the *unrenormalised* FOCK value. So the UNBUNCHED output matches neither the docstring formula nor a physical threshold-detection kernel without postselection.
- Evidence:
  - Regression tests in `papers/QK_LSTM/tests/test_qk_lstm_merlin.py`:
    - `test_unbunched_space_is_not_a_fidelity_kernel` (cross path) asserts `max|K−Kᵀ| > 1e-3` for UNBUNCHED, `< 1e-5` for FOCK, and `K_unbunched ≥ K_fock` element-wise;
    - `test_unbunched_gram_path_is_symmetric_but_renormalised` (Gram path) asserts symmetry but a deviation above 1e-2 from FOCK.
    Both pass on every run.
  - `papers/QK_LSTM/LOG.md`: "Phase 4 Amendment" and "MerLin API Notes (Phase 4)".
  - Minimal reproduction:

    ```python
    import torch, merlin as ml

    b = ml.CircuitBuilder(n_modes=4)
    b.add_entangling_layer(trainable=False, seed=7)
    b.add_angle_encoding(modes=[0, 1, 2, 3], name="x")
    b.add_entangling_layer(trainable=False, seed=8)
    fm = ml.FeatureMap(builder=b, input_size=4, input_parameters="x")
    z = torch.randn(4, 4)
    for space in (ml.ComputationSpace.UNBUNCHED, ml.ComputationSpace.FOCK):
        K = ml.FidelityKernel(
            fm, input_state=[1, 0, 1, 0], computation_space=space, force_psd=False
        )(z, z)
        print(space.name, float((K - K.T).abs().max()))  # UNBUNCHED ~0.04, FOCK ~3e-8
    ```

- Why it matters:
  - The Gram path gives a symmetric matrix, so the deviation is easy to miss in SVM or kernel-alignment use, yet the values are not the documented fidelity.
  - In the cross-kernel path (test vs train matrices, or kernels used inside models) the result is also asymmetric, so the test kernel is inconsistent with the training Gram matrix.
  - The renormalisation also removes the bunching-probability information, which is physically meaningful, and it changes the kernel's value range.
  - The cookbook and policy both recommend UNBUNCHED as the default for threshold detection, so users are likely to hit this.
- Suggested action:
  - (1) Decide and document the intended semantics in the `FidelityKernel` docstring (renormalised postselection vs return probability).
  - (2) If unintended, compute the unrenormalised return probability in UNBUNCHED space, so it equals FOCK for collision-free |s⟩.
  - (3) If intended, expose a flag (e.g. `postselect=False`) and warn that the result is not symmetric.
  - (4) Add unit tests for `FidelityKernel` in each computation space checking that the Gram path and the cross path agree, that the result is symmetric, and that it matches the FOCK fidelity for collision-free input states.

### Low — FidelityKernel / CircuitBuilder: document cross-kernel use inside models and seeded fixed meshes

- Type: missing documentation
- What happened: two capabilities we relied on work correctly but are not described in the docs or cookbook:
  - (a) `FidelityKernel.forward(x1, x2)` returns a differentiable `(B, N)` cross-kernel with gradients w.r.t. `x1`, so it can sit inside a `torch` model trained by backpropagation. For rectangular cross-kernels `force_psd=False` is required.
  - (b) `CircuitBuilder.add_entangling_layer(trainable=False, seed=k)` produces a fixed mesh determined only by `seed`, independent of the global torch RNG. This is useful for seed-paired comparisons.
- Evidence:
  - `papers/QK_LSTM/tests/test_qk_lstm_merlin.py::test_forward_backward_and_gradient_reaches_compression` (gradient reaches the upstream layer through the kernel);
  - `test_mesh_is_fixed_by_its_seeds_not_the_global_rng` (identical kernels for global seeds 0 and 123, different for mesh seeds 9/10);
  - probe numbers in `papers/QK_LSTM/LOG.md`, "Phase 4 Photonic Assessment" (about 2 ms forward+backward for a 1×4 cross-kernel with 2 photons / 4 modes).
- Why it matters: a photonic kernel used as a learned similarity inside a recurrent or attention model is a natural QML pattern, and the current kernel docs present `FidelityKernel` mainly as a Gram-matrix tool for SVMs.
- Suggested action: add a short "kernel inside a model" example (cross-kernel, gradients, `force_psd=False`), and document that `seed` makes non-trainable layers reproducible independently of `torch.manual_seed`.

### Low — Packaging: merlinquantum 0.3.1 + perceval 1.0.0 fails to import under NumPy 2

- Type: integration blocker (resolved in current versions)
- What happened: in an environment with `merlinquantum 0.3.1`, `perceval 1.0.0` and NumPy 2.x, `import merlin` fails in `perceval/components/processor.py` with `ImportError: cannot import name 'Inf' from 'numpy'` (`numpy.Inf` was removed in NumPy 2.0). With merlin 0.4.1 / perceval 1.3.1 the import works.
- Evidence: `papers/QK_LSTM/LOG.md`, Compute Environment and Dependency Additions (full traceback recorded in the Phase 1 session; environment: framework Python 3.12.2 with user site-packages).
- Why it matters: users with older pinned environments hit an opaque import failure from MerLin's dependency.
- Suggested action: if older merlinquantum releases are still installable, constrain them to `numpy<2` (or to a perceval release compatible with NumPy 2) in their metadata, or mention this in the compatibility page.
