# INSIGHTS — QK-LSTM (arXiv:2412.08851)

Durable lessons for future reproductions of kernel-gated recurrent models, quantum time-series forecasting and photonic kernels.

## Scientific results worth remembering

- **Kernel-gated LSTMs reduce to a bottleneck plus a fixed similarity function.** In QK-LSTM the trainable part is Linear([h; x] → 4) plus gate/readout weights. Swapping the quantum fidelity kernel for a classical RBF kernel or a 2-photon photonic kernel changes test RMSE by ≤ 0.33 AQI, against a sample std across seeds of about 4. Treat claims that "the quantum kernel captures nonlinear dependencies" as untested unless the same architecture is run with a classical kernel.
- **"Fewer parameters" is not an advantage until accuracy is compared at iso-parameters.** A 196-parameter `nn.LSTM(11,3)` beat the 209-parameter QK-LSTM. Always add an iso-parameter classical baseline for compactness claims.
- **Persistence is the baseline to beat in daily air-quality forecasting.** AQI(t+1) = AQI(t) gave RMSE 11.4 / R² 0.66 and was not beaten with statistical support by any learned model, and the gap widened under causal preprocessing. A lag-1-looking prediction trace (like the paper's Fig. 3d) is a warning sign.
- **Window dependence.** Rankings and absolute metrics changed between test windows: persistence was best on the 80/20 test window but worse than every trained model on the validation window. The train period has much higher AQI variance (std 42.6 vs 19.6 on test), and every learned model over-predicted the lower-AQI test period (+5 to +8 AQI bias). Report at least two windows.
- **Single-run papers cannot rank models here.** The per-model sample std across seeds was 2–4 RMSE. Use ≥ 5 seeds and paired, same-seed comparisons: they gave CIs of ±0.3–0.7 for kernel swaps.
- **Five development seeds are not a replication.** On fresh seeds 5–9, two of four "QK-LSTM significantly worse" comparisons became "not separated", although the mean direction held. Pooling seeds 0–9 restored significance. Re-run decisive comparisons on unseen seeds before stating significance.
- **Two-sided interpolation is look-ahead, even when the test set is clean.** The paper's Eq. 3 interpolation filled training gaps from later days: 712 of 1,604 training windows were affected, but no test input and nothing across the train–test boundary. Switching to causal forward-fill made the LSTM +2.6 RMSE worse and left the kernel models unchanged, so the LSTM's significant lead over QK-LSTM disappeared. Preprocessing can shift a model ranking without any test leakage.

## Implementation pitfalls and debugging heuristics

- **Recover undocumented architectures from parameter counts.**
  - 1,873 = `torch.nn.LSTM(11,16)` + `Linear(16,1)`, because PyTorch keeps *two* bias vectors per gate. A single-bias cell gives 1,809.
  - 209 = Linear(27→4) + Linear(4→16) + Linear(16→1).
  - Cross-check any decomposition on a second paper by the same authors: the same template reproduced the parent paper's 183, and only with a vocabulary of 9, not the 5 printed in its table.
- **Check a paper's metric table for internal consistency.** On one test set, var(y) = RMSE²/(1−R²) must agree across rows. Here the QK-LSTM row implied a test std of 23 and the LSTM row 34; allowing ±0.005 rounding, the variance ranges still do not overlap. So the two rows cannot be single, pooled evaluations on the same targets under the stated R² definition. This is cheap to run during Phase 1. Caveats:
  - The identity holds only for standard R² = 1 − SSE/SST. It fails for squared correlation and for explained-variance scores, which ignore bias, and for metrics averaged separately over runs or folds.
  - A common affine rescaling cannot break it.
  - Squared correlation can look much better than R² for biased forecasters: QK-LSTM 0.67 vs 0.33 here (`results/r2_variant_check.md`).
- **A data-independent entangler cancels in a fidelity kernel.** With U = U_ent · U_enc(x) · H, the overlap ⟨φ(x_j)|φ(x_t)⟩ cancels U_ent, so the paper's "H → R_y R_z → CNOT chain" kernel is a product of single-qubit fidelities. Entanglement only matters when data-dependent gates follow the entangler (Fig. 2(d) layout). Unit-test the factorisation.
- **Known-answer tests must bypass asymmetric preprocessing.** A target equal to an input feature is *not* an identity when inputs are outlier-filtered or scaled differently from the target. Disable preprocessing for known-answer checks.
- **Forecast horizon matters.** Same-day (nowcasting) evaluation brought classical models to the paper's reported RMSE level, but not QK-LSTM. Test the horizon whenever a forecasting paper does not state it.
- **The shared runtime's `find_repo_root()` stops at a paper-local `implementation.py`,** which mis-resolves `data_root`. Do not keep one in the paper folder.

## Optimizer and trainability behaviour

- At the paper's lr 1e-3 × 20 epochs × batch 1, LSTMs are under-trained: validation RMSE was 17.9 at lr 1e-3 vs 15.6 at lr 1e-2. Kernel models were best at 1e-3, and higher lr hurt them. An equal-budget lr grid on validation is therefore needed for fairness: the paper's fixed setting favoured the quantum model.
- More references (N = 16 instead of 4) made the Eq. 6 kernel LSTM *worse* on validation.
- Autograd through an exact statevector or SLOS simulation equals the parameter-shift gradient (with the 1/2 factor that Eq. 15 of the paper omits). Use autograd for simulation and document only the estimator difference.

## Photonic translation and MerLin patterns

- **Photonic kernel inside a model.** `ml.FeatureMap(builder=..., input_size=d, input_parameters="x")` + `ml.FidelityKernel(fm, input_state=..., computation_space=FOCK, force_psd=False)`. `kernel(z, z_ref)` gives a differentiable (B × N) cross-kernel. About 14 ms per BPTT training step for 2 photons / 4 modes on CPU.
- **Fixed, reproducible meshes.** `CircuitBuilder.add_entangling_layer(trainable=False, seed=k)` depends only on `seed`, not the global torch RNG. Wrap kernel construction in `torch.random.fork_rng()` so seed-paired initialisation matches the qubit and RBF models exactly.
- **`UNBUNCHED` is not a fidelity kernel in merlin 0.4.1.** `FidelityKernel(computation_space=UNBUNCHED)` renormalises by the collision-free probability mass (up to 0.20 absolute / 2.4× deviation), and its cross-kernel path is asymmetric. For a collision-free input state, the `FOCK` return probability equals what threshold detectors measure without postselection, so use FOCK and record the detector model as threshold-equivalent.
- **Multi-photon interference did not help.** 2-photon vs 1-photon kernels: ΔRMSE +0.20 [−0.75, +1.15]. A non-factorising photonic kernel is not automatically more useful than a product or RBF kernel.
- **Mapping choice.** Map the *role* (a fidelity similarity at a fixed parameter budget) rather than the qubit circuit. A dual-rail emulation of a factorising qubit kernel adds cost (probabilistic CNOTs) without scientific value.

## Baseline fairness concerns

- The paper's single untuned LSTM baseline cannot support either accuracy or compactness claims. Plan these before seeing results: persistence, ridge, an iso-parameter LSTM, a same-architecture classical-kernel ablation, and equal tuning.
- Persistence reads past AQI, which the learned models did not take as input. Label it partially commensurable: a floor, not a matched control.
- The RBF bandwidth γ was fixed at 1, untuned. Tuning it could only strengthen the classical side; note it as a caveat on kernel-attribution statements.

## Scaling limitations

- Qubit statevector (16 amplitudes) and SLOS (2 photons, 4 modes) are both trivial here. Runtime is dominated by Python per-step overhead with batch 1: about 5–10 min per kernel-model run vs under 1 min per LSTM run on CPU. This is an implementation cost, not evidence about algorithmic speed.

## Promising directions (if revisited)

- Tune the RBF γ, and try a learned-bandwidth classical kernel, to settle whether the small qubit-over-RBF edge on one window (−0.33 RMSE) is a bandwidth effect.
- Test kernel-gated recurrent models on longer sequences or tasks where persistence is weak; this dataset gives little headroom beyond lag-1 information.
- Photonic kernels with trainable meshes or data re-uploading would add circuit parameters, so they would need a new iso-parameter baseline.
