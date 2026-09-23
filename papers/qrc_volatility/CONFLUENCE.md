# Quantum Reservoir Computing for Realized Volatility Forecasting — reproduction

- **Paper:** [arXiv:2505.13933v2](https://arxiv.org/abs/2505.13933) — Qingyu Li, Chiranjib Mukhopadhyay, Abolfazl Bayat, Ali Habibnia (9 Apr 2026, quant-ph)
- **Original code:** <https://github.com/LeeQY1996/Quantum-Reservoir-computing-for-Realized-Volatility-Forecasting> (commit `d2e9b0a`; Julia + Jupyter, README is a stub)
- **Internal repo / branch:** `reproduced_papers`, `papers/qrc_volatility`
- **Jira ticket:** not created (to be filed against the PAPER backlog)
- **PR reproduced_papers:** not opened yet
- **PR MerLin / Perceval:** none; four MerLin items are collected in `ACTION_REQUIRED_SEND_TO_MERLIN_TEAM.md`

## 1. Executive summary

- **What the paper does:** forecasts monthly S&P 500 realized volatility
  (1950-2017) with a 10-qubit quantum reservoir — a fixed transverse-field Ising
  Hamiltonian with 7 feature-encoding qubits and 3 memory qubits — training only a
  ridge readout on the ten Pauli-Z expectations.
- **Why it matters:** it is one of the few QML papers applying a NISQ-scale model
  to a real, well-studied econometric benchmark with the field's standard
  statistical machinery (Model Confidence Set, Diebold-Mariano), and its
  architecture (encode / evolve / partial-trace) is the canonical quantum
  reservoir. It is therefore a good test of whether QRC advantages survive fair
  benchmarking, and a good candidate for photonic translation.
- **Main claims:** QR1/QR2 outperform nine classical benchmarks on MSE and QLIKE
  at one-step-ahead horizon (Table II); QR2 is the single best model; the advantage
  persists at a 5-step closed-loop horizon; `n1 = 7` input features is optimal.
- **Bottom line:** `partially reproduced` — every reported number reproduces, the
  central comparative claim does not.
- **Main takeaways:**
  1. The quantum numbers reproduce essentially exactly (QR1 0.1051 vs 0.105;
     QR2 0.1038 vs 0.103; QLIKE to four decimals), and our reimplementation matches
     the authors' own saved forecasts to float32 precision. Implementation
     confidence is HIGH.
  2. The paper's HAR and HARX baselines are handicapped by a one-month indexing
     error in the released notebook. Correcting it moves HARX from 0.1448 to
     **0.1016** MSE — past both quantum reservoirs.
  3. The reported quantum results are the best of 100 reservoir draws and sit at
     roughly the **5th percentile** of their own distributions; a typical draw
     (0.1095) is worse than the paper's own best classical baseline. Giving a
     classical iso-readout echo state network the identical best-of-100 budget
     gives **0.0974**.
  4. The feature-selection claims *do* reproduce exactly, including the published
     optimal set and its ordering, under both scoring splits.
  5. A MerLin photonic reservoir (10 modes, 3 photons, frozen Haar mesh) matches
     the qubit reservoir (0.1004 vs 0.1018 best-of-100) — so nothing about the task
     requires qubits — but it does not beat the classical controls either.

## 2. Paper overview

- **Core idea:** use fixed quantum dynamics as an untrained nonlinear feature map
  over a lagged window of macro-financial features, and train only a linear
  readout. Memory comes from a hidden qubit register that survives the partial
  trace over the re-encoded input qubits.
- **Similar works in the literature:** Fujii & Nakajima (PRApplied 2017)
  introduced disordered-ensemble QRC; extensions cover photonic platforms
  (Garcia-Beni et al. 2023, Nerenberg et al. 2025), Rydberg arrays (Bravo et al.
  2022) and superconducting hardware (Kubota et al. 2023). Xiong et al. (2025) is
  the exponential-concentration critique the paper answers in Appendix C.
- **Coverage at Quandela:** related but distinct. `papers/QORC` reproduces
  boson-sampling optical reservoir computing (MNIST classification);
  `papers/qrc_memristor` reproduces a photonic quantum-memristor reservoir
  (NARMA10). Neither shares this paper's dataset, task or architecture. This is a
  **novel** reproduction; the MerLin papers database should be updated and a PAPER
  backlog ticket filed.
- **Method summary:** `H = sum_{i<j} J_ij X_i X_j + v sum_i Z_i` on 10 qubits with
  `v = 1` and `J` fixed; features encoded with `RY(pi x)` on 7 input qubits at each
  of 3 lags; evolution `exp(-i H tau)` with `tau = 1`; input qubits traced out
  between lags; Pauli-Z readout; ridge readout with `delta = 1e-8`, re-estimated at
  every one of 245 rolling monthly origins. QR2 additionally reads out at `tau/2`.
- **Main figure / pipeline:** paper Fig. 2 (reservoir schematic), Fig. 3 (QR2
  ensemble), Table II (headline comparison), Fig. 6 (feature-count curve).
- **Key takeaways from the paper:** a 10-qubit reservoir plus a linear readout is
  competitive with HAR-family econometrics and LSTMs on real volatility data; the
  authors explicitly decline to claim a proven quantum advantage.

## 3. Reproduction scope

- **Targeted:** QR1 and QR2 at paper settings (`S = 1`); the HAR, HARX, AR1, AR3,
  ARMAX, RC and RCX baselines; MCS and Diebold-Mariano tests; the `S = 5`
  closed-loop horizon; the full 100-instance reservoir distribution; corrected and
  selection-matched fair baselines; wrapper forward feature selection (QR1); a
  MerLin photonic counterpart.
- **Not targeted:** LSTM and LSTMX (cost; both far from the decision boundary);
  the Shapley interpretability analysis (Fig. 8); QR2's forward-selection path;
  finite-shot or hardware-noise studies; QPU execution.
- **Success criteria:** (i) reproduce the published QR1/QR2 metrics within their
  printed precision; (ii) reproduce the authors' own saved forecasts; (iii) decide
  the comparative claim against baselines that are correctly implemented and
  matched on the claimed advantage axis; (iv) assign a photonic status with
  evidence.

## 4. Original method

| Item | Paper | Reimplementation | Notes |
| --- | --- | --- | --- |
| Architecture | 10-qubit fully connected transverse-field Ising reservoir, `n1 = 7` input / `n2 = 3` hidden qubits, `RY(pi x)` encoding, partial trace between lags, Pauli-Z readout | Identical, exact density-matrix simulation in NumPy (`lib/qrc.py`) | Exploits the fact that the joint state's rank never exceeds `2 ** n2 = 8`, so at most 8 state vectors are propagated instead of a 1024x1024 matrix. Exact, and ~5 s for a full 816-month pass |
| Training setup | Ridge readout only, no intercept, re-estimated at each of 245 rolling origins with a 571-month window | Identical | Verified against the authors' saved forecasts |
| Hyperparameters | `v = 1`, `tau = 1`, `k = 3`, `delta = 1e-8`, 100 reservoir instances, best reported | Identical; the authors' saved `coeff_10.jld2` matrices are used, and all 100 are evaluated rather than only the best | |
| Missing details / assumptions | Feature scaling stated as `[-pi, pi]` but the released code uses `pi * x` on min-max normalised columns; `J` stated as `U[0,1]` but the code rescales to unit spectral radius; QLIKE formula in the text differs from the code; MCS bootstrap seed unset; `S = 5` chain origin unstated; the selection split for "best-performing reservoir" unstated | Reference-code behaviour adopted wherever it disagrees with the text, each disagreement recorded in a source-evidence table in `LOG.md` | The QLIKE and `J` discrepancies are documentation gaps; the selection-split and `S = 5` gaps are scientifically material |

## 5. Reproduction implementation

### 5.1. Quantum implementation

- **Repo / scripts:** `papers/qrc_volatility/lib/{data,qrc,baselines,metrics,runner}.py`;
  configs in `configs/`; tables regenerated by `utils/make_tables.py`.
- **How to run:**
  `python implementation.py --paper qrc_volatility --config configs/table2_no_lstm.json`
  (Table II/III, ~6 min), then `configs/instance_sweep.json` (~45 min),
  `configs/photonic.json` (~18 min), `configs/feature_selection_qr1.json` (~30 min).
- **Compute used:** CPU only, 8 cores, 3 GB RAM; ~2.5 h wall clock total. No GPU,
  no QPU, no paid compute. Estimated API cost of the whole reproduction ~USD 50.
- **Deviations from paper:** nine, all recorded as D1-D9 in `LOG.md`. The material
  ones: the neural and classical-reservoir baselines see the published *normalised*
  feature scale because the authors' raw panel is missing upstream (D3); RC/RCX are
  a NumPy echo state network rather than `reservoirpy` (D4); LSTM/LSTMX and the
  Shapley analysis are not run (D6, D7).

### 5.2. Classical comparison

- **Present in the paper:** `yes` — but `not conclusive enough`. The paper's nine
  baselines are not matched to the quantum model on the axis its advantage is
  claimed on, and two of them are miscomputed.
- **Description of baselines.** Reproduced as published: HAR, HARX, AR1, AR3,
  ARMAX, RC, RCX. Added by this reproduction:
  1. **HAR-aligned / HARX-aligned** — the same models with the released code's
     one-month regressor misalignment corrected (no change to the window, no
     look-ahead introduced).
  2. **Linear-lag** — the same rolling ridge readout applied to the 21 *raw*
     lagged features the reservoir is given, isolating the contribution of the
     quantum feature map.
  3. **ESN-iso-10 / ESN-iso-20** — classical leaky echo state networks with
     exactly the quantum readout dimension, the same seven features, the same
     three-step window and the same ridge, reported under both the paper's
     best-of-100-on-test protocol and leakage-free validation selection.

## 6. Reproduction results

- **Result status:** metric agreement `reproduced` at `S = 1`; trend agreement
  `not reproduced`; claim support `unsupported`; `S = 5` `unresolved`;
  feature-selection claims `reproduced`.

**Table II, `S = 1`, 245 forecasts (MSE on `log RV`):**

| Model | Reproduced | Paper | Note |
| --- | ---: | ---: | --- |
| ESN-iso-20, best of 100 on test | **0.0974** | – | new, matched control |
| ESN-iso-10, best of 100 on test | 0.1009 | – | new, matched control |
| HARX, corrected indexing | 0.1016 | – | new, defect fixed |
| ESN-iso-20, validation-selected | 0.1025 | – | new, leakage-free |
| Linear-lag ridge, no reservoir | 0.1031 | – | new |
| RCX | 0.1034 | 0.1089 | |
| **QR2** | 0.1038 | 0.1030 | reproduced |
| **QR1** | 0.1051 | 0.1050 | reproduced |
| ARMAX | 0.1073 | 0.1145 | |
| HAR, corrected indexing | 0.1157 | – | new, defect fixed |
| AR3 | 0.1179 | 0.1178 | reproduced |
| HAR, as published | 0.1477 | 0.1476 | reproduced |

**Reservoir-instance distribution (100 published coupling matrices):**

| Variant | Mean | SD | Median | Min | 5th pct | Paper's value | Validation-selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QR1 | 0.1095 | 0.0026 | 0.1094 | 0.1032 | 0.1051 | 0.1050 | 0.1063 |
| QR2 | 0.1098 | 0.0036 | 0.1095 | 0.1018 | 0.1037 | 0.1030 | 0.1073 |

- **Figures reproduced:** Table II (`S = 1` fully, `S = 5` qualitatively only),
  Table III (Diebold-Mariano matrix), Fig. 6(a) (feature-count curve and optimal
  set, exactly). Fig. 8 (Shapley) and Fig. 9 (concentration) not attempted.
- **Explanation of differences.** The quantum and autoregressive rows match to
  three or four decimals. HARX, ARMAX, RC and RCX differ by 3-18 % because the
  authors' raw feature panel is missing upstream and because RC/RCX use a
  different echo-state implementation; none of these differences affects the
  verdict. `S = 5` differs materially for every model whose forecast depends on
  feedback (QR1 0.1181 vs 0.1556, ARMAX 0.1218 vs 0.2134) while the purely
  autoregressive models match — a pattern that points at an unstated difference in
  the closed-loop protocol rather than an implementation error.
- **Comparison to baseline.** Six classical models beat both quantum reservoirs
  once the HAR indexing is corrected and the selection budget is matched. The MCS
  does not separate quantum from classical: QR1 and QR2 are in the 95 % set, but
  so are HARX-aligned, RCX, ARMAX and both ESN controls, and ESN-iso-20 holds
  `p = 1.0000`.

## 7. Photonic translation

- **Photonic objective:** determine whether the paper's reservoir role — a fixed,
  untrained nonlinear map of a lagged window feeding a small trained linear
  readout — survives a realistic linear-optical implementation, and whether the
  claimed accuracy behaviour carries over.
- **Proposed photonic formulation:** 10 optical modes and 3 photons. A frozen
  Haar-random MZI mesh replaces `exp(-i H tau)`; the three lags are encoded in
  sequence on the same register (a data-reuploading structure), 7 modes carry
  features and 3 modes are never re-encoded and act as the memory register.
  Nonlinearity comes from multi-photon interference, which is why at least two
  photons are mandatory — a single-photon circuit would be a trivial linear map.
- **Encoding:** phase shifters on modes 0-6, angle `= scale * x`. Because a phase
  shifter is `2 pi`-periodic while `RY(pi x)` is `4 pi`-periodic in its argument,
  the gate-model scale is not transferable; `scale` is swept over `pi/{1,2,4,8}`
  and selected on validation data.
- **Circuit / model:** `mesh -> encode(lag 1) -> mesh -> encode(lag 2) -> mesh ->
  encode(lag 3) -> mesh`, then a per-mode photon-number readout. PQR1 reads one
  reservoir (10 features); PQR2 mirrors QR2 by adding a second reservoir that
  shares every mesh except the last (20 features). Only the ridge readout is
  trained; 360 (PQR1) or 720 (PQR2) circuit phases stay frozen.

### 7.1. MerLin feasibility

- **Can this be done in MerLin?** `partially`. Status:
  **`PARTIAL_MERLIN_TRANSLATION`**. Everything above is built with high-level
  MerLin primitives (`CircuitBuilder.add_entangling_layer`, `add_angle_encoding`,
  `QuantumLayer`, `MeasurementStrategy.mode_expectations`).
- **What could not be done, and why.** The paper's memory mechanism is a *partial
  trace*: discard the input register, keep the reduced mixed hidden state, inject
  fresh photons, repeat. `merlin.QuantumLayer` accepts only a pure Fock
  `input_state`, and `MeasurementStrategy.partial` exposes the branch
  decomposition but offers no way to re-enter it as an input, so a mixed hidden
  state cannot be carried across timesteps. Implementing it would require a custom
  Fock-space density-matrix simulator on low-level Perceval — a second simulator
  rather than a MerLin adaptation. Consequence: the photonic register retains
  information the qubit reservoir throws away, so this translation is an *upper
  bound* on the qubit architecture's memory. Two further MerLin obstacles were
  worked around rather than solved: `add_entangling_layer(trainable=False)` yields
  an identity mesh rather than a frozen random one, and layer names differing only
  by a trailing integer are silently merged into one parameter tensor. All four
  items are written up in `ACTION_REQUIRED_SEND_TO_MERLIN_TEAM.md`.
- **Fallback used:** `None` — the defensible scope was implemented entirely in
  MerLin and the untranslatable part was documented instead of being faked.

### 7.2. Photonic implementation and results

- **What was implemented:** `lib/photonic.py` (`PhotonicReservoir`,
  `photonic_closed_loop_forecast`, `run_photonic`), driven by `configs/photonic.json`.
- **Backend:** MerLin 0.4.0 CPU simulator, analytic statevector, `shots = 0`,
  UNBUNCHED computation space (threshold detectors), no postselection.
- **Modes / photons / layers:** 10 modes, 3 photons, input state
  `[1,0,0,1,0,0,1,0,0,0]`, 4 frozen meshes interleaved with 3 encoding blocks;
  readout width 10 (PQR1) or 20 (PQR2). ~0.09 s (PQR1) / ~0.18 s (PQR2) per
  816-month pass on an idle 8-core container.
- **Training settings:** none for the circuit (frozen); rolling ridge readout,
  `delta = 1e-8`, identical to the qubit path. 25 mesh seeds x 4 encoding scales
  per variant; `(scale, instance)` selected jointly on the validation window.

| Metric / Figure | Original (paper) | Classical (best fair control) | Photonic | Comment |
| --- | ---: | ---: | ---: | --- |
| Table II `S=1` MSE, best-of-N protocol | 0.1030 (QR2) | 0.0974 (ESN-iso-20) | **0.1004** (PQR2) | photonic matches the qubit reservoir; classical control still wins |
| Table II `S=1` MSE, leakage-free selection | 0.1073 (QR2, ours) | 0.1025 (ESN-iso-20) | 0.1067 (PQR1) / 0.1209 (PQR2) | PQR1 indistinguishable from QR1 (0.1063) |
| Table II `S=1` QLIKE, leakage-free | 1.4823 (QR2, ours) | 1.4102 (ESN-iso-20) | 1.4830 (PQR1) | same ordering as MSE |
| Instance-to-instance SD of MSE | 0.0026-0.0036 | – | 0.0283-0.0425 | photonic is ~10x more draw-sensitive |
| `S = 5` MSE at the selected candidate | 0.1663 (QR2) | 0.1237 (ESN-iso-10) | 0.1598 (PQR1) / 0.1695 (PQR2) | photonic tracks the qubit model |
| Encoding-scale trend (mean MSE, `pi` -> `pi/8`) | n/a | n/a | 0.2070 -> 0.1482 (PQR1); 0.1765 -> 0.1136 (PQR2) | monotone improvement toward the linear limit |

- **Photonic assessment.** The translation is scientifically meaningful and the
  answer is informative in both directions. Positively: a 10-mode, 3-photon
  linear-optical reservoir is as good a feature map for this task as the 10-qubit
  Ising reservoir, so nothing about the problem requires qubits, and the
  architecture ports cleanly to photonics with high-level MerLin primitives.
  Negatively: the photonic reservoir inherits the paper's weakness in amplified
  form — its across-draw spread is an order of magnitude larger, so best-of-N
  selection flatters it more, and its accuracy improves monotonically as the
  feature map is driven toward its *linear* limit, which is direct evidence that
  the quantum nonlinearity is not what helps. Neither photonic variant beats the
  corrected or selection-matched classical controls.

## 8. Conclusions

- **What has been done.** A complete, exact Python reimplementation of the paper's
  quantum reservoir, pinned against the authors' own published forecasts to
  float32 precision; reproduction of Table II (`S = 1`), Table III and Fig. 6(a);
  a 200-run reservoir-instance sweep; corrected and selection-matched classical
  baselines; a `S = 5` closed-loop study; and a 200-run MerLin photonic
  adaptation with a full hardware-aware report.
- **What we conclude.** The paper is *reproducible* but its central comparative
  claim is *unsupported*. Two defects fully account for the reported advantage:
  a one-month regressor misalignment that inflates the HAR-family losses, and
  best-of-100 reservoir selection whose classical counterpart the paper never
  runs. With both addressed, six classical models — including plain OLS — beat
  both quantum reservoirs, and the Model Confidence Set does not separate quantum
  from classical. Separately, the paper's feature-selection analysis reproduces
  exactly and is robust to the selection split, and the architecture translates
  usefully to photonics. Implementation confidence is HIGH, so this is evidence
  *against* the claim rather than an inconclusive result.
- **Recommendation:** `do not pursue` as a quantum-advantage result;
  **`pursue with modifications`** as reusable infrastructure. The rank-truncated
  reservoir simulator, the iso-readout-dimension control, and the
  best-of-N-matching protocol are directly reusable by other reservoir-computing
  reproductions, and the photonic reservoir module is a clean MerLin pattern for
  fixed-mesh feature maps.

## 9. Next steps

- **What we could do next.**
  1. Run the two LSTM baselines and the fresh-coupling-draw sweep
     (`configs/instance_sweep_fresh.json`) to close the last two gaps in Table II
     and confirm the instance distribution is not specific to the published draws.
  2. Add a finite-shot study: all results here are analytic, so the paper's
     Appendix C concentration argument is untested. With a 10-feature readout the
     shot cost is the main obstacle to a hardware claim, and this is the natural
     axis on which a photonic implementation could genuinely differ.
  3. Test the architecture on a task that is *not* close to linear (the
     `Linear-lag` control beats the reservoir here), for example a chaotic or
     regime-switching series, where a fixed nonlinear feature map has room to help.
  4. Implement the true partial-trace photonic memory once MerLin can accept a
     mixed input state, and compare it against the reuploading variant. That is the
     one architectural question this reproduction could not answer.
- **What we could not do next.** A definitive `S = 5` comparison, without the
  authors' exact closed-loop indexing; and exact MCS p-value reproduction, because
  the bootstrap seed is unset upstream.
- **Blockers.** MerLin cannot re-inject a mixed photonic state (item 3 in
  `ACTION_REQUIRED_SEND_TO_MERLIN_TEAM.md`); the authors' `Data_raw.csv` /
  `dff.csv` are absent from their repository; no QPU access.

## 10. Deliverables checklist

- [x] Original method reproduced
- [ ] Results reported on Confluence (this file is ready to publish)
- [x] Photonic version defined
- [x] Implemented in MerLin or Perceval (MerLin, `lib/photonic.py`)
- [x] MerLin limitation documented if needed (`ACTION_REQUIRED_SEND_TO_MERLIN_TEAM.md`)
- [x] Photonic version run (200-candidate sweep, hardware-aware report)
- [x] Figure reproduced / adapted (Table II, Table III, Fig. 6a)
- [ ] PR to [reproduced_papers](https://github.com/merlinquantum/reproduced_papers) prepared
- [ ] PR to [MerLin](https://github.com/merlinquantum/merlin) prepared
- [x] Final recommendation written (parts 8 and 9)
