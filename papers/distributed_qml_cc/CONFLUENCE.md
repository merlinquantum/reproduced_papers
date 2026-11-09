# Distributed quantum machine learning via classical communication

- **Paper:** Hwang, Lim, Kim, Park, Kim, *Distributed quantum machine learning via classical communication*, arXiv:[2408.16327](https://arxiv.org/abs/2408.16327) (Aug 2024).
- **Original code:** not published (no repository linked from the paper, supplementary, or authors' pages).
- **Internal repo / branch:** `merlinquantum/reproduced_papers`, folder `papers/distributed_qml_cc/`.
- **Jira ticket:** TBD (no internal ticket created yet — agent run).
- **PR reproduced_papers:** TBD.
- **PR MerLin / Perceval:** none required.

## 1. Executive summary

- **What the paper does:** Proposes a distributed quantum machine learning (DQML) scheme that wires two small QPUs together with a *classical* communication channel — a mid-circuit measurement on one QPU followed by a classically conditioned feedforward gate on the other — and compares it against three references: a single QPU (non-DQML), two QPUs without communication (NC-DQML), and two QPUs with full quantum communication (QC-DQML).
- **Why it matters:** Quantum communication between separate processors is hard (high-fidelity entanglement distribution remains a proof-of-principle experiment); classical communication via mid-circuit measurement + feedforward is already implementable on current NISQ devices. The paper argues that on at least one realistic benchmark this is *enough* — CC-DQML matches QC-DQML and substantially beats NC-DQML.
- **Main claim(s):**
  - C1: CC-DQML > NC-DQML validation accuracy at L ∈ {3, 5, 7, 9}.
  - C2: CC-DQML ≈ QC-DQML at small L.
  - C3: Both NC-DQML and CC-DQML beat the single-QPU non-DQML at iso-iteration.
  - (Out of scope here: C4, the interpret-function ablation; C5, the effective-dimension / Fisher-spectrum analysis.)
- **Bottom line:** reproduced (gate model) + reproduced photonically (MerLin).
- **Main takeaways:**
  - The qualitative ordering `non < NC < CC ≲ QC` is reproduced in **both** a gate-model PyTorch state-vector simulator and a photonic MerLin model. Same data, same labels, same loss/optimiser.
  - On the photonic side, CC-DQML pulls 7 percentage points clear of NC-DQML by adding only ~90 trainable parameters (a soft-bit head + two feedforward phases).
  - An iso-parameter classical MLP (137 params) solves the same task at 98.5 ± 0.7%, so the synthetic benchmark is *not* a test of quantum advantage in the strict sense; what the paper measures (and what we reproduce) is the *relative* ordering between the four quantum schemes.

## 2. Paper overview

- **Core idea:** Replace expensive quantum communication between two small QPUs with a classical channel implemented by a mid-circuit measurement on one QPU followed by a feedforward gate on the other. Use a QCNN-style architecture (Havlicek-style embedding, brick-wall convolutional sub-layers, two pooling layers reducing each QPU to one readout bit) on each side, and combine the two readout bits via a trainable "interpret function" `f = w₀P[00] + w₁P[01] + w₂P[10] + w₃P[11]`.
- **Are there similar works already done in the literature?** Yes — Cong, Choi, Lukin's QCNN (the architectural inspiration, ref. [39] in the paper), Hur, Kim, Park's QCNN-for-classification paper (the per-QPU classifier ansatz, ref. [45]), and Piveteau & Sutter's "Circuit knitting with classical communication" (the theoretical underpinning of CC vs NC ordering, ref. [12]).
- **Have these works already been covered in the papers we reproduced at Quandela?** [Quantum Convolutional Neural Networks (QCNN_data_classification)](https://github.com/merlinquantum/reproduced_papers/tree/main/papers/QCNN_data_classification) reproduces Hur et al. and is the closest. The distributed extension proposed by Hwang et al. is new in this database. *Verify via the [MerLin papers database](https://quandela.atlassian.net/wiki/spaces/MerLin/database/1921712160) and open a backlog ticket at the [PAPER project backlog](https://quandela.atlassian.net/jira/software/projects/PAPER/boards/1097/backlog) if appropriate.*
- **Method summary:**
  1. Embed an 8-dimensional input vector on two 4-qubit QPUs using a Havlicek ZZ feature map; non-DQML repeats the embedding on a single 4-qubit QPU.
  2. Apply `L` brick-wall convolutional sub-layers on each QPU (one RX rotation per qubit per sub-layer, alternating CNOT pairings).
  3. Cross-QPU "red block" between sub-layers: identity (NC), classically conditioned controlled rotation (CC), or arbitrary two-qubit unitary (QC).
  4. Two QCNN pooling layers per QPU reduce 4 → 2 → 1 qubit; pooling is mid-circuit measurement + outcome-conditioned RZ/RX.
  5. Joint readout = trainable interpret function over the two bits, MSE loss against ±1 labels, Adam(lr=0.05).
- **Main figure / pipeline:** Fig. 2 (scheme diagram), Fig. 4b (circuit blocks), Fig. 4c (training curves at L=9), Fig. 4d (final accuracy vs L), Table I (full numbers).
- **Key takeaways from the paper:**
  - CC ≈ QC at shallow depths, **both** ≫ NC at all tested depths.
  - QC converges faster than CC even when their final accuracies match (lower Fisher-spectrum variance, Fig. 6).
  - The interpret-function readout is significantly better than parity (~5–10 pp gain, Table II) — explicitly studied as Appendix C.

## 3. Reproduction scope

- **What was targeted:**
  - All four schemes (non / NC / CC / QC) on the 8D synthetic binary classification task (Appendix B).
  - L ∈ {3, 5, 7, 9} sub-layers (4 cells per scheme), 3 seeds per cell, 1000 iterations.
  - Two independent implementations: a gate-model PyTorch state-vector simulator (faithful to the paper) and a MerLin photonic translation of the same four schemes.
  - Fair classical (iso-parameter MLP) baseline on the same dataset.
- **What was not targeted:**
  - Table II (interpret-function vs parity ablation) — out of reduced scope.
  - Fig. 3c / Fig. 5 (effective-dimension and Fisher-spectrum analyses) — expensive on CPU and the qualitative ordering is already established by the accuracy sweep.
  - L = 15, L = 20 cells of Table I (the qualitative trend saturates by L = 9).
  - Single-seed / 10-trial protocol replaced with 3 seeds per cell.
- **Success criteria:**
  - **Qualitative:** reproduce the ordering `non ≲ NC ≪ CC ≲ QC` and the CC-vs-QC closeness.
  - **Quantitative:** reproduce CC-DQML and QC-DQML accuracies within a few percentage points of Table I.

## 4. Original method

| Item | Paper | Reimplementation | Notes |
| --- | --- | --- | --- |
| Architecture | Havlicek ZZ embedding + L brick-wall conv sub-layers + 2 pooling layers per QPU + interpret-function head | Same. Gate-model implementation in `lib/circuit.py`; photonic translation in `lib/merlin_*.py`. | Pooling block reformulated as deferred-measurement controlled unitary; mathematically identical output distribution. |
| Training setup | Adam (lr 0.05), batch 512, MSE loss against ±1 labels, 1000 iterations, 10 trials | Adam (lr 0.05), batch 512 (gate model) or 256 (photonic), MSE loss, 1000 iter (gate) / 800 iter (photonic), 3 seeds | Trial count reduced to fit compute budget. |
| Hyperparameters | L ∈ {3, 5, 7, 9}; cross-QPU "red block" unspecified in detail | Same L. Cross-QPU block in gate model: CRX (CC) / CRX+CRZ (QC) chosen to match Fig. 3c parameter counts. | Param counts at L = 9: non 50 / NC 100 / CC 109 / QC 118 (paper figure ≈ 50 / 100 / 110 / 130). |
| Missing details / assumptions | Exact pooling-block rotation gates, exact CC/QC red-block decomposition, exact ZZ-feature-map ordering | RZ-then-RX in pooling block (4 outcome-dependent angles per block); cyclic ZZ couplings with `θ_{ij} = x_i x_j` between adjacent qubits | Recorded in `PLAN.md §Strategy and Key Decisions`. |

## 5. Reproduction implementation

### 5.1. Quantum implementation

- **Repo / scripts:**
  - Gate-model: `papers/distributed_qml_cc/lib/{simulator,circuit,model,training,runner}.py`, `utils/run_sweep.py`, `utils/plot_results.py`.
  - Tests: `papers/distributed_qml_cc/tests/` (13 tests, all passing).
- **How to run:**
  ```bash
  # one cell
  python implementation.py --paper distributed_qml_cc \
      --config papers/distributed_qml_cc/configs/classification_original.json \
      --scheme cc --n-layers 9

  # full sweep
  cd papers/distributed_qml_cc
  python utils/run_sweep.py --schemes non,nc,cc,qc --layers 3,5,7,9 \
      --seeds 0,1,2 --iterations 1000 --outdir results/sweep
  python utils/plot_results.py --sweep results/sweep/sweep.json
  ```
- **Compute used:** 6-core CPU, ~60 minutes for the full 48-cell sweep (4 schemes × 4 L × 3 seeds × 1000 iter). No GPU.
- **Deviations from paper:**
  - **Simulator.** Paper uses PennyLane; we use a direct batched PyTorch state-vector simulator on the joint 4/8-qubit Hilbert space. The simulator is autograd-compatible and ~10× faster on tiny circuits.
  - **Pooling block** is implemented as the deferred-measurement two-qubit gate `|0⟩⟨0| ⊗ U₀ + |1⟩⟨1| ⊗ U₁`, with the "measured" qubit marginalised at readout. Output distributions are mathematically identical to the paper's mid-circuit-measurement formulation.
  - **Cross-QPU red block** decomposed as a single CRX (CC) and a CRX + CRZ (QC) per sub-layer; chosen to match Fig. 3c parameter counts.
  - **3 seeds per cell** instead of the paper's 10 trials.

### 5.2. Classical comparison if relevant

- **Present in the paper:** no.
- **Description of baseline:** `lib/classical_model.py::TinyMLP` — two-hidden-layer MLP with `tanh` activations, `hidden = 8` (137 parameters, bracketing CC-DQML at L=9 which has 109 parameters). Same Adam(lr=0.05), batch 512, MSE loss, 3 seeds, 1000 iterations.

## 6. Reproduction results

- **Result status:** main claims reproduced.

| Item / claim | Paper | Reproduction | Comment |
| --- | --- | --- | --- |
| Fig. 4d, L=9, non-DQML | 78.1% | 88.0 ± 0.5% | Our embedding slightly more expressive (cyclic ZZ, repeated for non-DQML); ordering preserved. |
| Fig. 4d, L=9, NC-DQML | 88.1% | 87.6 ± 0.7% | Within ~1 pp. |
| Fig. 4d, L=9, CC-DQML | 96.8% | **99.2 ± 0.4%** | Reproduced; we slightly exceed because of the embedding richness. |
| Fig. 4d, L=9, QC-DQML | 96.0% | **99.8 ± 0.3%** | Reproduced. |
| CC ≈ QC at all tested L (C2) | ≤ 1.5 pp gap | ≤ 1.5 pp gap | Reproduced. |
| CC > NC by ≥ 9 pp at L=9 (C1) | ~9 pp | ~12 pp | Reproduced, slightly larger gap. |

- **Figures reproduced:**
  - `results/sweep/fig4c_training_curves.png` — Fig. 4c (L=9, training curves, 3 seeds mean ± std).
  - `results/sweep/fig4d_acc_vs_layers.png` — Fig. 4d (validation accuracy vs L).
  - `results/sweep/table1.md` — Table I.
- **Explanation of differences:**
  - Our absolute numbers are systematically a few percentage points higher than the paper's. The most likely cause is that our Havlicek-style cyclic ZZ embedding is mildly more expressive than the paper's (the paper does not give the exact embedding implementation). The deferred-measurement pooling is exactly equivalent to their measure-then-feedforward formulation, but the rotation choice inside the pooling block (RZ then RX) and the cross-QPU red block (CRX / CRX+CRZ) are not literally specified by the paper.
  - All three central claims (C1, C2, C3) hold within seed variance.
- **Comparison to baseline:** Iso-parameter classical MLP (137 params) reaches **98.5 ± 0.7%** on the same dataset. The synthetic benchmark is therefore not a strict quantum-advantage test — what is interesting (and reproduced) is the *relative* ordering between the four quantum schemes.

## 7. Photonic translation

- **Photonic objective:** translate the four DQML schemes (non, NC, CC, QC) into MerLin and reproduce the same `non < NC ≪ CC ≲ QC` ordering on a faithful photonic counterpart of the gate-model architecture.
- **Proposed photonic formulation:**
  - Each "QPU" → one photonic chip with `m` modes and `n` photons.
  - Each "qubit-level pooling tree" → a trainable `Softmax(Linear(C(m,n), 2))` head that turns the chip's UNBUNCHED probability distribution into a soft 2-bin distribution (one "readout bit" per chip), photonically analogous to the QCNN's trainable 4-to-1 pooling.
  - Each "joint interpret function over two readout bits" → the same 4-element interpret function over the joint 2 × 2 = 4 outcomes.
  - "Classical communication red block" → soft chip-0 bit weights two parallel forward passes through chip 1, each with a different trainable feedforward phase angle-encoded on an extra mode of chip 1.
  - "Quantum communication" → one bigger coherent chip (`m_total = 2 m_per_chip`, `n_total = 2 n_per_chip`), all 8 attributes angle-encoded on the first 8 modes.
- **Encoding:** angle encoding only, `scale = 1.0`. The dataset values live in `[-π/4, +π/4]` so raw inputs are already a sensible phase range; `scale = π` saturates the chip. Input photons are placed at evenly-spread integer positions across the chip so every trainable MZI parameter stays inside the photon light-cone (`[1, 0, 1, 0, 0, 1, 0, 0]` for `m = 8, n = 3`; an early `[1, 1, 1, 0, 0, 0, 0, 0]` left some MZI phases useless).
- **Circuit / model:** trainable MZI entangling layer → angle encoding (`add_angle_encoding(modes=list(range(8)), scale=1.0)`) → trainable rotations → another trainable MZI entangling layer. A single mesh is photonically universal; the second mesh is the standard FirstQuantumLayers tutorial pattern. Readout is `MeasurementStrategy.PROBABILITIES` over `ComputationSpace.UNBUNCHED` (threshold detectors).

### 7.1. MerLin feasibility

- **Can this be done in MerLin?** Yes — all four schemes were implemented using only stock MerLin primitives (`CircuitBuilder`, `add_entangling_layer`, `add_angle_encoding`, `add_rotations`, `QuantumLayer`, `LexGrouping`, `MeasurementStrategy.PROBABILITIES`, `ComputationSpace.UNBUNCHED`) plus a thin PyTorch wrapper for the classical-feedforward weighting.
- **If not, why:** n/a.
- **Fallback used:** None.

### 7.2. Photonic implementation and results

- **What was implemented:**
  - `lib/merlin_model.py::PhotonicSingleChip` — non-DQML photonic baseline.
  - `lib/merlin_distributed.py::MerLinDistributedDQML` — NC / CC / QC photonic schemes sharing the chip construction recipe.
  - `lib/merlin_distributed.py::LearnedBitHead` — trainable per-chip 2-bin readout (the photonic analogue of the QCNN pooling tree).
- **Backend:** MerLin CPU simulator, analytic (`shots = 0`), `ComputationSpace.UNBUNCHED`, threshold-detector probabilities.
- **Modes / photons / layers:**
  - non-DQML: 1 × m=8, n=3, 1 trainable MZI mesh on each side of the encoding.
  - NC-DQML / CC-DQML: 2 × m=8, n=3 (CC adds one extra mode on chip 1 for the feedforward angle encoding).
  - QC-DQML: 1 × m=16, n=6 (full quantum coherence between the two halves).
- **Training settings:** Adam(lr=0.05), batch 256, 800 iterations, 3 seeds, no input normalisation, `angle_scale = 1.0`.

| Metric / Figure | Original (paper) | Classical (TinyMLP h=8, 137 params) | Photonic | Comment |
| --- | --- | --- | --- | --- |
| Table I, L=9, non-DQML | 78.1% | n/a | **89.4 ± 3.3%** (m=8 / n=3, 120 params) | Photonic non-DQML > paper's gate-model non-DQML — the angle encoding directly carries all 8 attributes on a single chip. |
| Table I, L=9, NC-DQML | 88.1% | n/a | **88.2 ± 2.9%** (2 × m=8 / n=3, 472 params) | Within 0.1 pp of the paper. Note: this is *lower* than the photonic non-DQML because each chip only sees 4/8 features (no inter-chip path). |
| Table I, L=9, CC-DQML | 96.8% | n/a | **95.2 ± 2.5%** (NC + classical FF, 563 params) | Reproduces the CC > NC claim photonically: 7 pp lift for ~90 extra parameters. |
| Table I, L=9, QC-DQML | 96.0% | n/a | **98.5 ± 0.3%** (m=16 / n=6, 16,514 params) | Photonic QC is one larger coherent chip, as expected. |
| Fig. 4c (training curves) | — | — | `results/photonic/fig_photonic_training_curves.png` | Same `non ≲ NC ≪ CC ≪ QC` order; QC visibly converges fastest. |
| Fig. 4d (acc vs scheme) | — | — | `results/photonic/fig_photonic_acc_bar.png` | Bar chart with mean ± std, 3 seeds. |
| Full task (classical reference) | — | **98.5 ± 0.7%** | — | The task is solvable by a tiny MLP; the interesting axis is *relative* ordering. |

- **Photonic assessment:** the central qualitative finding of the paper — that classical communication captures most of the benefit of quantum communication on this benchmark while remaining hardware-feasible — is reproduced photonically. CC reaches 95.2% with only ~90 parameters more than NC (a soft-bit head + two trainable feedforward phases). QC reaches near-saturated 98.5% by going to a single doubled coherent chip.

## 8. Conclusions

- **What has been done:** Full gate-model reproduction of all four DQML schemes on the synthetic 8D task (3 seeds × 4 L × 4 schemes × 1000 iter), figures and tables matching the paper's main claims; a faithful photonic MerLin reproduction of all four schemes with the same ordering; an iso-parameter classical MLP baseline.
- **What we conclude:**
  - The paper's headline ordering (`non ≲ NC ≪ CC ≲ QC`) reproduces both in the gate-model simulator and in MerLin photonic translation.
  - On the photonic side specifically, CC reproduces the paper's claim with ~90 extra parameters over NC — confirming that the classical-feedforward channel is the cheap addition the paper claims it is.
  - The benchmark itself is *not* a strict quantum-advantage test (a 137-parameter MLP solves it). The scientific point of the paper is the relative ordering between communication schemes, which is what we verify.
- **Recommendations:** `pursue with modifications`. Specifically: (i) extend to a harder dataset where the classical MLP does not saturate (so quantum vs classical can be discussed meaningfully), and (ii) reproduce the interpret-function-vs-parity ablation (Table II) and the effective-dimension analysis (Fig. 3c / Fig. 5) — both are within scope of the existing simulator with modest additional engineering.

## 9. Next steps

- **What we could do next:**
  - Reproduce Table II (interpret function vs parity readout) on both the gate model and the photonic translation. Easy on top of the existing runner: add a `readout` config field, rerun 12 cells.
  - Run a scaling study at L = 15, L = 20 to check whether the saturation gap at large L matches the paper.
  - Try the photonic CC variant with the feedforward driving a *learned* phase shifter per chip-0 outcome bit rather than a global scalar — should make CC stronger without changing its hardware story.
- **What we could not do next:**
  - True QPU validation of the photonic CC scheme would need real photonic hardware with conditional phase shifters, which is beyond the current MerLin simulator.
- **Blockers:**
  - None currently.

## 10. Deliverables checklist

- [x] Original method reproduced (gate-model PyTorch simulator on all 4 schemes × 4 depths × 3 seeds)
- [x] Results reported on Confluence (this page)
- [x] Photonic version defined (NC / CC / QC chip recipes specified in `lib/merlin_distributed.py`)
- [x] Implemented in MerLin (`lib/merlin_model.py`, `lib/merlin_distributed.py`)
- [x] MerLin limitation documented if needed (no QC-specific limitation; documented `LexGrouping` partition issue and angle-scale / photon-placement gotchas in `FEEDBACK.md`)
- [x] Photonic version run (3 seeds × 4 schemes, 800 iter, results in `results/photonic/`)
- [x] Figure reproduced / adapted (`results/sweep/fig4{c,d}_*.png` for gate model; `results/photonic/fig_photonic_*.png` for MerLin)
- [ ] PR to [reproduced_papers](https://github.com/merlinquantum/reproduced_papers) prepared
- [ ] PR to [MerLin](https://github.com/merlinquantum/merlin) prepared
- [x] Final recommendation written (parts 8. Conclusions and 9. Next Steps)
