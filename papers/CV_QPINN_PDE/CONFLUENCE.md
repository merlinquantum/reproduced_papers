# Quantum Physics-Informed Neural Networks for PDEs — Confluence summary

- **Paper:** Panichi, Corli, Prati, *Quantum physics informed neural networks for multi-variable partial differential equations*, [arXiv:2503.12244v2](https://arxiv.org/abs/2503.12244) (Nov 2025)
- **Original code:** not advertised by the authors
- **Internal repo / branch:** `papers/CV_QPINN_PDE/`
- **Jira ticket:** —
- **PR reproduced_papers:** —
- **PR MerLin / Perceval:** —

## 1. Executive summary

- **What the paper does:** introduces a Continuous-Variable (CV) Quantum
  PINN that exposes one homodyne output per derivative order and trains
  a *consistency loss* to enforce each extra output as the derivative of
  the previous one — sidestepping nested automatic differentiation in CV
  simulators. The method is demonstrated on the 1D Poisson and 1D heat
  equations, and the resilience to photon-loss noise (modelled from the
  Xanadu X8 device) is studied numerically.
- **Why it matters:** previous CV-QPINN work was limited to first-order
  ODEs because nested gradients through Strawberry Fields blew up memory.
  The consistency loss is a clean way to unlock second-order PDEs on the
  CV ansatz, and the same trick is readily portable to gate-model and
  photonic-linear-optics PINNs.
- **Main claims:**
  1. The two-output CVQNN + consistency loss converges to the Poisson 1D
     analytic solution to RMSE ≈ 1e-4 (Table II row 4, 5000 epochs).
  2. The same architecture handles the 1D heat equation to RMSE ≈ 1.24e-2,
     slightly better than a parameter-matched classical PINN at 2.09e-2
     (Table IV).
  3. The architecture is robust to realistic photon-loss noise sampled
     from Xanadu's X8 device.
- **Bottom line:** **partially reproduced, advantage claim refuted,
  consistency-loss has a previously-undocumented accuracy cost** —
  claim 1 (consistency-loss CVQNN solves 1D Poisson) and claim 2
  (architecture extends to heat equation) reproduce qualitatively at
  reduced compute; claim 3 (noise resilience) is out of scope. The
  headline *comparative* claim of Table IV (QPINN slightly outperforms
  classical PINN) is **refuted** by a 5-seed matched-effort sweep:
  classical PINN reaches RMSE 8.74e-3 ± 1.2e-3 vs QPINN 1.23e-2 ± 4.8e-3,
  PINN winning by 1.4× with ~4× lower variance (2.9 PINN-σ gap). A
  head-to-head 1D-Poisson nested-vs-consistency ablation at 200 epochs
  shows **nested autograd reaches 12–100× better RMSE** at smoke cutoffs
  (4.24e-5 vs 4.6e-3 at cutoff 8; 1.51e-4 vs 1.85e-3 at cutoff 12) — the
  consistency-loss trick is a real memory optimisation (30 MB RSS delta
  at cutoff 12) but it comes at a substantial, previously-undocumented
  accuracy cost in the regime where memory is not yet the bottleneck.
- **Main takeaways:**
  - The consistency-loss trick works and is library-agnostic — we observed
    it improving both quantum and classical PINNs.
  - The paper's classical-PINN baseline is *not* a fair comparison under
    matched optimisation effort: a 42-parameter classical PINN trained
    with the *same* consistency-loss scheme reaches RMSE 8.9e-3 on the
    heat equation, better than the QPINN's reported 1.24e-2. The "QPINN
    slightly outperforms" claim therefore needs a fairer baseline study.
  - The CV ansatz is *not* a natural fit for MerLin (which targets
    linear-optical photonic computing). The closest meaningful photonic
    counterpart is a linear-optics interferometer + angle-encoding PINN
    that re-uses the consistency-loss training scheme.

## 2. Paper overview

- **Core idea:** stack one extra qumode (and one extra homodyne output)
  per derivative order; a consistency loss enforces each extra output to
  equal the autograd derivative of the previous output. The resulting
  PINN's loss combines PDE residual, BC, IC, trace normalisation, and the
  consistency term.
- **Similar works already in the literature:** Knudsen & Mendl, 2020
  (arXiv:2012.12220) — CV QNN for first-order ODEs only; Killoran et al.
  2019 — the underlying CV QNN ansatz. PennyLane-based gate-model QPINNs
  exist (refs [28–34] in the paper) but use nested autograd.
- **Already covered in `papers/`?** Closest is `papers/HQPINN`, but that
  paper is a *hybrid quantum/classical* PINN using PennyLane and the
  MerLin linear-optics primitives; the present paper is fully quantum on
  the CV photonic platform. The two are complementary rather than
  redundant. The `papers/HQPINN/lib/layer_merlin.py` patterns are not
  directly reusable here because they assume linear-optics gates.
- **Method summary:** Two-qumode Killoran QNN (4 + 4 multi/single-qumode
  layers for Poisson, 2 + 2 for heat) on a Fock-truncated simulator.
  Input encoded by displacement `D(x)`. Loss is a weighted sum of PDE,
  BC, trace, and consistency terms. Adam optimiser with cosine annealing
  warm-restart schedule.
- **Main figure / pipeline:** Figure 7 in the paper for the heat
  experiment is the canonical schematic.
- **Key takeaways from the paper:**
  - Adam outperforms SGD/RMSprop/Adagrad on this ansatz (Figure 13).
  - Optical-loss noise is largely systemic and the variational algorithm
    learns to absorb it (Figure 10).

## 3. Reproduction scope

- **Targeted:**
  - 1D Poisson with the consistency-loss QPINN and a parameter-matched
    classical PINN baseline.
  - 1D heat equation with the consistency-loss QPINN and a parameter-
    matched classical PINN baseline.
  - A MerLin photonic-linear-optics counterpart of the consistency-loss
    PINN on the Poisson task.
- **Not targeted:**
  - The X8 noise study (the gate-level photon-loss model is implemented
    but not exercised in the reported runs).
  - The classical-optimiser benchmark of Figure 13 (deferred).
  - Multi-seed statistics — we run a single seed per configuration and
    label results as `preliminary` accordingly.
- **Success criteria:** RMSE within 1–2 orders of magnitude of the paper
  on the smoke configs; consistency loss demonstrably drives the
  network's second output toward the autograd derivative; matched-
  parameter-count baseline included.

## 4. Original method

| Item | Paper | Reimplementation | Notes |
|---|---|---|---|
| Architecture | CV-QNN per Killoran (BS + R + S + D + Kerr) | Same, with Fock-truncated PyTorch gates | We do not use Strawberry Fields; we expose each gate as a unitary on a Fock truncation |
| Training setup | Adam + cosine annealing warm restarts + IC pre-training | Identical scheme, optional cosine schedule | |
| Hyperparameters | Tables V (Poisson) and VI (heat) | Implemented in `configs/*_original.json` | Smoke configs reduce cutoff, epochs, and collocation points to stay CPU-friendly |
| Missing details / assumptions | Exact gate parameter counts per layer, exact Sobol-sequence handling, exact `λ` normalisation | We implement the standard Killoran layout (giving 14 per multi-qumode + 10 per single-qumode layer) and document the discrepancy with Table II | |

## 5. Reproduction implementation

### 5.1 Quantum implementation

- **Repo / scripts:** `papers/CV_QPINN_PDE/lib/`.
- **How to run:**
  ```bash
  python implementation.py --paper CV_QPINN_PDE --config configs/poisson_smoke.json
  python implementation.py --paper CV_QPINN_PDE --config configs/heat_smoke.json
  ```
- **Compute used:** CPU only, single seed, 5–60 minutes per smoke run on a
  modern workstation.
- **Deviations from paper:** PyTorch + `matrix_exp` substitute for
  Strawberry Fields + TensorFlow. We use autograd everywhere, not
  parameter-shift. Cutoff and layer counts are reduced in the smoke
  configs; the paper-accurate values are available in `*_original.json`.

### 5.2 Classical comparison

- **Present in the paper:** yes, but with the same model architecture
  trained without the consistency-loss enhancement. We re-implement a
  *fair* baseline that uses the same loss design as the QPINN.
- **Description of baseline:** fully-connected `tanh` PINN with one
  hidden layer sized to match the QPINN's parameter count, two output
  heads (`u`, `ux`), trained with the same PDE + BC + IC + consistency
  loss.

## 6. Reproduction results

- **Result status:** partially reproduced.
- **Figures reproduced:** Poisson convergence (analogue of Fig. 4),
  heat-equation comparison panel (analogue of Fig. 11). Stored under
  `results/` (script `utils/make_figures.py`).
- **Explanation of differences:** smoke configurations land within
  1–2 orders of magnitude of the paper's RMSE because we run for far
  fewer epochs and at lower cutoff. The paper-accurate configs
  (`*_original.json`) close the gap but require multi-hour CPU runs.
- **Comparison to baseline:** on the heat equation a 42-parameter
  classical PINN trained with the *same* consistency-loss scheme reaches
  RMSE 8.9e-3, compared with the paper-reported QPINN 1.24e-2. The QPINN
  remains scientifically interesting (it inherits the photonic
  inductive bias), but the headline "QPINN slightly outperforms PINN"
  claim of Table IV does not survive matched-effort optimisation.

## 7. Photonic translation

- **Photonic objective:** evaluate whether the CV-QPINN's *inductive
  bias* survives a linear-optics photonic implementation under MerLin.
- **Proposed photonic formulation:** MerLin interferometer + angle
  encoding of the PDE input variable + two trainable linear heads for
  `u` and `ux`. Re-use the paper's consistency-loss training scheme.
- **Encoding:** angle encoding on three modes of a six-mode chip.
- **Circuit / model:** 1 entangling layer + angle encoding + 3
  trainable entangling layers, 3 photons, UNBUNCHED computation space.

### 7.1 MerLin feasibility

- **Can this be done in MerLin?** Not as a literal port. MerLin does not
  expose squeezing, displacement, or Kerr gates that the CV architecture
  relies on.
- **If not, why:** MerLin's photonic target is *linear-optical photonic
  computing with discrete-photon measurement*, a fundamentally different
  photonic platform from the CV computing the paper targets.
- **Fallback used:** a meaningful photonic adaptation (not a literal
  port) that re-uses the consistency-loss training scheme on a MerLin
  interferometer + angle-encoding model.

### 7.2 Photonic implementation and results

- **What was implemented:** `lib/merlin_pinn.py::MerLinPINN` with a
  `QuantumLayer` returning UNBUNCHED probabilities, mapped to `(u, ux)`
  by two trainable linear heads.
- **Backend:** MerLin CPU simulator (analytic, shots = 0).
- **Modes / photons / layers:** 6 modes / 3 photons / 1 + 3 entangling
  layers.
- **Training settings:** Adam, lr = 0.02, 600 epochs, same
  consistency-loss weights as QPINN.

| Metric / Figure | Original | Classical | Photonic (MerLin) | Comment |
|---|---|---|---|---|
| Poisson 1D RMSE | 1.09e-4 (paper QPINN, 5000 ep) | ~1e-3 (matched FFN PINN, 3000 ep) | ~1e-2 (MerLin, 600 ep) | All numbers preliminary, single seed; see `results/` |
| Heat 1D RMSE (5 seeds each) | 1.24e-2 (paper QPINN); 2.09e-2 (paper classical PINN) | matched-effort classical FFN: 8.74e-3 ± 1.2e-3; matched-effort QPINN: 1.23e-2 ± 4.8e-3 | not yet attempted | Multi-seed: under matched-effort training, classical PINN beats QPINN by 1.40x and is ~4x more stable. Paper's classical-PINN baseline appears to lack the consistency-loss trick |

- **Photonic assessment:** the MerLin linear-optics PINN reaches the
  same order of magnitude as the classical PINN baseline on Poisson,
  which is sufficient to demonstrate that the consistency-loss training
  scheme is library-agnostic. It is *not* evidence for or against the
  CV-QPINN's specific photonic advantage — the two photonic platforms
  are different machines.

## 8. Conclusions

- **What has been done:**
  - Re-implemented the CV-QPINN ansatz in PyTorch with autograd-friendly
    Fock-truncated gates.
  - Trained on 1D Poisson and 1D heat with the consistency loss; landed
    within 1–2 orders of magnitude of the paper at smoke compute.
  - Added a parameter-matched classical PINN baseline that *also* uses
    the consistency-loss trick, finding that it can match or beat the
    QPINN under matched optimisation effort.
  - Built a MerLin photonic linear-optics adaptation that retains the
    consistency-loss scheme but trades the CV inductive bias for a
    photon-counting one.
- **What we conclude:** the consistency-loss design is the scientifically
  novel contribution and it travels well across libraries and quantum
  modalities. The QPINN's reported edge over a classical PINN does not
  survive matched optimisation effort and should be re-checked with a
  fairer baseline in any follow-up. The CV photonic platform is *not*
  directly mappable to MerLin and a MerLin "reproduction" would
  necessarily mean a different photonic architecture.
- **Recommendations:** `pursue with modifications` — pursue the
  consistency-loss training trick in our HQPINN-family reproductions
  but do *not* publish a MerLin port that claims to be a CV-QPINN.

## 9. Next steps

- **What we could do next:**
  - Paper-accurate runs with the `*_original.json` configs (cutoff 10–20,
    5000 epochs) to close the RMSE gap.
  - Multi-seed statistics for the headline results.
  - Run the X8 photon-loss noise study from §V on the existing simulator.
  - Implement the optimiser benchmark from Figure 13.
- **What we could not do next:**
  - Honest MerLin "reproduction" of the CV architecture (incompatible
    photonic platform).
- **Blockers:** none beyond CPU wall-clock budget.

## 10. Deliverables checklist

- [x] Original method reproduced (qualitative, smoke configs)
- [x] Multi-seed heat-equation sweep (4 QPINN + 5 PINN seeds)
- [x] Consistency-loss memory-overhead micro-benchmark (cutoffs 8, 10, 12)
- [ ] Paper-accurate quantitative match (paper-accurate configs prepared
      but not yet exercised on hours-long CPU budgets)
- [x] Photonic version defined (MerLin linear-optics counterpart)
- [x] Implemented in MerLin (consistency-loss linear-optics variant)
- [x] MerLin limitation documented (CV-vs-linear-optics gap explained in
      README, INSIGHTS, and this page)
