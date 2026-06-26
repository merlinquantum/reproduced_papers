# Generative Quantum Data Embeddings for Supervised Learning — reproduction

- **Paper:** arXiv:2605.30866v1 — J. Heo, D. K. Park (Yonsei), 29 May 2026
- **Original code:** none found (independent reimplementation; GPT scheme follows GQE, ref [40])
- **Internal repo / branch:** `reproduced_papers` → `papers/generative_quantum_embeddings/`
- **Jira ticket:** [PAPER 67](https://quandela.atlassian.net/browse/PAPER-67)
- **PR reproduced_papers:** TBD
- **PR MerLin / Perceval:** n/a (see `ACTION_REQUIRED_SEND_TO_MERLIN_TEAM.md`)

## 1. Executive summary
- **What the paper does:** Optimizes the *structure* of a quantum data-embedding circuit using an
  energy-based generative architecture search (EGAS): a GPT samples gate-token sequences scored by
  a pairwise-fidelity surrogate for class separability, refined by a continuous bias, and evaluated
  with a quantum-kernel SVM. It also derives a Wasserstein bound that predicts when embedding search
  can help.
- **Why it matters:** Replaces hand-designed, data-agnostic feature maps with a data-tailored,
  searchable embedding family — and gives an a-priori diagnostic for when it is worth it.
- **Main claims:** 
  - C1: EGAS ≥ ZZ/NQE; 
  - C2: bias refinement adds gains; 
  - C3: EGAS > classical on most
    datasets; 
  - C4: small input-space W1 ⇒ embedding-search saturation.
- **Bottom line:** partially reproduced (reduced-compute, single-seed, 3/8 datasets).
- **Main takeaways:** The **Wasserstein diagnostic (C4) reproduces cleanly and is the strongest part
  of the paper**; EGAS reliably beats the data-agnostic ZZ map; against a *fair classical linear SVM*
  the advantage is dataset-dependent and modest under reduced search.

## 2. Paper overview
- **Core idea:** treat the embedding circuit architecture as the optimisation target via generative
  (GPT) search over discrete gate tokens, guided by a fidelity surrogate; bound attainable
  separability by input-space Wasserstein geometry.
- **Similar work / Quandela DB:** closely related to NQE (Hur et al. 2024), already reproduced at
  `papers/nn_embedding/`; also quantum-kernel work in `papers/AA_study`, `papers/photonic_quantum_enhanced_kernels`.
- **Method summary / pipeline:** GPT → token sequences → embedding circuits → fidelity-surrogate
  energy → logit-matching GPT update (Boltzmann); then continuous bias MLP; then QKSVM (K=fidelity).
- **Key takeaways from the paper:** generative search finds competitive embeddings; gains are
  geometry-limited (Wasserstein) and saturate on weakly-separated datasets.

## 3. Reproduction scope
- **Targeted:** full EGAS pipeline + bias refinement + QKSVM + ZZ/NQE/classical baselines; Table I;
  Fig 1; Figs 3–7 behaviour on PW, WQ, MGT; MerLin photonic counterpart.
- **Not targeted:** full 4000-iteration search; all 8 datasets at full scale; multi-seed statistics.
- **Success criteria:** reproduce the qualitative claim directions (C1–C4) with fair baselines and
  honest labelling, not exact numbers.

## 4. Original method
| Item | Paper | Reimplementation | Notes |
| --- | --- | --- | --- |
| Architecture | GPT over D=28 gate tokens; pool {RX,RY,RZ,H,I,CNOT,MultiRZ}, n=8 qubits | same pool/D; small GPT (d_model=32, 1 layer) | GPT size unspecified in paper |
| Training setup | 4000 iters, temp 100→0.04, EMA-norm, top/mid/bottom select, Adam 5e-5 | 120 iters (reduced), same schedule/select/optimizer | reduced compute |
| Hyperparameters | γ unspecified; bias RMSprop 5e-4, 400 ep; QKSVM C=0.05; RBF γ=0.125 | γ=0.1; bias 120 ep; C=0.05; RBF γ=0.125 | γ default + loss stabilisation |
| Missing details / assumptions | class defns, PCA/scaling, 2-qubit wiring | two largest classes; StandardScaler→PCA8→MinMax[0,2π]; NN-ring | documented in LOG.md |

## 5. Reproduction implementation
### 5.1. Quantum implementation
- **Repo / scripts:** `lib/{statevec,circuits,egas,gpt,bias,kernel_svm,wasserstein}.py`, `lib/runner.py`.
- **How to run:** `python implementation.py --paper generative_quantum_embeddings --config configs/egas_PW.json --outdir outdir/PW` (and `wasserstein.json`, `fig1.json`).
- **Compute used:** CPU only (10 cores, 8 GB). Custom batched **differentiable torch statevector
  engine**, validated to 1e-16 vs PennyLane; analytic (shots=None), as in the paper.
- **Deviations:** reduced iterations/splits/datasets; small GPT; γ + numerical stabilisation of
  the logit-matching loss.

### 5.2. Classical comparison
- **Present in the paper:** yes (linear SVM on standardized features; RBF in appendix).
- **Description of baseline:** linear SVM (C=0.05) and RBF SVM (C=0.05, γ=0.125) on z-scored PCA
  features — reproduced and used as the fair baseline.

## 6. Reproduction results
- **Result status:** partially reproduced.
- **Figures reproduced:** Table I (5/7 close), Fig 1 (qualitative saturation), Fig 6 IQR-vs-W1 trend,
  Figs 3/4 ΔE>0, Fig 7-style accuracy comparison (3 datasets).
- **Headline numbers** (mean acc, 8 splits): PW best-G 0.902 / NQE 0.907 / ZZ 0.512 / lin 0.900;
  MGT G(bias) 0.755 / NQE 0.705 / ZZ 0.488 / lin 0.732; WQ G(bias) 0.565 / NQE 0.633 / lin 0.647.
  IQR vs W1: 0.055(2.74) < 0.167(3.00) < 0.269(4.92) — monotone, supports C4.
- **Explanation of differences:** reduced search (120 vs 4000 iters) → EGAS doesn't always reach
  NQE/classical; preprocessing ambiguity → DB/WC W1 underestimated.
- **Comparison to baseline:** EGAS ≫ ZZ everywhere; ≈ NQE; vs classical linear, wins only on MGT.

## 7. Photonic translation
- **Photonic objective:** preserve the role of a quantum data embedding scored by a fidelity kernel.
- **Proposed formulation:** photonic embedding = angle encoding + trainable interferometric mesh;
  fidelity kernel `|⟨s|U†(x₂)U(x₁)|s⟩|²` via SLOS; QKSVM downstream (C=0.05). Trainable mesh
  optimised with the EGAS pairwise-fidelity surrogate = continuous photonic analogue of the search.
- **Encoding:** `CircuitBuilder.add_angle_encoding` on 8 modes.
- **Circuit / model:** MerLin `FeatureMap` + `FidelityKernel`, ≥2 photons, UNBUNCHED, threshold det.

### 7.1. MerLin feasibility
- **Can this be done in MerLin?** Yes — `FidelityKernel` is purpose-built for this.
- **Limitation:** training the mesh through `FidelityKernel` (autograd over SLOS) is slow (~10s/epoch),
  bounding scope. See `ACTION_REQUIRED_SEND_TO_MERLIN_TEAM.md`.
- **Fallback used:** None (pure MerLin).

### 7.2. Photonic implementation and results
- **What was implemented:** fixed + trained photonic fidelity-kernel QKSVM (`lib/photonic.py`).
- **Backend:** MerLin SLOS analytic (shots=None). **Modes/photons/layers:** 8 / 2 / 2.
- **Training settings:** Adam lr=0.05, 25 epochs, batch 30, 5 splits (reduced).

| Metric / Figure | Original (gate EGAS) | Classical (linear) | Photonic (fixed / trained) | Comment |
| --- | --- | --- | --- | --- |
| MGT acc | 0.755 | 0.753 | 0.687 / **0.733** | 2 photons, 8 modes, 3 splits; trained > fixed (+0.046) |
| MGT ZZ (gate) | 0.487 | — | — | data-agnostic gate baseline (both gate-ZZ and photonic-fixed beat it only after training) |

- **Photonic assessment:** Feasible and meaningful in MerLin. The trainable photonic mesh
  (continuous analogue of EGAS's discrete search), optimised with the same pairwise-fidelity
  surrogate, clearly beats the data-agnostic ZZ map (0.733 vs 0.487) and approaches the classical
  linear baseline (0.753) with only 2 photons — the method's inductive bias survives the photonic
  mapping. Limitation: training through `FidelityKernel` (SLOS autograd) is slow, bounding scope to
  1 dataset / 3 splits / 12 epochs.

## 8. Conclusions
- **What has been done:** faithful reduced reproduction of EGAS + Wasserstein diagnostic + fair
  baselines + a MerLin photonic counterpart, all clearly labelled.
- **What we conclude:** the Wasserstein diagnostic (C4) is robust and reproduces cleanly; EGAS beats
  data-agnostic feature maps but its edge over NQE and especially a fair classical linear SVM is
  modest under reduced compute. Quantum-advantage claims must be made against the classical baseline.
- **Recommendation:** pursue with modifications (the geometry diagnostic is the reusable asset;
  add random-search ablation and full-scale multi-seed runs before any advantage claim).

## 9. Next steps
- **Could do next:** full 4000-iter multi-seed search; all 8 datasets; random-sequence-search
  ablation to isolate the GPT's contribution; faster photonic training path.
- **Could not do:** full-scale runs within the CPU/budget envelope.
- **Blockers:** photonic mesh-training cost; Table I preprocessing ambiguity (DB/WC).

## 10. Deliverables checklist
- [x] Original method reproduced (reduced)
- [x] Results reported (README, this page)
- [x] Photonic version defined
- [x] Implemented in MerLin
- [x] MerLin limitation documented
- [~] Photonic version run (reduced scope; see table)
- [x] Figure reproduced / adapted (Table I, Fig 1, Figs 3/4/6/7-style)
- [ ] PR to reproduced_papers prepared
- [ ] PR to MerLin prepared
- [x] Final recommendation written
