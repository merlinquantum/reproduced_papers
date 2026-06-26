# LOG.md — Generative Quantum Data Embeddings for Supervised Learning (EGAS)

Paper: arXiv:2605.30866v1, J. Heo & D. K. Park (Yonsei), 29 May 2026.
Local PDF: `$REPRO_SCRATCH_DIR/paper.pdf`, text: `$REPRO_SCRATCH_DIR/paper.txt`.

## Paper Summary
The paper proposes **EGAS** (Energy-based Generative Architecture Search): a quantum
data-embedding circuit is represented as a length-`D` sequence of depth-one subcircuit
tokens drawn from a fixed gate pool. An autoregressive GPT generator samples candidate
sequences; each is translated to an embedding circuit and scored by a **pairwise-fidelity
surrogate energy** `E(s) = mean_{(i,j) in batch} |δ_{y_i y_j} − F_Φ(x_i,x_j)|`
(same-class → high overlap, cross-class → low overlap). The GPT is updated by a
logit-matching loss toward a Boltzmann distribution over evaluated energies. After the
discrete search, a **continuous parameter refinement** stage adds a learnable MLP bias
`b_ω(x)` to gate angles (`φ̃(x)=r·x_i + b_ω(x)`), trained with a BCE fidelity loss. Searched
embeddings are evaluated with a **quantum-kernel SVM (QKSVM)**, `K_ij=F_Φ(x_i,x_j)`, and
compared to ZZ feature map, NQE (ZZ + neural preprocessing), and classical linear/RBF SVM.
Theory: a **Wasserstein bound** `D_tr(ρ+,ρ−) ≤ κ_F · W1(P̂+,P̂−)` (Eq. 7) shows attainable
class separation within an embedding family is limited by input-space geometry; small W1 ⇒
embedding search yields little gain (saturation).

## Compute Environment
- Python 3.12.3; torch 2.12.1 (CPU only, CUDA unavailable)
- pennylane 0.45.0 (`lightning.qubit`), scikit-learn 1.9.0, POT 0.9.6 (`ot`)
- CPU RAM: ~8 GB total; CPU cores: 10
- Docker / system: project container, `/opt/venv` venv.
- Additional dependencies installed this session: `pennylane`, `pennylane-lightning`,
  `ucimlrepo`, `pot`. See Dependency Additions.

## Claim Inventory
| ID | Claim | Evidence in paper | Reproduction test | Required baseline | Possible confounders | Status |
|---|---|---|---|---|---|---|
| C1 | EGAS finds embeddings whose QKSVM accuracy matches/exceeds ZZ and NQE across datasets | Fig 5, Fig 7 | Run EGAS, compare QKSVM acc vs ZZ/NQE on representative datasets | ZZ, NQE (fixed-circuit quantum) | reduced search, dataset/class choice, PCA, split protocol | PARTIAL (>ZZ; ≈NQE) |
| C2 | Continuous bias refinement gives additional gains (architecture/dataset dependent) | Fig 3, Fig 4, Fig 7 | Compare G/B groups before vs after bias on ΔE and QKSVM acc | self (before vs after) | overfitting, run variance | SUPPORTED (ΔE>0) |
| C3 | EGAS embeddings beat classical linear SVM on multiple splits in most datasets | Fig 5 | Win/tie/loss over 10 splits vs linear SVM | classical linear SVM | class balance, PCA, C value | PARTIAL (classical strong) |
| C4 | Wasserstein diagnostic: small input-space W1 ⇒ low embedding sensitivity (small IQR); D_tr ≤ κ_F·W1 | Fig 1, Fig 6, Table I | Reproduce Table I W1 distances; Fig 1 trace-dist vs W1; correlate W1 with IQR of QKSVM acc | n/a (theory) | PCA pipeline, class definition | SUPPORTED |

## Experiment Prioritization
1. **Wasserstein diagnostic (C4)** — Table I (W1 per dataset) + Fig 1 (trace dist vs W1 for ZZ).
   No GPT training; cheap and high scientific value. Reproduce broadly.
2. **EGAS core pipeline (C1, C2, C3)** — implement faithfully; run REDUCED iterations on a
   representative subset of datasets spanning the W1 range (high: WC/DB; low: WQ/MGT).
3. **QKSVM + baselines (ZZ, NQE, classical linear/RBF)** — supports C1/C3.
4. **MerLin photonic counterpart** — photonic fidelity-kernel QKSVM (≥2 photons), Phase 4.

## Experiment Inventory
| ID | Paper location | Description | Dataset | Metric | Paper value | Tier | Config | Status |
|---|---|---|---|---|---|---|---|---|
| E1 | Table I | Empirical 1-Wasserstein dist between class-conditional input dists | all 8 (PCA n=8) | W1 | see Table I | GREEN | wasserstein | DONE (5/7 match) |
| E2 | Fig 1 | Trace dist upper bound vs W1 (single/double ZZ layer, translated Gaussians) | synthetic | trace dist | curve saturates | GREEN | fig1 | DONE (qual.) |
| E3 | Fig 7 | QKSVM test acc per embedding (ZZ, NQE, classical, G/B ±bias) | subset | mean±std acc over 10 splits | Fig 7 table | AMBER | egas_* | DONE (3 ds) |
| E4 | Fig 3/4 | ΔE from bias refinement (G vs B groups) | subset | ΔE | Fig 3/4 | AMBER | egas_* | DONE |
| E5 | Fig 5 | Win/tie/loss vs classical linear SVM over 10 splits | subset | W/T/L | Fig 5 | AMBER | egas_* | DONE |
| E6 | Fig 6 | IQR of mean QKSVM acc across embeddings; correlate with W1 | subset | IQR | Fig 6 | GREEN | (derived from E3) | DONE |
| P1 | Phase 4 | MerLin photonic fidelity-kernel QKSVM counterpart | subset | acc | n/a (adaptation) | AMBER | merlin | DONE |

## Key paper hyperparameters (Appendix A)
- Qubits n=8; features PCA→8; rescaled to [0, 2π]; binary task (two classes w/ enough points).
- Token pool C: gates {RX,RY,RZ} (param), {H,I} (non-param), {CNOT, MultiRZ} (two-qubit).
  Token = (gate_type, qubit_index, data_index, coeff r ∈ {0.1,0.3,0.5,0.7,1.0}); param angle φ(x)=r·x_i.
- Sequence length D=28 (matches single-layer ZZ depth). ZZ map: U_ZZ Eq. A1, ϑ_j(x)=x_j, ϑ_{j,j+1}=(π−x_j)(π−x_{j+1}).
- EGAS: GPT vocab |C|+1 (start token); 4000 iters; temp 100→0.04 linear; energy EMA normalize;
  select top-k + bottom-k + middle k/2; Adam lr=5e-5, wd=1e-2, β=(0.9,0.999); logit-match loss Eq.10 (γ).
- Bias MLP: zero-init output head, output ×10 gain; RMSprop lr=5e-4, grad clip 2.0, 400 epochs,
  batch 25, L2 1e-6 on bias output; metrics avg over last 10 epochs. BCE loss Eq.12 with clip ε.
- QKSVM: precomputed fidelity kernel, C=0.05. Classical linear SVM on z-scored features (stats from train).
  RBF SVM C=0.05, gamma=0.125. Eval: slices of 400 train + 50 test, 10 repeats, same split per embedding.
- Quantum sim: PennyLane lightning.qubit, analytic (shots=None).

## Available Resources
- Original repo: NONE found in paper (no code link). Searched: paper text. (TODO: web search.)
- Datasets: all public UCI/OpenML — see Data Acquisition Log.
- Framework in paper: PennyLane (lightning.qubit). Reproduction uses same.
- Pretrained weights: none.
- Hardware access: none (CPU sim only). Matches paper (shots=None analytic sim).
- Related reproduction in repo: `papers/nn_embedding/` reproduces NQE (ref [19], key baseline).
  Reuse NQE/quantum-kernel patterns there.

## Data Acquisition Log
- Source: `ucimlrepo` (UCI ML Repository) — confirmed working (network OK).
  Wine Quality (id 186) fetched: 6497×11. IDs: PW=327, WDGV1=107, DB=602, WineQuality=186,
  MGT=159, EGSSD=471. Pol is OpenML (ref [52,53]) — fetch via openml or skip.
- WC (Wine Color) = red vs white from Wine Quality source; WQ (Wine Quality) = quality binary.
- Fraction obtained: full small CSVs; subsample per slice protocol (400+50).
- Fallback: none needed (real data accessible).

## Fair Baseline Plan
- Advantage axis: accuracy/separability at fixed downstream learner (QKSVM) and fixed input geom.
- Baselines: (a) classical linear SVM on z-scored PCA features; (b) classical RBF SVM (C=0.05, γ=0.125);
  (c) quantum ZZ feature map kernel; (d) NQE (ZZ + trainable neural preprocessing).
- Matching criterion: identical PCA features, identical 10 train/test splits, identical QKSVM C.
- Seeds: paper uses 10 repeated splits; reduced runs labeled.

## Strategy and Key Decisions
- 2026-06-23: Paper is gate-based QML (not photonic). Plan: faithful EGAS implementation,
  REDUCED-compute runs (fewer EGAS iters / candidates) on representative datasets spanning the
  W1 range. Reproduce cheap high-value theory (Table I, Fig 1) broadly. MerLin photonic
  counterpart in Phase 4 (≥2 photons). Target validity tier V2 (reduced-compute real-data).
- Ambiguities to resolve with documented defaults: exact two-class definitions per dataset,
  GPT architecture size (paper unspecified — use small GPT), γ value (unspecified), M candidates/iter.

## Dependency Additions
- `pip install pennylane pennylane-lightning ucimlrepo pot`
  - Reason: paper uses PennyLane lightning.qubit; ucimlrepo for UCI datasets; POT for exact
    multivariate 1-Wasserstein (Table I); sklearn used for SVM.
  - Restore on fresh Docker: `pip install pennylane pennylane-lightning ucimlrepo pot`

## Results Summary (reduced-compute, single seed)
EGAS QKSVM mean test accuracy (8 splits), ordered by W1:
| DS | W1 | bestG | bestG(bias) | NQE | ZZ | Lin | RBF | IQR | EGAS minE |
|----|----|-------|-------------|-----|----|-----|-----|-----|-----------|
| WQ | 2.74 | 0.583 | 0.565 | 0.633 | 0.525 | 0.647 | 0.617 | 0.055 | 0.456 |
| MGT| 3.00 | 0.738 | 0.755 | 0.705 | 0.488 | 0.732 | 0.728 | 0.167 | 0.424 |
| PW | 4.92 | 0.902 | 0.882 | 0.907 | 0.512 | 0.900 | 0.863 | 0.269 | 0.400 |

Claim verdicts:
- **C1 PARTIAL-SUPPORTED**: EGAS > ZZ on all 3; ≈ NQE (above on MGT, ~tied PW, below WQ). Reduced search.
- **C2 SUPPORTED (ΔE), nuanced (acc)**: bias refinement reduces surrogate energy on all (G:+0.047..+0.071,
  B:+0.046..+0.134; larger for B group, as Fig 3/4). Accuracy effect dataset-dependent (helps MGT,
  slightly hurts WQ/PW) — matches paper's stated nuance.
- **C3 PARTIAL/NEGATIVE (fair baseline)**: classical linear SVM strong; reduced EGAS beats it only on
  MGT (WTL 4/2/2); loses on WQ (1/0/7), mixed on PW (3/1/4). Honest fair-baseline result. (F6-adjacent.)
- **C4 SUPPORTED (the standout result)**: IQR rises monotonically with W1 (0.055→0.167→0.269) and EGAS
  min-energy falls with W1 (0.456→0.424→0.400). Table I matches 5/7; Fig 1 saturation qualitatively
  reproduced. Low-W1 datasets (WQ,MGT) in saturation regime — exactly as predicted.

Reproduction confidence: MEDIUM. Implementation confidence: MEDIUM-HIGH (statevector engine validated
to 1e-16 vs PennyLane; pipeline deterministic; baselines fair). Validity tier: V2 (reduced-compute,
real-data). Failure classes touched: F3 (underspecification of GPT/γ/classes), F5 (Table I preprocessing).

## Reproduced Figures and Tables (progress)
- **Table I (W1)**: pipeline = StandardScaler → PCA(8) → per-feature MinMax[0,2π]; W1 = exact L1
  optimal transport (POT). Reproduced (run config `wasserstein.json`):
  PW 4.92 (paper 5.24), WDGV1 5.16 (5.16), WQ 2.74 (3.01), MGT 3.00 (3.30), EGSSD 4.41 (3.56) —
  close. DB 3.38 (paper 13.91) and WC 3.73 (10.86) come out much smaller: per-feature MinMax to
  [0,2π] caps per-component class separation, so the two *most-separable* datasets are compressed.
  The diagnostic-relevant ordering (WQ, MGT smallest ⇒ saturation regime) IS reproduced. (F5:
  preprocessing/metric ambiguity for the absolute magnitudes of the most-separable sets.)
- **Fig 1 (trace dist vs W1)**: reproduced qualitatively — trace distance rises with input W1 then
  saturates; absolute scale differs (synthetic Gaussian setup, n=4). `fig1.json`.

## Resolved Decisions / Defaults (underspecified in paper)
- GPT: small autoregressive transformer (reduced: d_model=32, 1 layer, 2 heads). Paper omits size.
- γ (inverse temp in Eq.10): 0.1; energies EMA-normalized. exp() args clamped + grad-clip for
  stability (raw logit-sum w_sum is unbounded → loss exploded at γ=1; RESOLVED).
- Two-qubit token wires = (q,(q+1) mod n) nearest-neighbour ring. H/I ignore data/coeff.
- Per-dataset binary task = two most-populous classes (WC = red vs white; targets→{-1,+1}).
- Reduced compute (V2): EGAS 120 iters / 12 candidates (paper 4000 iters); top-4 G & B; 8 splits.
  Justification: cost governance + search energy plateaus early; clearly labeled partial.

## Blockers and Open Questions
- RESOLVED (2026-06-23): No original code repo; used documented defaults above.
- RESOLVED (2026-06-23): logit-matching loss numerical blow-up — fixed via γ=0.1 + exp clamp + grad clip.
- OPEN (2026-06-23): DB/WC absolute W1 mismatch (F5) — preprocessing of most-separable sets;
  does not affect the saturation diagnostic (low-W1 datasets correctly identified).

## Hardware-Aware Result Summary (MerLin photonic, P1)
Photonic fidelity-kernel QKSVM on MGT (reduced-scope), `outdir/phot_MGT2/run_*`:
| Field | Value |
|---|---|
| Computation space | UNBUNCHED |
| Detector model | threshold |
| Photon number | 2 |
| Number of modes | 8 |
| Input state | [1,0,1,0,0,0,0,0] |
| Encoding | angle (CircuitBuilder.add_angle_encoding), 2 entangling layers |
| Measurement | FidelityKernel (SLOS transition probability) |
| Postselection | none |
| Simulator / QPU | MerLin SLOS analytic (shots=None) |
| Wall-clock | ~7.5 min (3 splits × 12 training epochs, full CPU) |
| Seeds | 1 (seed 0), 3 splits |

Result (mean acc ± std): photonic-trained 0.733±0.025 > photonic-fixed 0.687±0.025 ≫ ZZ 0.487±0.066;
classical-linear 0.753±0.019, classical-rbf 0.720±0.043. Trained photonic mesh (continuous EGAS
analogue) beats data-agnostic ZZ and approaches classical with only 2 photons. P1 DONE.

## Cost
- Estimated API cost ~ $26 of $50 budget at end of main experiments (CPU compute is free; cost is
  agent reasoning/tooling). Within budget.

## Session Handoff
### Session — 2026-06-23
- Python 3.12.3; project container venv `/opt/venv`. Packages installed: pennylane,
  pennylane-lightning, ucimlrepo, pot (restore: `pip install pennylane pennylane-lightning ucimlrepo pot`).
- Completed: full EGAS pipeline (`lib/`), statevector engine validated vs PennyLane (1e-16),
  Table I (`wasserstein.json`), Fig 1 (`fig1.json`), EGAS eval on PW/WQ/MGT (`outdir/{PW,WQ,MGT}/run_*`),
  MerLin photonic counterpart (`lib/photonic.py`, `photonic_eval` task), README/INSIGHTS/FEEDBACK/
  CONFLUENCE/ACTION_REQUIRED, notebook, plots in `results/`. 7 pytest pass.
- Key results: C4 supported (IQR & search-energy track W1); C1 partial (EGAS>ZZ, ≈NQE); C2 supported
  (ΔE>0); C3 fair-baseline finding (classical linear strong; EGAS wins only on MGT). Tier V2.
- IMPORTANT gotcha: parallel runs starting in the same second collide on `run_TIMESTAMP` dir →
  use distinct `--outdir` per run (fixed by writing to outdir/<DS>).
- Last successful command:
  `python implementation.py --paper generative_quantum_embeddings --config configs/photonic_MGT.json --outdir outdir/phot_MGT2`
- Exact next step: read `outdir/phot_MGT2/run_*/metrics.json`, fill photonic row in README/CONFLUENCE,
  regenerate `results/egas_summary.png`. Then reproduction is complete.
- Open blockers: none (DB/WC Table I magnitude mismatch documented, F5; does not affect verdicts).
