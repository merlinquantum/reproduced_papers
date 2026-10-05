# Q-CHIPP: evidence and reproduction specification

## Reference and scope
Ryan Peters et al., **Quantum convolutional HLA immunogenic peptide prediction (Q-CHIPP): Next-generation neoantigen prediction with quantum neural network**, Science Advances (2026), DOI 10.1126/sciadv.aec3824. The open article includes the complete main text, Tables 1–5, Figures 1–7, methods and references.[1]

The source method is a gate-based QCNN, **not a photonic model**. Q-CHIPP is the AND combination of two independent predictions: MHC binding from peptide positions 2, 4, 5, 9 and immunogenicity from positions 3, 5, 6, 7, with the latter trained exclusively on experimentally confirmed binders.[1] This distinction is central: a binder model alone is not an immunogenicity predictor.

The reference code is the authors' release v1.0.0, commit `9f72bd9fc87522bc45c806f9849d07210a64e48d`, archived by Zenodo DOI 10.5281/zenodo.19186890.[4][5] Figure-code release `qchipp_publication`, commit `4b19620deb7e890bd2a306c4bc518879d6e2b2db`, is linked by the separate Zenodo concept record 19187502.[6][8]

## Main claims and tests

### Current evidence classification (supersedes the earlier status table below)
| Claim | Status | Evidence / limitation |
|---|---|---|
| C1 small-data binding | PARTIALLY TESTED | Three-seed source-circuit 100-update binding result is near the reported range, but release test size differs; no paper-accurate numerical replication |
| C2 QRAC versus basis compression | UNRESOLVED | QRAC forward independently verified; 20-qubit basis performance comparison omitted from bounded scope |
| C3 IBM warm-start hardware robustness | UNRESOLVED / BLOCKED | No QPU access or historical warm-start assets; optical sampling is not IBM device noise |
| C4 compact quantum advantage over classical | PARTIALLY TESTED | Source 81 parameters versus canonical one-hot logistic 81 and matched photonic controls; no demonstrated advantage under reduced protocol; original limited/large CNN architectures not replicated |
| C5 binder-population conditioning | PARTIALLY TESTED | Provided BA-rank diagnostic on public release tests changes AUROC from 0.956 binding to 0.761 mixed immunogenicity and 0.334 binder-only; different cohort from paper Fig. 5, no causal conclusion |
| C6 TCR positions and immunogenicity | PARTIALLY TESTED | Prespecified 3, 5, 6, 7 evaluated on genuine binder-only split; published F1 not matched; no new position search/test selection |
| C7 full 46-qubit peptide model | UNRESOLVED / BLOCKED | Structural 399-parameter count verified; full simulation/hardware execution outside feasible local scope |
| C8 clinical survival utility of AND framework | UNRESOLVED / BLOCKED | Trained two-head AND inference and checkpoint restoration work; no patient-level survival validation or clinical claim |
| C9 distinctive predicted peptide set | UNRESOLVED | Clinical 209,889-peptide prediction and intersection assets unavailable |
| C10 differential motifs | UNRESOLVED | No validated clinical prediction sets or new experimental motif validation |

### Current experiment coverage
| Experiment | Status | Scope and reason |
|---|---|---|
| E1 positions 2, 5, 9 plus BA binding | BLOCKED in bounded protocol | Exact historical settings unresolved; not implemented in the four-position photonic architecture; no claim of infeasibility in general |
| E2 BA-only fitted classifier | PARTIAL | Provided-rank diagnostic executed without fitting; exact source BA-only QCNN omitted |
| E3 four-position binding | PARTIAL | Source QRAC, corrected loss, conventional/one-hot classical and photonic comparisons executed; basis 20 and the exact historical split omitted |
| E4 IBM 20/46 timing and warm start | BLOCKED | No hardware access; source structural counts only |
| E5 binder-only immunogenicity | PARTIAL | Actual 300/1000 release split, source/corrected/photonic/classical comparisons; protocol differs from the reported 300/100 |
| E6 BA-only binder immunogenicity | PARTIAL | Raw BA-rank diagnostic; no trained BA-only quantum model |
| E7 mixed-population trained models | PARTIAL | Released mixed test population evaluated with provided affinity scores; trained mixed QCNN variants not run |
| E8 exhaustive 126-position search | BLOCKED in bounded protocol | Prespecified published positions only; stopped to avoid speculative tuning before historical configuration resolution |
| E9 basis 46 hardware | BLOCKED | Local statevector/hardware resources inadequate; no substituted toy 46-qubit result |
| E10 internal Alban benchmark | PARTIAL | Public DataS1 conditioning diagnostic is an explicit different-cohort comparison; internal cohort assets absent |
| E11 clinical survival/intersections/motifs | BLOCKED | Original patient assets/access unavailable; not approximated with synthetic data |
| E12 gate/parameter structure | DONE | Tagged-source oracle and structural parameter counts 4/8/10/20/46; no hardware gate-depth timing replication |
| E13 supplementary analysis | DONE | Authentic PDF and DataS1 verified and read; measured versus extrapolated timing distinction recorded |

BLOCKED here denotes an omitted bounded-study item or missing resource, not a general claim that the method cannot be implemented. Reference-mode physics correction remains under benchmark; all earlier photonic 6-mode results are retained as phase-invariant ablations.

| ID | Claim | Paper evidence | Reproduction test | Required baseline | Confounders | Current status |
|---|---|---|---|---|---|---|
| C1 | Small-data QCNN learns peptide-MHC binding | Table 1; Fig. 3; 150/50 train/test | Fixed-split QRAC and basis experiments at specified residues | Limited/large CNN; RF; BA-only control | BA rank already predicts binding; split/encoding ambiguity | Untested |
| C2 | QRAC compression changes classification performance | Fig. 3B–C: QRAC F1 0.73 versus basis 0.80 | Same raw data/positions with 10 versus 20 qubits | Same classical inputs | Computational budgets and checkpoints | Untested |
| C3 | Warm-start hardware training preserves simulated behavior | Fig. 3D; F1 about 0.80 | Same weights, circuit, layout, estimator/shot and mitigation settings on IBM hardware | Simulator warm-start reference | Device drift, readout, shot noise | Not accessible locally |
| C4 | Parameter-constrained classical models perform worse | Tables 1, 2, 4, 5 | Same split/features with exact parameter counts plus tuned classical control | Limited CNN, large CNN, RF and matched-budget MLP | Table 2 budgets not exactly matched; optimizer effort | Untested |
| C5 | Removing nonbinders exposes immunogenicity difficulty | Fig. 5; Table 5 | Binder-only versus mixed population evaluation | BA-rank threshold and RF | AUC is not accuracy; prevalence alone does not explain AUC change; population composition matters | Untested |
| C6 | QCNN extracts TCR-relevant positional information | Fig. S3; Table 5: 3, 5, 6, 7 F1 0.67 (0.70 with BA) | Declared validation-only position selection; final held-out test | Same positional search budget for classical control | Selecting among 126 combinations; manuscript/code selection conflicts to inspect | Untested |
| C7 | 46-qubit full peptide model works with 150 samples | Fig. 6; Table 4 F1 0.70 | Source-hardware run, 20k shots, noise mitigation | Full-sequence classical models on same 150/50 split | Original classical row used 300 training; drift after iteration 400 | Not locally reproduced |
| C8 | AND combination predicts clinically informative neoantigen load | Fig. 7: Q-CHIPP log-rank p=0.0085, NetMHC p=0.0275 | Exact patient predictions, aggregation, median grouping, survival endpoint | CNN AND, RF AND and individual heads | Gated clinical provenance; observational association, multiple models | Untested |
| C9 | Predictions differ across methods | Fig. 4, 7E: 25,542 Q-CHIPP-only peptides | Exact identity-based intersections and labels | Source comparator predictions | Different predictions do not establish correctness | Untested |
| C10 | Peptide motifs differ across models | Fig. 7F–H | Frequency ratio with epsilon 1e-6 and clipped log2 heatmaps | CNN/RF motif ratios | No experimental validation of newly predicted peptides | Untested |

## Experiment inventory

All reported experiment families are retained here; implementation should prioritize E1 and E5 before hardware or clinical work.

| ID | Location | Task and data | Metric / paper value | Proposed feasibility |
|---|---|---|---|---|
| E1 | Table 1/Fig. 3A | MHC binding, positions 2, 5, 9 plus BA, QRAC 8, train 150 / test 50 | F1 QCNN 0.93; limited CNN 0.84; large CNN 0.90; RF 0.88 | CPU pilot and reduced/full iterations after timing |
| E2 | Table 1 BA row | MHC binding, BA only | F1 0.89/0.90/0.90/0.88 (QCNN/limited/large/RF) | Low cost |
| E3 | Fig. 3B–C, Table 1 | Binding positions 2, 4, 5, 9, QRAC 10 versus basis 20 | QRAC 0.73 in prose; basis 0.80 in Table 1; 0.66/0.88/0.83 classical | QRAC feasible; basis 20 needs a pilot, RAM constraints |
| E4 | Fig. 3D–E/Table 3 | Warm-start basis 20, hardware 20/46 shot-runtime comparison | F1 about 0.80; 20k shots about 15–17.5 s; gate counts below | IBM access required, no equivalence from local photons |
| E5 | Table 5 | Binder-only immunogenicity, positions 3, 5, 6, 7, QRAC 10, train 300 / test 100 | F1 0.67/0.42/0.68/0.65; with BA 0.70/0.35/0.65/0.69 | Central useful local experiment |
| E6 | Table 5 BA row | Binder-only immunogenicity, BA only | F1 0.54/0.33/0.54/0.62 | Low cost |
| E7 | Table 4 | Mixed binder/nonbinder immunogenicity, train 300 / test 100 | BA 0.73/0.73/0.73/0.68; positions 1, 4, 5, 9 0.71/0.33/0.67/0.64, with BA 0.71/0.50/0.72/0.72; positions 4, 6, 7, 8 0.49/0.39/0.33/0.52, with BA 0.54/0.35/0.68/0.72; positions 4, 6, 8, 9 0.70/0.35/0.71/0.68, with BA 0.74/0.55/0.71/0.77 | More tasks, preserve configs even when omitted |
| E8 | Fig. S3 | 126 four-position searches, with/without BA, mixed and binder-only | Mixed best 1, 3, 4, 9 0.77 (0.73 BA), average 0.64; binder-only best 3, 5, 6, 7 0.67 (0.70 BA), average 0.51/0.52 | Expensive; prespecified validation selection needed |
| E9 | Fig. 6/Table 4 | Full nine-residue basis 46 hardware immunogenicity, train 150 / test 50 | F1 0.70; limited/large/RF 0.60/0.65/0.66 on larger source training split | RED: hardware unavailable; never simulate the full 46-qubit statevector locally |
| E10 | Fig. 5/Fig. S4 | Internal Alban cohort 1253/171 immunogenic; binder-only versus mixed benchmarking | NetMHC binding AUC 0.85; mixed immunogenicity AUC 0.79; binder-only AUC 0.55 | Depends on public assay records |
| E11 | Fig. 7/S5 | Clinical 111 HLA patients, 209,889 clean peptides, AND heads, survival, intersections, motifs | p=0.0085 Q-CHIPP, 0.0275 NetMHC, 0.0052 RF | Patient assets/access provenance must be verified |
| E12 | Tables 2–3 | Parameter and gate counts for 4, 8, 20, 46 qubits | Params 21/54/168/399; gates 54/140/458/1190; 2q 18/49/177/524; 2q depth 13/22/42/63 | Structural circuit tests |
| E13 | Fig. S1/S2, Tables S1/S2 | Encoding pipeline, full 20 circuit and supplemental characteristics | Authentic PDF and DataS1 verified and read (see E13 status above) | None for the selected scope |

Numbers above are paper-reported, not obtained reproduction results.[1]

## Encoding and data protocol
- Target allele HLA-A*02:01; peptides exactly nine canonical amino acids. Primary IEDB retrieval 8470 records, 34% binders, 36% of binders confirmed immunogenic; deduplication described but count after deduplication unclear.[1]
- Source `mylib.py:196–253` fits `category_encoders.BinaryEncoder` on a sorted union of categories from train **and test**, across selected motif columns. Codes are category ranks starting at 1, not raw amino-acid integer bits. Width depends on observed category count. This is transductive preprocessing; a reproduction must disclose it and offer a train-only mapping, never silently call the source mapping leakage-free.[4]
- Source positions are zero-based `motif0` through `motif8`; displayed paper positions are one-based. BA is binary `rank < 2.0` (strict inequality). Model labels are source {-1,+1}; application metrics should explicitly convert to {0,1}.[1][4]
- QRAC 2:1: for pair b0, b1, `RY(pi/4 + pi/2*(b1-b0) + pi*b1*b0)` on one qubit (`qrac_parametervector.py:37–51`). Pad classical bits so both the QRAC grouping and the even-qubit pyramid requirements hold. Basis is `RX(pi*b)` (`qcnn_lib.py:45–57`), equivalent to computational basis for binary bits up to input-dependent global phase.[4]
- Manuscript Methods conflict: Datasets says split seed 42; Dataset splits says pandas.sample/train_test_split seed 1. Initialization seed 49 is separate. Use released exact files if identity and label evidence are valid; otherwise document selected seed and do not claim exact numerical replication.[1]
- Immunogenicity head must exclude nonbinders before sampling/training. Do not derive the binding truth label from BA prediction. Preserve official released splits when valid and audit exact peptide overlap and conflicts.

## Gate architecture specification
Released `qcnn_lib.py:311–382` uses the pyramid ansatz. Start on n even qubits, global center=n/2; while active width>2, process the central register and update active width to `ceil(ceil(width/2)/2)*2`.[4]

For width divisible by 4, convolution first processes even nearest-neighbor pairs then shifted odd pairs, with no periodic wrap. For width not divisible by 4, the shifted pass occurs **before** the even pass. Pooling uses the source/sink layout in lines 343–350, preserving the central two qubits when active width is 2 modulo 4. Two mirrored swap pyramids compact the retained register. Final convolution/pooling acts on the center pair.[4]

Each convolution has RZ(-pi/2) on local 1, CX 1→0, RZ(theta0) on 0, RY(theta1) on 1, CX 0→1, RY(theta2) on 1, CX 1→0, RZ(pi/2) on 0. Pooling drops the last CX/RZ portion and has two CX plus three trainable gates (`qcnn_lib.py:470–483,595–605`). Parameters are distinct per local unitary; despite manuscript wording about strong sharing, the source does **not** tie them across positions.[4]

Readout conflict: source pyramid estimator uses `0.5*Z` without identity offset (`qcnn_lib.py:222–230`); due to Qiskit little-endian Pauli strings, its Z targets physical qubit `n/2`, not `n/2-1`. Preserve explicit bit order in tests. A probabilistic binary cross-entropy loss requires a mapping from signed expectation to probability; source `NeuralNetworkClassifier` semantics must be inspected rather than assuming direct BCE on `0.5*Z` is equivalent.[4]

## Optimizers and classical benchmarks
Manuscript gate training: SPSA, learning schedule `.06/(k+1)^.602`, perturbation `.2/(k+1)^.101`, initialization seed 49; max iteration count experiment-specific (around 300/660/350 discussed). Release default instead says 5 iterations and learning-rate amplitude 1.62. These defaults are a test config, not paper-accurate experimental settings.[1][4]

Limited CNN: Conv1D kernel 3, same padding, ReLU, L2 0.0005, maxpool 2, global-average-pool, sigmoid output; exact filter schedule must be taken from released figure code. Large CNN prose lists 32, 54, 128 filters, hidden 128/64, dropout 0.3, while the reported 55,873 parameters may correspond to a different filter width; verify counts. Both use Adam/BCE. RF: 300 trees on one-hot residue features, 100 repetitions reported; not the same input representation as QCNN.[1]

## Clinical and hardware limits
The clinical experiment is observational and uses median burden groups (`>=median` high), OS_months, Kaplan-Meier and log-rank test. A smaller p-value is not itself proof of better predictive discrimination, and unique predictions are not experimentally validated neoantigens. IBM Pauli twirling/dynamical decoupling are device-specific; photonic shot evaluation cannot reproduce these mechanisms.[1]

The 46-qubit experiment requires resources outside this CPU-only environment. The clinical raw data are associated with controlled-access EGA/dbGaP resources; no login or approval is assumed. Derived publicly released assets may allow an artifact-level check but must not be called a newly trained clinical reproduction without appropriate provenance.

## MerLin mapping assessment inputs
Built-in `QCNNClassifier` is a square-image two-register amplitude-encoded model with adaptive state injection; reshaping peptide bits into a fictitious image would impose a different spatial structure. It is not a drop-in source-QCNN translation.[9]

A defensible adaptation should preserve independent heads, residue-local processing, staged pooling and the binder-only population, using at least two photons and declared classical readout costs. A global mesh-only binary classifier is at most an ablation, not a faithful hierarchical adaptation. Choosing angle encoding can avoid amplitude state preparation, but fewer photons/modes change the representational scale; state this explicitly. Full Fock probabilities retain bunching outcomes; collision-free UNBUNCHED semantics and success/normalization must be verified from installed MerLin implementation, not assumed to mean no postselection.[10]

## Unavailable materials and confidence
Update 2026-10-02: the authentic supplement PDF and DataS1 have been obtained, checksummed and inspected. The 8-page PDF includes the S1/S2 timings, figures S1–S5 and the experimental-assay legend; the original CSV of 8470 unique peptides includes the official splits and true assay labels. DATA_PROVENANCE.md records the 150/1000 binding and 300/1000 immunogenic split sizes, one ambiguous binding-test assay exclusion and no within-task overlap. The source signed scalar cross-entropy and gradient mismatch is verified in LOSS_SEMANTICS.md. Source-circuit/Qiskit equivalence and photonic numerical tests pass, with 63 project tests and Ruff. Eight one-update biological smoke candidates completed and the corrected v3 sweep logging validates. The main 100-update three-seed comparison is running; no substantive QCNN-reproduction verdict yet. Methodological confidence remains MEDIUM because historical configurations/labels and manuscript/released test-size differences are unresolved; empirical metric reproduction remains partial. Verdict: unresolved pending comparative results.

## Sources
[1] https://pmc.ncbi.nlm.nih.gov/articles/PMC13398535
[4] https://github.com/IBM/Q-CHIPP/tree/9f72bd9fc87522bc45c806f9849d07210a64e48d
[5] https://doi.org/10.5281/zenodo.19186890
[6] https://doi.org/10.5281/zenodo.19187502
[8] https://github.com/RyanPeters785/QCHIPP_figures/tree/qchipp_publication
[9] https://merlinquantum.ai/0.4/user_guide/models/qcnn.html
[10] https://merlinquantum.ai/0.4/user_guide/measurement_strategy.html
