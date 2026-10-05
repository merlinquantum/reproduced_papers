# MerLin adaptation: explicit design and scope

## Current primary architecture: seven-mode phase reference

The initial six-mode design below is retained as a **legacy ablation**, not the recommended adaptation. A physics test showed that encoding every mode makes a common phase shift globally unobservable at fixed photon number. In the actual source category code, A=1 and L=10 yield local QRAC angle vectors differing by a common pi/2 shift; the legacy model returned identical logits. Numeric input equality with a classical model is therefore not sufficient evidence of equal accessible information.

ReferencePhotonicQCNN uses seven modes, two photons and explicit state [1,0,0,1,0,0,0]. Modes 0–5 encode the six local features; mode 6 remains unencoded. The first universal mesh places the reference mode in the photon light cone, making input phases relative to this reference. Both stages retain full FOCK probabilities (28 outcomes), ideal PNR, no postselection and measured classical pooling. The alias regression test verifies nonzero probability separation. Measured parameters: 168 optical + 178 classical = 346; the matched shared-hierarchy MLP of width 17 has 350. No extra photon is introduced.

The corrected reference architecture, seeds 49/50/51, 100 full-batch Adam updates at 0.01 and analytic or 1000-shot-per-cell evaluation were declared before its benchmark. Legacy measurements remain unchanged. The architecture is still PARTIAL_MERLIN_TRANSLATION: coherent pooling and full source representational scale are not preserved. A reference mode fixes the demonstrated observability defect, not every possible expressivity or generalization limitation.

**Formal status: PARTIAL_MERLIN_TRANSLATION.** This design preserves independent binding/immunogenicity tasks and peptide-local, hierarchical processing, not the full coherent gate circuit. It is a hybrid measured-feature QCNN adaptation, not an optical implementation of the source unitaries or a quantum-advantage claim.

## Candidates examined

1. MerLin `models.QCNNClassifier`: rejected as the default. It encodes square images into separate row/column photon registers; peptides are ordered categorical sequences, not square images. Reshaping bits into an arbitrary image adds an unsupported spatial prior.
2. One global universal interferometer: retained only as a possible ablation. It removes locality and hierarchy, so it does not reproduce the QCNN scientific role.
3. A coherent many-photon/dual-rail transcription: potentially higher fidelity but requires entangling gate resources, ancillas/postselection and a pooling channel not represented by a passive mesh alone. MerLin QuantumBridge may support an expensive reference simulation, but that would not demonstrate a practical photonic classifier and is not selected for this first bounded reproduction.
4. **Selected: shared local photonic filters followed by measured bottleneck pooling.** Two neighboring residues are processed by a six-mode/two-photon cell with shared parameters; a three-feature classical projection compresses the measured probabilities. Four selected residues become two three-feature sites, then a second cell becomes one site. Final linear readout predicts a binary logit. This preserves local shared processing and staged dimensional reduction, but replaces coherent gate pooling with measure/project/re-encode operations.

## Optical cell

Use cookbook Pattern B with an explicit six-mode state `[1,0,0,1,0,0]`. The two photons start in different residue registers. A trainable mesh before encoding puts every encoding mode in the light cone; a second mesh mixes encoded inputs. Each input residue has a five-bit categorical code, mapped to three source QRAC angles (pad one zero bit within the residue). For each local pair, six angles are injected at scale 1. The second-stage inputs are three tanh-limited latent features per site, explicitly multiplied by pi; no silent phase wrapping.

Output uses **full FOCK probabilities**, ideal photon-number-resolving detection, no outcome postselection, ideal analytic MerLin CPU simulation. FOCK is selected over cookbook's default UNBUNCHED to retain collision events and avoid ambiguity about discarded bunching probability. A trainable `Linear(output_size,3)` with tanh performs the pooling bottleneck. This is a classical layer; all its parameters must be counted.

At least two photons are required in the reported adaptation. One-photon cells may be used only as labeled ablations. The state is validated against configured modes/photon number; no truncation or repair is allowed. Parameter counts and output dimension are to be obtained from the installed API, not assumed.

## Data alignment and comparisons

Both source QCNN and adaptation receive the same selected residues from the same verified train/test files. Binding: positions 2, 4, 5, 9. Binder-only immunogenicity: positions 3, 5, 6, 7. A positions 2, 5, 9 plus BA reference can be added if raw verified data support it; do not fill a missing biological residue with an arbitrary input.

MerLin-specific residue-local QRAC padding changes the source bit grouping: source concatenates five-bit residue encodings and groups them globally. Preserve source global grouping in the gate reproduction and disclose the change here. No trainable preprocessing on the test set. Exact source union-category encoding is a separate explicitly transductive reproduction mode.

Baselines: source gate QCNN forward verified against Qiskit; same hierarchy with classical shared local MLP cells and approximately matched total parameter budget; logistic regression and random forest controls on the same inputs. Include a frozen-photonic-core control if training budget allows. Same data, seed list, fixed thresholds and output metrics. Hyperparameter choices are prespecified rather than chosen by test performance. Report quantum, classical and total parameters separately and training/runtime differences.

## Training and finite shots

Autograd through the optical simulator with Adam is the default. The gate reference retains source SPSA when feasible; an autograd variant is a separately named deviation, not optimizer-equivalent. Record derivative methods and objective explicitly.

Train analytically. MerLin finite-shot calls are effective only in eval plus `torch.no_grad()`. The first-layer sampled measured features are passed to the second stage; every local-cell measurement needs its own shot budget. Report shots **per cell**, number of cells per peptide, and total shots per peptide. Finite-shot evaluation is shot noise only, not a reproduction of IBM device-noise mitigation. Partial measurement and amplitudes are simulation-only; this adaptation deliberately emits measured probabilities instead.

## Omitted scope and impact

No IBM warm-start hardware run, no full 46-qubit coherent circuit, no exhaustive 126-position search, no new clinical survival claim without authentic patient assets. These omissions materially limit claims about quantum benefit, large-sequence scaling and clinical utility. Remaining evidence can establish a runnable local photonic adaptation and comparative behavior on the two biological tasks; it cannot establish equivalence or advantage.

## Required evidence before a result claim

- Installed MerLin version/path and compatible Perceval import; explicit no-GPU probe.
- Dataset identity, release commits, hashes, duplicate/conflict checks, binder-only labels and official split provenance.
- Normalized nonnegative probability vectors, parameter gradients, changed weights after a step, tests of photon conservation and input-state validation.
- Per-run logs/config/status/metrics, predictions, checkpoints and data fingerprints.
- Measured parameters, seeds, epochs/optimizer steps, wall time and any budget differences.
- Analytic and finite-shot results with optical assumptions in machine-readable metrics and report.
