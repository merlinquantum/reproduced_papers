# Insights — Vertical Quantum Federated Learning (Ballester et al., ICEA '25)

Durable notes for future reproductions of vertical or distributed QML, and for photonic adaptations of multi-party models in MerLin.

## Reading the paper's claims

- **The headline number is a test-selected peak.** ">94%" for 8 clients holds only as the peak of the authors' mean test-accuracy curve (94.04% at epoch 27). Their final-epoch mean is 93.0%, and ours is 92.85%. Before chasing a gap, check which statistic a headline value uses: final epoch, best epoch, or peak of the mean.
- **The paper's ablations are not quantum effects.** The paired pattern is 8 clients ≫ 1 client, distinct features > duplicated features, and half coverage only slightly worse. A parameter-matched classical vertical-FL model reproduces it: E1 − E2 is +18 points for classical against +21 for quantum. Distributed-QML claims about feature coverage should always be checked against a classical model with the same partitions.
- **No accuracy benefit at matched size.** A classical model with matched parameters reaches 93.6%, against 92.9% for the 72,000-parameter quantum model. That is no clear difference, but classical is ahead in all layouts. Vertical FL on Semeion is easy for small MLPs; even a 2,700-parameter centralised MLP reaches 91.5%.

## Reproduction techniques that paid off

- **Validate against released weights before any long run.** Loading the authors' trained parameters into the re-implemented circuit reproduced their recorded test accuracy exactly. That one check settles the circuit layout, gate conventions, readout and data pipeline at once.
- **Match the split exactly.** The authors' `data_tensors.pt` let us test that the stratified split and permutation were identical, bit for bit.
- **Know when to abandon a dependency.** TorchQuantum 0.1.8 no longer installs cleanly, because of conflicting qiskit pins. A minimal batched statevector simulator was the faster route: about 200 lines covering RX, RY, a CZ ring, and ⟨Z⟩/⟨X⟩. Splitting the state into two 4-qubit halves and applying layers as `A @ state @ B.T` made 8-qubit, 500-layer circuits cheap on CPU.
- **Check the loss behind the paper's "logits".** The released code applies softmax before cross-entropy. A control without it gave identical accuracy at E1, so it is a harmless deviation, but it changes the reported loss values.

## Photonic (MerLin) translation lessons

- **Readout is the first bottleneck.** A fixed `LexGrouping` of the server's output distribution was the largest single limitation, worth 5.3 points of validation accuracy averaged over the other factors. A small trainable head lifted it, as `papers/distributed_qml_cc` had already found. Check the readout before enlarging the circuit.
- **Don't copy the gate model's training budget.** Photonic models trained with Adam (lr 0.01) were still improving at epoch 30; 100 epochs gave +4.8 points.
- **`UNBUNCHED` post-selects.** It keeps only collision-free outcomes and renormalises them; it is not threshold detection. Report the discarded probability as post-selection cost. A one-line Hong-Ou-Mandel probe shows the semantics: input |1,1⟩, a 50:50 splitter, a phase, a second 50:50 splitter.
- **`FOCK` mean photon numbers lose multi-photon interference.** In `FOCK`, ⟨n_k⟩ depends only on |U|². With one photon per mode it is constant, so a client sending `mode_expectations` with input `[1]*8` transmits nothing. Use click probabilities P(n_k ≥ 1) or photon-number marginals when the message should carry multi-photon interference.
- **More photons did not help here.** With click-probability messages, 8 photons reached about 82% validation accuracy. That is the same level as the best 4-photon design (82.0%), but with much larger variance (std 6.4 against 1.1 points). Re-uploading the data twice hurt. Expressivity growth with photon number (as in Gan et al.) did not turn into accuracy on this task.
- **Encoding scale matters, and a MerLin quirk can hide it.**
  - Phases of about π/8 per binary pixel worked best; π/2 and π collapsed training to chance.
  - Keep encoding names free of trailing digits: `add_angle_encoding(name="x0")` silently ignores `scale`. Use letter-only stage names such as `xa`, `xb`.
- **Size the classical head to the marginals.** A trainable head on the full Fock distribution grows with C(m+n−1, n): 6,435 outcomes for 8 photons in 8 modes means 64,000 classical parameters. A head on per-mode photon-number marginals (m·(n+1) inputs) keeps the classical part to a few hundred parameters.
- **Expect noisy final-epoch values.** Photonic training curves at lr 0.01 oscillate by about ±5 points between epochs. Consider averaging the last few epochs, or using a validation-selected checkpoint, when designing protocols.

## Fair comparison design

- **One matched size per compared family.** A 72,000-parameter gate model and roughly 3,000-parameter photonic models need separate parameter-matched baselines. Let the epoch budget follow the compared system.
- **Classical baselines are nearly free.** Each classical run took 1-4 s, against 16-27 min per simulated quantum or photonic run on the same CPU. That makes 3 training seeds per distribution and a centralised reference affordable. The time gap is simulation cost, not an algorithmic property.
- **Tuning asymmetry should be explicit.** The paper's quantum model is untuned, while the photonic and classical models had validation-based learning-rate (and scale) selection. State it in every comparison.

## Open directions

- **Vertical splitting on larger-feature data.** On larger inputs, each client's slice is too large for one small chip, and the partition actually matters for resources.
- **Privacy is untested.** The privacy claim needs an attack model (message or gradient inversion). Neither the paper nor this reproduction tests it.
- **Noise-aware photonic runs.** With Perceval noise models, these would test the NISQ-suitability argument, which the paper makes without evidence.
