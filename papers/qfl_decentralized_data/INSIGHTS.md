# Insights — Quantum Federated Learning With Decentralized Data (Huang, Tan, Xu 2022)

Durable notes for future reproductions of federated or distributed QML, small variational classifiers on downsampled images, and photonic adaptations of such models in MerLin.

## Reading the paper's claims

- **Differences of 0.01 to 0.15 points are not evidence.** The paper's bold "best" values (Tables II and III) separate federated from non-federated training by less than the seed noise of any model tested here (0.2 to 2.6 points over 5 seeds). Treat "QFL improves accuracy" as untestable at this resolution; "comparable" is the defensible reading.
- **Federation is cheap for small models on these tasks.** Three different quantum models (hybrid gate, faithful gate, photonic) are federation-invariant within noise, and a strongly non-i.i.d. Dirichlet(0.5) split costs nothing measurable against an i.i.d. split. A classical MLP loses slightly more from federation (up to about 1 point at 16 clients) than the quantum models do, but stays well ahead in absolute accuracy.
- **"20 rounds suffice" is not quantum-specific.** Quantum models reach 99 % of their final accuracy in 2 to 9 rounds, a classical MLP in 1 to 3.

## The underspecified circuit is the whole problem

- **Read-out decides accuracy.** For a tree tensor network read out on a single qubit, the read-out is the bottleneck. Depth beyond 2 layers, learning rate, initial scale, encoding range, feature scaling and training length each moved validation accuracy by at most 3.5 points, none by more than its seed spread. A linear layer on all eight single-qubit <Z> values moved it by +6.7 points and cut seed variance from about 5 to about 1.5 points.
- **Encoding scale matters in one direction.** Halving the angle range (0.5 pi per pixel) cost 11 points; doubling it changed little. 4x4 block-mean pixels occupy roughly [0, 0.6], so a full pi range is needed to use the qubit's polar angle. The photonic model shows the same pattern: phase = pi x pixel was best for every placement and photon number.
- **Measure the data ceiling before setting targets.** 4x4 area-downsampled MNIST caps a logistic regression at 95.6 % (0 vs 9) and 97.4 % (0 vs 1). A known-answer threshold of 98 % on 0 vs 1 is unattainable for any model on these features. Fit a trivial classical model on the exact preprocessed inputs first.
- **Seed instability marks a weak read-out.** With the root-qubit read-out, about one seed in three stalls near 80 %. Unstable seeds are a symptom worth diagnosing before averaging them away.

## Federated implementation details

- **Average parameters, not the whole `state_dict`.** MerLin layers (and, in general, models with constant buffers) carry float buffers. Weighted averaging with weights like 7/31 + 11/31 + 13/31 is not exactly 1 in floating point, so a constant buffer drifts. Average `named_parameters()` only and keep buffers.
- **Use a fresh optimiser per client per round, and one epoch over local data.** This reproduces the paper's B = 20, E = 1 and keeps the round axis comparable to non-federated epochs.
- **Seed the partition separately from the training seed.** A dedicated data seed keeps every model on identical client shards, which is what makes cross-model comparisons cell by cell possible.

## Photonic (MerLin) translation lessons

- **Re-uploading maps "two pixels per qubit" naturally.** Encode even pixels as phases in one stage and odd pixels in a second stage, around trainable meshes. Input columns feed encoding stages in builder order.
- **A fixed superposition layer in front of a universal trainable mesh is absorbed.** It cannot change expressivity, only the starting point, and it gave no benefit. Used instead of the first mesh it spreads photons before encoding with fewer parameters (112 against 168 phases) and was the best and most stable model.
- **Two photons were enough and more stable than four** at 8 modes for this task (96.9 % against 95.9 % validation). More photons do not automatically buy accuracy.
- **`UNBUNCHED` click probabilities carry multi-photon interference.** `FOCK` mean photon numbers do not: they reduce to single-photon interference. Prefer click probabilities or outcome distributions as read-out features.
- **A fully photonic read-out is competitive here.** A fixed `LexGrouping` of the 28 two-photon outcomes, with no classical parameter, reaches the paper's 0 vs 9 accuracy; the linear head adds 1.7 to 3.0 points.
- **Silent MerLin pitfall:** an encoding stage name ending in a digit (`x0`) makes `add_angle_encoding` ignore `scale` (confirmed on 0.4.1). Use letter-only names (`xa`, `xb`).
- **Report what the API cannot give.** `mode_expectations(UNBUNCHED)` hides the post-selection acceptance probability, which is part of the hardware cost.

## Fair comparison

- **The paper's own baseline is the right control for its claims,** because they concern federated versus non-federated training, not quantum versus classical. A matched classical model is still needed to read any accuracy as meaningful.
- **A tiny classical MLP wins.** With 59 parameters it beats a 60-parameter hybrid VQTN by 1 to 5 points and a 130-parameter photonic model by 0.7 to 1.7 points on every cell, with an untuned learning rate. When the smaller, untuned baseline already wins, "no advantage" holds without a size-matched rerun.
- **Simulation cost is not an algorithmic property.** About 100 s per quantum run against about 5 s per classical run on a CPU reflects simulating 8 qubits or modes, not training difficulty.

## Compute

- **Small circuits run faster on CPU.** 8-qubit statevector or 8-mode, 2-photon simulations at batch 20 ran 3.3 times slower on an NVIDIA L4 than on a laptop CPU, because per-gate launch overhead dominates. Time a smoke run before planning GPU sweeps.
- **Pin `merlinquantum>=0.4` in `requirements.txt`.** An unpinned requirement resolved to 0.3.1 next to Perceval 1.3.1 on a fresh environment and failed at import.

## Open directions

- **The 3 vs 6 and 4-class gaps** (3 and 8 points below the paper) were not diagnosed. The paper's circuit may use a different read-out or depth for them.
- **Photonic hardware realism:** acceptance probability, photon loss and distinguishability would decide whether the photonic advantage over the gate model survives on hardware.
- **Privacy:** the paper argues privacy by construction. Gradient or parameter-inversion attacks on the shared phases are untested here and in the paper.
- **QAOA and VQE federation** (code and tests exist in this folder, runs descoped).
