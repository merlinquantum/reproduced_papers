# Photonic Adaptation of QuaRK

Status: full MerLin implementation, simulated. It is our adaptation; the paper has none. Results: message 5 of the README; every table, including the setup of each run: [`results/photonic_comparison.md`](../results/photonic_comparison.md).

## What it keeps

QuaRK's results rest on four elements, and the photonic version keeps all of them:
- fixed random dynamics, never trained, driven by the window one step at a time;
- a partial reset at every step, which makes old inputs fade;
- features that are 1- and 2-local observables, estimated from one shot budget shared by all of them;
- the Matérn kernel ridge readout, with the same data, split, seeds, tuning and metrics.

## The mapping

| QuaRK (per sub-reservoir) | Photonic version (per sub-reservoir) |
|---|---|
| 5 qubits, starting in \|+⟩ | 3 photons in 14 modes, in modes 0, 5 and 9, then a fixed random interferometer (MZI mesh) |
| A fixed random Ising unitary after each step | The same fixed interferometer after each step; each sub-reservoir has its own |
| Input step x_t: rotations of angle π(1 − 10⁻⁶)·tanh, after a random projection 3 → 5 | Input step x_t: 14 phase shifters set to π(1 − 10⁻⁶)·tanh, after a random projection 3 → 14 |
| Partial reset to \|+⟩ with probability 1 − λ at each step | The same reset, reproduced exactly (below), with QuaRK's λ at the same seed |
| 105 Pauli expectations: 15 one-qubit, 90 two-qubit | 105 click probabilities: 14 one-mode, 91 two-mode |
| Shots: independent noise per observable (authors' code) or random-Pauli shadows (paper text) | Each shot records which of the 14 threshold detectors click; all 105 features come from the same shots |

**Why the reset is exact.** QuaRK's reset returns the state to its starting point. After 25 steps, its state is therefore a mixture: with probability (1 − λ)λʲ (j < 25) or λ²⁵ (j = 25), only the last j steps acted. Detection probabilities are linear in the state, so the photonic version mixes the outputs of the circuits that run only the last j steps, with the same weights. On hardware, a random draw of j before each shot does the same. A unit test checks the identity against QuaRK's own simulator to 10⁻¹².

**Detectors.** Threshold detectors report click or no click; two photons in one mode give one click. MerLin's default space for this case, `UNBUNCHED`, keeps only the outcomes without two photons in one mode and renormalizes them, which amounts to postselection. We simulate the full Fock space and read the clicks (`occupancy_readout=True`), so no shot is discarded.

**Interference control.** The same circuits with distinguishable photons (Perceval noise model, indistinguishability 0), which cannot interfere.

## Alternatives set aside

| Alternative | Why we set it aside |
|---|---|
| Translating QuaRK's qubit circuit gate by gate (dual-rail qubits) | Photonic entangling gates succeed only with some probability; 25 steps of couplings would almost never succeed, and the reset needs extra photons |
| A photonic feature map of the whole window at once | No step-by-step driving and no fading memory |
| A recurrent circuit with partial measurements between steps | The number of branches grows with every step, and photons must be stored between steps |
| Feeding measured outputs back as the next input | The memory becomes classical |
| A photonic fidelity kernel on the raw window | It replaces both the reservoir and the readout, so it is no longer QuaRK |

## Assumptions and cost

- Ideal components: perfect single-photon sources, perfect indistinguishability (except in the control), no loss, detectors without dark counts.
- Simulation only: MerLin's exact probabilities on a CPU, in float64; finite shots are drawn from them.
- Hardware cost: each shot needs the circuit reprogrammed to run the last j steps, up to 25 phase layers of 14 phases. At 1000 shots per window and sub-reservoir, the 10,000 windows need 3·10⁷ shots.
- 3 photons in 14 modes (469 click patterns) is classically simulable. This tests the reservoir's behaviour, not a regime beyond classical simulation.

## Confidence

High that the mapping keeps QuaRK's role: the reset identity is checked against QuaRK's simulator, and MerLin's probabilities against permanents computed from the same circuit, with and without interference (10⁻¹⁰). The generalization bound (Theorem 2) is not re-derived for click features.
