# How the swap-test kernel is computed

This note details the "Simulation setup" deviation of the README: why drawing the shots from a binomial law gives the kernel of the authors' circuits, how we check it, and how the MerLin adaptation computes the same kernel.

## 1. The kernel and its finite-shot estimate

Each sample $x \in \mathbb{R}^d$ is amplitude-encoded: zero-padded to $2^n$ entries, permuted by a fixed permutation shared by both parties, and normalised, $|\psi(x)\rangle = \sum_i \frac{x_i}{\lVert x \rVert} |i\rangle$. The permutation does not change any overlap.

For two states $|a\rangle = |\psi(x)\rangle$ and $|b\rangle = |\psi(y)\rangle$, the swap test reads 0 with probability (paper, App. A)

$$p_0 = \tfrac12 + \tfrac12\, |\langle a | b \rangle|^2 .$$

With $M = 1{,}024$ shots and $Z$ zeros, the authors' estimate (`library.py`, `teleport_and_inner_product`) is

$$\hat K = \max\!\left(0,\ \tfrac{2Z}{M} - 1\right).$$

Each shot is a Bernoulli trial: the ancilla reads 0 with probability $p_0$ (the paper's App. B uses the same model). Over $M$ independent shots, the number of zeros $Z$ is therefore $\mathrm{Binomial}(M, p_0)$, and the shortcut draws $Z$ directly from this law instead of simulating the circuit $M$ times. Before the clipping at 0,

$$\mathbb{E}\!\left[\tfrac{2Z}{M} - 1\right] = 2p_0 - 1 = |\langle a | b \rangle|^2, \qquad \mathrm{std}\!\left[\tfrac{2Z}{M} - 1\right] = 2\sqrt{\tfrac{p_0(1-p_0)}{M}} \le \tfrac{1}{\sqrt M} \approx 0.031 .$$

The kernel is therefore the **squared** overlap, i.e. the squared cosine similarity of the standardised features, $\left(\frac{x \cdot y}{\lVert x\rVert \lVert y\rVert}\right)^2$. The paper's Eq. 11 takes the square root of $2\Pr(0) - 1$; the authors' code does not, and neither do we. The clipping at 0 slightly raises entries whose overlap is close to 0 (by at most $\approx 0.031/\sqrt{2\pi} \approx 0.012$ when the overlap is 0); our shortcut uses the same estimator, so it has the same bias.

## 2. The shortcut

The authors run every kernel entry shot by shot on Qiskit Aer (one 1,024-shot run of the distributed circuit per entry). We compute $p_0$ exactly and draw $Z \sim \mathrm{Binomial}(M, p_0)$ directly (`lib/kernels.py`, `estimate_from_zero_probability`; one draw per unordered pair, mirrored, as the authors' `compute_partial_kernel` does).

Without noise this is exact, not an approximation:

- teleportation reproduces each party's state exactly on the server (the authors' extra Hadamard before each teleportation applies $H^{\otimes n}$ to both states, which leaves their overlap unchanged);
- the swap-test outcome then depends only on the two states, so the distributed and centralised circuits give the same $p_0$.

With noise (Fig. 3 only), $p_0$ is computed instead from the noise channel of the transpiled circuit (`lib/noise.py`).

## 3. The check against per-shot Aer runs

`lib/protocol.py` ports the authors' circuits to Qiskit 2.x gate for gate (same registers and gate order, `if_test` instead of `c_if`). For 30 random state pairs (20 with 2 qubits per party, 10 with 4), each circuit is run on Aer with 1,024 shots, and the fraction of zeros is compared with $p_0$:

![Per-shot Aer runs vs the swap-test law](../results/protocol_check.png)

Every point lies on the diagonal; as expected for ±2 standard deviations, nearly all lie inside the grey band. A pooled χ² test over the 30 pairs agrees (p = 0.14 distributed, 0.26 centralised), while it rejects the wrong laws "$p_0 = \frac12$" (ancilla at random) and "$p_0 = \frac12 + \frac12|\langle a|b\rangle|$" (no square) with p < 10⁻²⁰⁰ (`utils/pooled_protocol_check.py` prints all three).

To reproduce (from the repository root, about 3 minutes; Aer is seeded, so the counts are identical):

```bash
python implementation.py --paper distributed_secure_kernels --config configs/protocol_equivalence_original.json --seed 0
python implementation.py --paper distributed_secure_kernels --config configs/protocol_equivalence_four_qubits.json --seed 0
python papers/distributed_secure_kernels/utils/plot_protocol_check.py
python papers/distributed_secure_kernels/utils/pooled_protocol_check.py
```

`tests/test_protocol.py` runs a smaller version (6 pairs), including the paper's Fig. 2 circuit built as drawn (6n + 1 qubits).

## 4. The MerLin adaptation: the same kernel with photons

Only the server's swap test is translated; teleportation has no deterministic linear-optics counterpart (see the README).

- **Encoding.** Each sample is amplitude-encoded on the collision-free Fock states of $m$ modes holding 2 photons: $\binom{m}{2} \ge d$ basis states, amplitudes $x_i / \lVert x \rVert$, zero-padded (`lib/photonic_merlin.py`, `encode_registers`). Wine and Framingham use $m = 6$, Parkinson's $m = 8$, Digits $m = 12$. The amplitudes are those of the qubit encoding, so the overlap $\langle a | b \rangle$ is the same.
- **Circuit.** The two registers enter one MerLin `QuantumLayer` on $2m$ modes with 4 photons, in the FOCK computation space. A permutation interleaves them (mode $i$ of each register side by side), then one 50:50 beam splitter mixes each pair of corresponding modes: register-wise Hong–Ou–Mandel interference.
- **Readout.** Photon-number-resolving detection gives the parity of the number of photons in the odd output ports, with $P(\text{even}) = \tfrac12 + \tfrac12 |\langle a | b \rangle|^2$ (Garcia-Escartin and Chamorro-Posada, 2013), the swap-test law. Threshold detectors cannot read this parity.
- **Estimate.** $P(\text{even})$ takes the place of $p_0$, followed by the same binomial draw and estimator as above. Every run checks $P(\text{even})$ against the swap-test law and fails above a tolerance; the largest deviation logged over all runs is $2.6 \times 10^{-8}$.
- **Assumptions.** Ideal sources, exact state preparation (not deterministic with linear optics in general), no loss, ideal photon-number-resolving detectors; simulation only.

**Why not `merlin.FidelityKernel`.** It computes $|\langle s | U^\dagger(x_2)\, U(x_1) | s \rangle|^2$ for a fixed input Fock state $|s\rangle$ and a circuit $U(x)$ whose phases encode the data (angle encoding). That is a different kernel: the data enter as phases, not as amplitudes. It also applies both data points in one circuit, whereas the paper's swap test compares two states prepared separately by two parties. For amplitude-encoded states, MerLin's own guidance is to build a `StateVector` and use a `QuantumLayer`, which is what this adaptation does.
