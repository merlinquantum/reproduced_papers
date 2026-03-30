# QRNN paper — Method extraction

This document summarizes the **method / model** described in the QRNN paper.
It is intentionally **paraphrased** to be implementation-oriented.

## 1. Quantum Recurrent Block (QRB)

A **quantum recurrent block (QRB)** is the quantum analogue of a classical recurrent block. At each discrete time step $t$, it:

- takes the current input element $x^{(t)}$,
- updates an internal quantum “history” state,
- outputs an intermediate prediction $y_t$.

The QRB uses two quantum registers:

- **Reg. D** (“data”): used to encode the current input element $x^{(t)}$.
- **Reg. H** (“history”): carries information from previous time steps and is forwarded to the next QRB.

Conceptually, the QRB is composed of:

1. **Data encoding** unitary $U_{in}(x^{(t)})$ acting on Reg. D.
2. **Ansatz** (a parameterized quantum circuit) $U_a(\theta)$ acting on Reg. D and Reg. H.
3. **Partial measurement**: measure part of Reg. D to obtain $y_t$, reset Reg. D, and keep Reg. H for the next time step.

## 2. Data encoding (feature map)

The paper focuses on **angle encoding** (chosen for NISQ practicality):

- Each scalar feature is rescaled to an angle $\tilde{x}_i \in [0,\pi]$.
- A single-qubit rotation (described as an $R_y$ encoding gate) embeds the rescaled value into a qubit.
- For an input vector with $N$ components, the basic scheme uses $N$ qubits.

The paper additionally mentions a **replicative embedding** strategy (a form of **input redundancy**), i.e. repeating/duplicating inputs across multiple qubits/time-encoding locations, and notes that such redundancy can improve accuracy.

## 3. Ansatz (hardware-efficient PQC)

The ansatz is a **hardware-efficient parameterized quantum circuit** built from alternating layers of:

- trainable **single-qubit rotations**, and
- entangling **two-qubit gates**.

### 3.1 Single-qubit gate parameterization

Single-qubit gates are represented using an $X$–$Z$ decomposition:

$$
U_{1q} = R_x(\alpha)\,R_z(\beta)\,R_x(\gamma),
$$

where $(\alpha,\beta,\gamma)$ are trainable parameters.

### 3.2 Two-qubit entangling gate

To increase expressibility/entanglement, the paper uses an $R_{zz}(\theta)$ gate:

$$
U_{2q} = R_{zz}(\theta) = \exp(i\theta\,Z_j Z_k),
$$

acting on qubits $j$ and $k$.

The paper also notes that $R_{zz}$ can be decomposed into native operations such as CNOT and single-qubit $Z$-rotations (see the figure referenced in the paper).

### 3.3 Connectivity / entangling pattern

Three connectivity patterns are discussed:

- nearest-neighbor (NN),
- circuit-block (CB),
- all-to-all (AA).

The paper motivates **CB** as a practical trade-off between expressibility and circuit cost.

## 4. Partial quantum measurement (prediction + state carry)

The QRB outputs a prediction $y_t$ while preserving the recurrent state.

Key points:

- Only **Reg. D** is measured (“partial measurement”).
- The paper measures **the first qubit of Reg. D** in the computational ($Z$) basis.
- The prediction is derived from the estimated probability $p(1)$ of that qubit collapsing to $\lvert 1\rangle$ (followed by a minor post-processing step).
- After obtaining the prediction, **all qubits of Reg. D** are measured and then **reinitialized** to $\lvert 0\rangle$.
- **Reg. H is not measured** and its quantum state is fed to the next QRB to carry history.

In simulation, the probability is computed via a **partial trace**:

- Starting from $\lvert 0\rangle$, apply encoding and ansatz to obtain a density matrix
  $\rho = U_a U_{in}\,\lvert 0\rangle\langle 0\rvert\,U_{in}^\dagger U_a^\dagger$.
- Reduce to the first qubit of Reg. D: $\rho^{(1)} = \mathrm{tr}_{\overline{1}}(\rho)$.
- Compute $p(1) = \mathrm{Tr}(\lvert 1\rangle\langle 1\rvert\,\rho^{(1)})$.

The paper also notes that measuring fewer qubits can reduce measurement error and can help with trainability (mitigating barren-plateau issues compared to global measurement).

## 5. Two QRNN architectures

### 5.1 pQRNN (plain QRNN)

The pQRNN is built by **stacking QRBs sequentially** over time steps (this is
exactly the “pipeline” shown in the pQRNN diagram / Fig. 7 in the paper):

At each step $t$:

1. **Initialize Reg. D** in $\lvert 0\rangle^{\otimes n_D}$ (Reg. H is *not*
  reset; it carries over from the previous step).
2. **Encode the current input** $x^{(t)}$ by applying $U_{in}(x^{(t)})$ on Reg.
  D.
3. **Apply the ansatz** $U_a(\theta)$ on **Reg. D ∪ Reg. H** to mix the new
  information with the recurrent “history” state.
4. **Partial measurement for the output**: measure **only the first qubit of
  Reg. D** (Pauli-$Z$ / computational basis), estimate $p(1)$, and map it (with
  minor post-processing) to the intermediate prediction $y_t$.
5. **Reset for the next step**: measure all qubits in Reg. D and reinitialize
  Reg. D to $\lvert 0\rangle^{\otimes n_D}$. Forward Reg. H unchanged to the
  next QRB.

In pQRNN, the assignment of physical qubits to **Reg. D** and **Reg. H** is fixed; qubits in Reg. H must remain coherent across the full sequence, implying stronger coherence-time requirements.

### 5.2 sQRNN (staggered QRNN)

The sQRNN arranges QRBs in a **staggered (shift-work) manner**:

- different physical qubits are **assigned to Reg. H in turn**,
- each qubit periodically gets a chance to be **reinitialized** to $\lvert 0\rangle$ after several time steps.

The paper’s motivation is to reduce coherence-time demands of the hardware while preserving a recurrent-history mechanism.

## 6. Parameter learning (hybrid optimization)

### 6.1 Rescaling the prediction to the label range

Measurement-derived predictions are first rescaled to the real-value range using input extrema:

$$
\tilde{y}_t = y_t\,(x_{\max}-x_{\min}) + x_{\min}.
$$

### 6.2 Loss

The paper uses an $L_2$ / MSE-style loss over $N$ samples (predictions vs. targets).

### 6.3 Gradient estimation

Because standard backpropagation does not directly apply, two gradient strategies are discussed:

- **Finite differences** (requires two circuit evaluations per parameter):
  $\partial L / \partial \theta_j \approx (L(\theta + \Delta e_j) - L(\theta - \Delta e_j)) / (2\Delta)$.
- **Analytical / parameter-shift rule** for suitable gates, by evaluating circuits at shifted parameters $\theta_j \pm \pi/2$.

The paper emphasizes that this is a **quantum–classical hybrid loop**:

- run circuits to obtain loss / expectations,
- compute gradient arithmetic and parameter updates classically.

### 6.4 Optimizers

The paper reports testing both **(vanilla) gradient descent** and **Adam**. It notes Adam can speed up training but may slightly reduce final accuracy.

## 7. Figure-dependent implementation details (may need manual inspection)

Some low-level circuit details are referenced primarily by figures:

- **Fig. 3**: exact “replicative” angle-encoding circuit layout.
- **Fig. 5 / Fig. 6**: the precise CB entangling schedule and full ansatz wiring for a given qubit count.
- **Fig. 8**: the exact staggering schedule / qubit-role rotation used in sQRNN.

