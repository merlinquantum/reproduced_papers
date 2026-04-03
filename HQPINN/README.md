# HQPINN

## Reproduced Paper

This folder reproduces:

- **Hybrid Quantum Physics-informed Neural Network: Towards Efficient Learning of High-speed Flows**
- **Authors**: Fong Yew Leong, Wei-Bin Ewe, Si Bui Quang Tran, Zhongyuan Zhang, Jun Yong Khoo
- **arXiv**: `http://arxiv.org/abs/2503.02202`

The paper compares hybrid quantum/classical PINNs across four case studies, from a 1D benchmark to 2D transonic flow around a NACA0012 airfoil.

### Methodology

#### Governing Equations

For the fluid-dynamics case studies, the governing equations are the inviscid compressible Euler equations:

$$
\partial_t U + \nabla \cdot G(U) = 0
$$

with state vector

$$
U = (\rho, \rho u, \rho v, \rho E)^T
$$

where $\rho$ is density, $(u, v)$ are velocity components, $E$ is total energy, and $p$ is pressure. The equation of state is

$$
p = (\gamma - 1)\left(\rho E - \frac{1}{2}\rho \lVert u \rVert^2\right)
$$

with $\gamma = 1.4$ for air, together with the ideal-gas relation

$$
p = \rho R T
$$

For the steady 2D airfoil case, this becomes

$$
\nabla \cdot G(U) = 0
$$

For `DHO` (Appendix A.2), the governing equation is

$$
m \partial_{tt} u + \mu \partial_t u + k u = 0, \qquad t \in (0, 1]
$$

#### PINN Loss

Section 2.2 defines a data loss and a physics loss. For the unsteady Euler formulation, the residual is

$$
F(x, t) = \partial_t U_{NN}(x, t) + \nabla \cdot G(U_{NN}(x, t))
$$

This is not a prediction-minus-ground-truth error: it is the governing equation evaluated on the network output, so the target is zero when the prediction satisfies the physics. The total objective is

$$
\mathcal{L} = \mathcal{L}_{BC} + \mathcal{L}_F
$$

- $\mathcal{L}_{BC}$ is the data loss, including initial and boundary condition points
- $\mathcal{L}_F$ is the physics loss, computed from collocation points through the residual

In the paper notation:

$$
\mathcal{L}_{BC} = \frac{1}{N_B}\sum_j \left| U_{NN}(x_j^B, t_j^B) - U(x_j^B, t_j^B) \right|^2
$$

$$
\mathcal{L}_F = \frac{1}{N_F}\sum_j \left| F(x_j^F, t_j^F) \right|^2
$$

`DHO` uses a separate problem-specific loss with an initial displacement term, an initial derivative term, and an ODE residual term.

#### Parameterized Quantum Circuit

The quantum part is a parameterized quantum circuit (`PQC`), written as

$$
f(\theta, \varphi) = \langle \psi(\theta, \varphi) \vert O \vert \psi(\theta, \varphi) \rangle
$$

The circuit alternates feature-map layers $S(\varphi)$ and trainable ansatz layers $A(\theta)$. In HQPINN, Pauli-$Z$ expectations provide the quantum branch outputs.

#### HQPINN

`HQPINN` uses two parallel branches:

- a quantum branch built from a PQC
- a classical branch built from a neural network

Their outputs are linearly fused to predict the physical state variables. The paper compares this hybrid model to classical-classical and quantum-quantum baselines.

### DHO - Damped Harmonic Oscillator

`DHO` is the introductory benchmark from **Appendix A.2**.

Main equations:

$$
m \partial_{tt} u + \mu \partial_t u + k u = 0, \qquad t \in (0, 1]
$$

with

$$
m = 1, \qquad \mu = 4, \qquad k = 400
$$

### SEE - Smooth Euler Equation

`SEE` corresponds to **Section 3.1**.

Main equations:

$$
\partial_t U + \partial_x F(U) = 0
$$

with

$$
U = (\rho, u, p)
$$

initial condition

$$
U_0 = (\rho_0, u_0, p_0) = (1.0 + 0.2 \sin(\pi x), 1.0, 1.0)
$$

and traveling-wave solution

$$
(\rho, u, p) = (1.0 + 0.2 \sin(\pi (x - t)), 1.0, 1.0)
$$

### DEE - Discontinuous Euler Equation

`DEE` corresponds to **Section 3.2**.

Main equations:

$$
\partial_t U + \partial_x F(U) = 0
$$

with

$$
U = (\rho, u, p)
$$

boundary states

$$
(\rho_L, u_L, p_L) = (\rho_R, u_R, p_R) = (1.0, 0.1, 1.0)
$$

and exact solution

$$
\rho(x, t) =
\begin{cases}
1.4, & x < 0.5 + 0.1 t \\
1.0, & x > 0.5 + 0.1 t
\end{cases}
$$

$$
u(x, t) = 0.1, \qquad p(x, t) = 1.0
$$

### TAF - Transonic Airfoil Flow

`TAF` corresponds to **Section 3.3**.

Main equations:

$$
\nabla \cdot G(U) = 0
$$

with

$$
U = (\rho, u, v, T)
$$

computational domain

$$
x \in (-1, 3.5), \qquad y \in (-2.25, 2.25)
$$

and inlet condition

$$
U_{in} = (\rho_{in}, u_{in}, v_{in}, T_{in}) = (1.225, 272.15, 0.0, 288.15)
$$

## Experiments

Implemented architecture variants:

- `cc`: classical-classical
- `hy-pl`: hybrid PennyLane
- `hy-m`: hybrid Merlin
- `hy-mp`: hybrid Merlin-Perceval
- `qq-pl`: quantum-quantum PennyLane
- `qq-m`: quantum-quantum Merlin
- `qq-mp`: quantum-quantum Merlin-Perceval (`DHO` only)

### Contexts

#### `DHO`

`DHO` comes from **Appendix A.2**.

Setup:

- ODE: $m \partial_{tt} u + \mu \partial_t u + k u = 0$
- domain: $t \in (0, 1]$
- parameters: $m = 1$, $\mu = 4$, $k = 400$
- optimization: Adam, learning rate `0.002`, about `2000` epochs
- model: 3-qubit / 3-layer PQC and a classical MLP with 2 hidden layers of width 16

#### `SEE`

`SEE` corresponds to **Section 3.1**.

Setup:

- initial condition:

$$
U_0 = (\rho_0, u_0, p_0) = (1.0 + 0.2 \sin(\pi x), 1.0, 1.0)
$$

- traveling-wave solution:

$$
(\rho, u, p) = (1.0 + 0.2 \sin(\pi (x - t)), 1.0, 1.0)
$$

- domain: $x \in (-1, 1)$, $t \in (0, 2)$
- boundary conditions: periodic
- training samples: $N_{ic} = 50$, $N_{bc} = 50$, $N_F = 2000$

#### `DEE`

`DEE` corresponds to **Section 3.2**.

Setup:

- boundary states:

$$
(\rho_L, u_L, p_L) = (\rho_R, u_R, p_R) = (1.0, 0.1, 1.0)
$$

- exact solution:

$$
\rho(x, t) =
\begin{cases}
1.4, & x < 0.5 + 0.1 t \\
1.0, & x > 0.5 + 0.1 t
\end{cases}
$$

$$
u(x, t) = 0.1, \qquad p(x, t) = 1.0
$$

- domain: $x \in (0, 1)$, $t \in (0, 2)$
- boundary conditions: Dirichlet
- training samples: $N_{ic} = 60$, $N_{bc} = 60$, $N_F = 1000$

#### `TAF`

`TAF` corresponds to **Section 3.3**.

Setup:

- governing equation: steady 2D Euler equation
- geometry: `NACA0012` airfoil with chord $(0, 1)$
- computational domain: $x \in (-1, 3.5)$, $y \in (-2.25, 2.25)$
- predicted variables: $(\rho, u, v, T)$
- inlet condition:

$$
U_{in} = (\rho_{in}, u_{in}, v_{in}, T_{in}) = (1.225, 272.15, 0.0, 288.15)
$$

- outlet condition: $P_{out} = 0$
- side boundaries: periodic
- wall condition: free-slip on the airfoil surface
- training: 40 boundary points per boundary, 4000 domain points for physics loss, adaptive gradient weight, Adam for 40000 steps with learning rate `0.0005`, then L-BFGS for 2000 steps

## Reproduction Limitations

This reproduction reflects practical CPU constraints.

- `SEE`, `DEE`, and `TAF` use mini-batched training. The main reproduced settings use `n_f_batch = 256`, and `TAF` also uses `n_wall_batch = 128` before a final full-batch L-BFGS refinement.
- The two branches are combined through a learned linear fusion layer to keep the readout compact and parameter budgets comparable across baselines.
- PennyLane variants outside `DHO` were not rerun in the consolidated reproduction because their CPU cost would be prohibitively high without a sizeable compute cluster. This is why [`HQPINN/run_all_train_jobs.sh`](/Users/jerome/git/reproduced_papers_fork/HQPINN/run_all_train_jobs.sh) focuses on `DHO` plus the `cc`, `hy-m`, and `qq-m` families for `SEE`, `DEE`, and `TAF`.
- For `DHO`, we also tested Merlin-Perceval variants (`dho-hy-mp` and `dho-qq-mp`) in addition to the generic Merlin interferometer approach.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r HQPINN/requirements.txt
```

## How To Run The Experiments

### Interface

Interactive mode:

```bash
python3 -m HQPINN
```

Config-based mode:

```bash
python3 -m HQPINN --config <path/to/config.json>
```

### Examples

Train a `DHO` case:

```bash
python3 -m HQPINN --config HQPINN/configs/dho_cc_train.json
```

Train a `SEE` case:

```bash
python3 -m HQPINN --config HQPINN/configs/see_ci_train_10-4-2.json
```

Train a `DEE` case:

```bash
python3 -m HQPINN --config HQPINN/configs/dee_ii_train_1.json
```

Train a `TAF` case:

```bash
python3 -m HQPINN --config HQPINN/configs/taf_cc_train_40-4.json
```

Run inference / plotting for a trained model:

```bash
python3 -m HQPINN --config HQPINN/configs/see_cc_run_10-4.json
```

### Batch Training

Launch the standard training queue:

```bash
bash HQPINN/run_all_train_jobs.sh
```

Preview the queue:

```bash
bash HQPINN/run_all_train_jobs.sh --dry-run
```

### `TAF` Case

The `.npy` files for the NACA0012 case are already present in `HQPINN/TAF/NACA0012/`. To regenerate them:

```bash
python3 -m HQPINN.TAF.generate_aerofoil_training_sets
```

## Where To Look At Results

- `DHO`: `HQPINN/DHO/results/dho_summary.csv`
- `SEE`: `HQPINN/SEE/results/see_summary.csv`
- `DEE`: `HQPINN/DEE/results/dee_summary.csv`
- `TAF`: `HQPINN/TAF/results/`

Checkpoints are saved in:

- `HQPINN/DHO/models/`
- `HQPINN/SEE/models/`
- `HQPINN/DEE/models/`
- `HQPINN/TAF/models/`

## Quick Structure

- `HQPINN/DHO/`: damped harmonic oscillator
- `HQPINN/SEE/`: smooth 1D Euler
- `HQPINN/DEE/`: discontinuous 1D Euler
- `HQPINN/TAF/`: 2D transonic flow around a NACA0012 airfoil
- `HQPINN/configs/`: ready-to-run configs
- `HQPINN/run_all_train_jobs.sh`: standard training batch
