# HQPINN

## Reproduced Paper

This implementation aims to reproduce the paper:

- **Hybrid Quantum Physics-informed Neural Network: Towards Efficient Learning of High-speed Flows**
- **Authors**: Fong Yew Leong, Wei-Bin Ewe, Si Bui Quang Tran, Zhongyuan Zhang, Jun Yong Khoo
- **arXiv**: `http://arxiv.org/abs/2503.02202`

The repository reproduces the main idea of the paper: comparing several hybrid quantum/classical PINN architectures across multiple physics problems, from a simple 1D benchmark to transonic flow around a NACA0012 airfoil.

### Methodology

#### Governing Equations

For the fluid dynamics case studies, the paper starts from the inviscid compressible Euler equations:

```text
∂t U + ∇ · G(U) = 0
```

with state vector

```text
U = (ρ, ρu, ρv, ρE)^T
```

where `ρ` is density, `(u, v)` are velocity components, `E` is total energy, and `p` is pressure. The equation of state is

```text
p = (γ - 1) (ρE - 1/2 ρ ||u||^2)
```

and the ideal-gas relation used in the paper is

```text
p = ρRT
```

with `γ = 1.4` for air. For the steady 2D transonic airfoil case, the governing equation simplifies to

```text
∇ · G(U) = 0
```

For the introductory `DHO` case study in Appendix A.2, the governing equation is the 1D damped harmonic oscillator ODE:

```text
m ∂tt u + μ ∂t u + k u = 0,   t ∈ (0, 1]
```

#### PINN Loss

Following the paper, the neural network approximation `U_NN(x, t)` is trained by minimizing a data term and a physics term. The residual is defined as

```text
F(x, t) = ∂t U_NN(x, t) + ∇ · G(U_NN(x, t))
```

and the total PINN objective is

```text
L = L_BC + L_F
```

where:

- `L_BC` is the mean-squared error on initial and boundary condition points
- `L_F` is the mean-squared error on collocation points enforcing the governing equations

In the paper notation:

```text
L_BC = (1 / N_B) Σ_j |U_NN(x_j^B, t_j^B) - U(x_j^B, t_j^B)|^2
L_F  = (1 / N_F) Σ_j |F(x_j^F, t_j^F)|^2
```

For `DHO`, the paper uses a problem-specific PINN loss combining the initial displacement, initial derivative, and ODE residual terms.

#### Parameterized Quantum Circuit

The quantum part of the method is based on a parameterized quantum circuit, or `PQC`. In the paper, the PQC is a variational quantum model `f(θ, φ)` where:

- `φ` encodes data or features
- `θ` contains trainable circuit parameters
- the output is an observable expectation value

This is written as:

```text
f(θ, φ) = <ψ(θ, φ)| O |ψ(θ, φ)>
```

The paper uses a data-reuploading structure that alternates feature-map layers `S(φ)` and trainable ansatz layers `A(θ)`. In the HQPINN setting, Pauli-Z expectations on the qubits provide the quantum branch outputs.

#### HQPINN

The main idea of the paper is the `HQPINN`, a hybrid quantum physics-informed neural network with two parallel branches:

- a quantum branch built from a PQC
- a classical branch built from a standard neural network

Both branches receive the same physical inputs, such as `(x, t)` in 1D Euler or `(x, y)` in the steady airfoil case. Their outputs are linearly fused to predict the physical state variables. In the paper, this is the hybrid alternative to:

- fully classical models
- fully quantum models

The objective is to check whether this hybrid design preserves the physics-informed training structure of PINNs while benefiting from expressive quantum features.

### DHO - Damped Harmonic Oscillator

This is the simplest case study in the repository: a 1D damped harmonic oscillator governed by an ordinary differential equation. In the paper, it is described in **Appendix A.2**. It serves as a lightweight benchmark to compare the different classical, hybrid, and quantum-only PINN architectures before moving to harder PDE problems.

### SEE - Smooth Euler Equation

This case study corresponds to the smooth 1D Euler setup used in Section 3.1 of the paper. It is a fluid dynamics benchmark with a smooth solution, used to evaluate how well the different architectures learn coupled physical quantities such as density, velocity, and pressure in a relatively well-behaved regime.

### DEE - Discontinuous Euler Equation

This case study corresponds to the discontinuous 1D Euler setup used in Section 3.2 of the paper. Compared with `SEE`, this problem is harder because the target solution includes a discontinuity, making it a more demanding test for PINN stability and approximation quality.

### TAF - Transonic Airfoil Flow

This is the most advanced case study in the paper and corresponds to the 2D transonic NACA0012 airfoil flow problem from Section 3.3. It is the closest setting to the paper's main application goal: learning high-speed flow fields with hybrid quantum physics-informed neural networks under realistic boundary and PDE constraints.

## Paper Experiments

The architecture variants implemented in the code are:

- `cc`: classical-classical
- `ci`: classical-interferometer
- `cp`: classical-PennyLane
- `ii`: interferometer-interferometer
- `pp`: PennyLane-PennyLane
- `cperc` and `percperc`: Perceval/Merlin variants specific to `DHO`

### Experimental Contexts

#### `DHO`

`DHO` is the introductory benchmark from **Appendix A.2**. It is a 1D damped harmonic oscillator used to test whether the hybrid classical-quantum design already helps on a simple PINN problem before moving to Euler flows.

Paper setup:

- ODE: `m ∂tt u + μ ∂t u + k u = 0`
- domain: `t ∈ (0, 1]`
- parameters: `m = 1`, `μ = 4`, `k = 400`
- optimization setup in the paper: Adam with learning rate `0.002` up to about `2000` epochs
- model context: 3-qubit / 3-layer PQC and a classical MLP with 2 hidden layers of width 16

#### `SEE`

`SEE` corresponds to **Section 3.1**, the smooth Euler equation in the harmonic regime. This is the well-behaved 1D Euler benchmark used to test the architectures when the target solution is smooth.

Paper setup:

- initial condition:

```text
U0 = (ρ0, u0, p0) = (1.0 + 0.2 sin(πx), 1.0, 1.0)
```

- exact traveling-wave solution:

```text
(ρ, u, p) = (1.0 + 0.2 sin(π(x - t)), 1.0, 1.0)
```

- domain: `x ∈ (-1, 1)`, `t ∈ (0, 2)`
- boundary conditions: periodic
- training samples in the paper: `N_ic = 50`, `N_bc = 50`, `N_F = 2000`

#### `DEE`

`DEE` corresponds to **Section 3.2**, the discontinuous Euler equation with a moving contact discontinuity. This is a harder 1D benchmark than `SEE`, because the target density field is no longer smooth.

Paper setup:

- left and right boundary states:

```text
(ρL, uL, pL) = (ρR, uR, pR) = (1.0, 0.1, 1.0)
```

- exact solution:

```text
ρ(x, t) = 1.4  if x < 0.5 + 0.1 t
ρ(x, t) = 1.0  if x > 0.5 + 0.1 t
u(x, t) = 0.1
p(x, t) = 1.0
```

- domain: `x ∈ (0, 1)`, `t ∈ (0, 2)`
- boundary conditions: Dirichlet
- training samples in the paper: `N_ic = 60`, `N_bc = 60`, `N_F = 1000`

#### `TAF`

`TAF` corresponds to **Section 3.3**, the 2D transonic airfoil flow case. This is the main high-speed-flow application in the paper and the most demanding benchmark in the repository.

Paper setup:

- governing equation: steady 2D Euler equation
- geometry: `NACA0012` airfoil with chord `(0, 1)`
- computational domain: `x ∈ (-1, 3.5)`, `y ∈ (-2.25, 2.25)`
- predicted variables: `(ρ, u, v, T)`
- inlet condition:

```text
U_in = (ρ_in, u_in, v_in, T_in) = (1.225, 272.15, 0.0, 288.15)
```

- outlet condition: `P_out = 0`
- side boundaries: periodic
- wall condition: free-slip on the airfoil surface
- paper training setup: 40 boundary points per boundary, 4000 domain points for physics loss, adaptive gradient weight, Adam for 40000 steps with learning rate `0.0005`, then L-BFGS for 2000 steps

## Reproduction Limitations

- The reproduction is **more advanced for `DHO`, `SEE`, and `DEE`** than for `TAF`.
- The scripts and configs cover more variants than the ones that have actually been rerun and archived in the summary CSV files.
- The standard batch script `HQPINN/run_all_train_jobs.sh` does not launch every experiment.
  It includes all `DHO` jobs.
  It launches `SEE`, `DEE`, and `TAF` in `cc`, `ci`, and `ii`.
  It **skips** `cp` variants outside `DHO`.
  It also **skips** `pp` variants for `SEE`, `DEE`, and `TAF`.
- Quantum variants are significantly more expensive to run than `cc`.
- Some summary CSV files contain duplicated rows when an existing checkpoint is reused and appended again to the summary.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r HQPINN/requirements.txt
```

## How To Run The Experiments

### General Interface

Interactive mode:

```bash
python3 -m HQPINN
```

Recommended config-based mode:

```bash
python3 -m HQPINN --config <path/to/config.json>
```

### Launch Examples

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

Run inference / plotting for an already trained model:

```bash
python3 -m HQPINN --config HQPINN/configs/see_cc_run_10-4.json
```

### Batch Training

To launch the standard training queue:

```bash
bash HQPINN/run_all_train_jobs.sh
```

To preview the queue without running it:

```bash
bash HQPINN/run_all_train_jobs.sh --dry-run
```

### `TAF` Case

The required `.npy` files for the NACA0012 case are already present in `HQPINN/TAF/NACA0012/`.

If needed, they can be regenerated with:

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
