This README provides an overview of the **QTRL** (Quantum-Train Reinforcement Learning) project, implemented within the **Merlin/Quandela** reproduction framework. It focuses on training a hybrid quantum-classical agent to solve standard Gymnasium environments like **CartPole** and **MiniGrid**.

---

# QTRL: Quantum-Train Reinforcement Learning

This repository contains a reproduction of a hybrid reinforcement learning algorithm where a **photonic quantum circuit** (simulated via Perceval/Merlin) generates the weights for a classical policy network. This method leverages the high expressivity of quantum systems to parameterize classical models.

## ## Project Structure

The project follows the standard "reproduced papers" architecture required by the `runtime_lib` orchestrator:

```text
papers/QTRL/
├── cli.json                # CLI argument definitions and mapping
├── configs/
│   └── defaults.json       # Default hyperparameters
├── lib/
│   ├── runner.py           # Main entry point (train_and_evaluate)
│   ├── util.py             # Quantum circuits and RL training logic
└── README.md

```

## ## How it Works

1. **Quantum Weight Generation**: A photonic circuit (linear optics) is initialized. The output states of this circuit are mapped through a small classical neural network to produce the weights of a policy.
2. **Policy Execution**: These generated weights are used to parameterize a linear layer that maps environment observations to action probabilities.
3. **REINFORCE Algorithm**: The agent is trained using the Policy Gradient theorem. We maximize the expected return by minimizing the negative log-likelihood of actions weighted by the discounted returns.
4. **Multiple Backends**: The user can choose among 4 different models: fully classical, photonic (Merlin), gate-based (TorchQuantum), and hybrid photonic + MPS.

## ## Parameters

| Parameter          | Default       | Description |
|--------------------|---------------|-----------|
| `env_name`         | `CartPole`    | Target Gym environment (`CartPole` or `MiniGrid`) |
| `backend`          | `merlin_mlp`  | Hybrid backend (`merlin_mlp`, `merlin_mps`, `torchquantum`, `classic`) |
| `nb_photons`       | `2`           | Number of photons in the photonic circuit (≤ `nb_modes`) |
| `nb_modes`         | `3`           | Number of spatial modes |
| `q_output_size`    | `4`           | Output size of the quantum circuit (before mapping) |
| `hidden_sizes`     | `[8, 8]`      | Hidden layers of the classical mapping MLP |
| `layer_dim`        | -             | Layer architecture for the pure classical model |
| `bond_dim`         | `2`           | Bond dimension for MPS model |
| `n_qubit`          | `4`           | Number of qubits (TorchQuantum backend) |
| `q_depth`          | `2`           | Circuit depth / number of layers (TorchQuantum) |
| `lr`               | `0.01`        | Learning rate for Adam optimizer |
| `num_episodes`     | `500`         | Number of training episodes |
| `final_output_size`| `8`           | Total number of weights to generate (`state_dim × action_dim`) |
| `run_mode`         | `train`       | Script execution mode (`train` or `gridsearch`) |

## ## Installation & Usage

### 1. Prerequisites

Ensure you are in a virtual environment with the necessary dependencies installed:

* `perceval-quandela` / `merlin`
* `gymnasium`
* `torch`
* `minigrid` (for MiniGrid tasks)

You will also need to install TorchQuantum to enable the gate-based backend.
**Installation steps**:

Go to the official TorchQuantum repository: `https://github.com/mit-han-lab/torchquantum`
Download or clone the repository.
Place the **torchquantum** folder inside the QTRL directory.
Navigate into the **torchquantum** folder and run the following command: `pip3 install -e .`

This will install TorchQuantum in editable mode so it can be properly imported by Python.

### 2. Running an Experiment

Launch the experiment from the repository root using the global implementation script:

**Run with default parameters (CartPole):**

```bash
python implementation.py --paper QTRL

```

**Run on MiniGrid**

```bash
python implementation.py --paper QTRL --env_name MiniGrid

```

**List available options:**

```bash
python implementation.py --paper QTRL --help

```

## ## Implementation Details

* **Optimization**: The optimizer updates both the quantum circuit parameters (interferometer phases) and the classical mapping weights simultaneously.
* **Discounted Returns**: Returns are computed recursively ($G_t = r_t + \gamma G_{t+1}$) and standardized using Z-score normalization to stabilize training.
* **Gradient Clipping**: Applied to prevent exploding gradients, which are common in hybrid quantum-classical models.
* **Model Factory**: All models are instantiated through the `create_hybrid_model` interface for better modularity.
---

> **Note:** This project is part of a research reproduction effort. Results may vary based on the photonic backend and noise models used.