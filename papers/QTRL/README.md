This README provides an overview of the **QTRL** (Quantum-Train Reinforcement Learning) project, implemented within the **Merlin/Quandela** reproduction framework. It focuses on training a hybrid quantum-classical agent to solve standard Gymnasium environments like **CartPole** and **MiniGrid**.

---

# QTRL: Quantum-Train Reinforcement Learning

This repository contains a reproduction of a hybrid reinforcement learning agent where a **photonic quantum circuit** (simulated via Perceval/Merlin) generates the weights for a classical policy network. This method leverages the high expressivity of quantum systems to parameterize classical models.

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
│   ├── CartPole_photonic.py # Specific logic for CartPole
│   └── miniGrid_photonic.py # Specific logic for MiniGrid
└── README.md

```

## ## How it Works

1. **Quantum Weight Generation**: A photonic circuit (linear optics) is initialized. The output states of this circuit are mapped through a small classical neural network to produce the weights of a policy.
2. **Policy Execution**: These generated weights are used to parameterize a linear layer that maps environment observations to action probabilities.
3. **REINFORCE Algorithm**: The agent is trained using the Policy Gradient theorem. We maximize the expected return by minimizing the negative log-likelihood of actions weighted by the discounted returns.

## ## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `env_name` | `CartPole` | Target environment (`CartPole` or `MiniGrid`) |
| `nb_photons` | `2` | Number of photons in the circuit |
| `nb_modes` | `3` | Number of spatial modes |
| `lr` | `0.01` | Learning rate for the Adam optimizer |
| `num_episodes` | `500` | Total training episodes |
| `hidden_sizes` | `[8, 8]` | Architecture of the classical mapping model |

## ## Installation & Usage

### 1. Prerequisites

Ensure you are in a virtual environment with the necessary dependencies installed:

* `perceval-quandela` / `merlin`
* `gymnasium`
* `torch`
* `minigrid` (for MiniGrid tasks)

### 2. Running an Experiment

Launch the experiment from the repository root using the global implementation script:

**Run with default parameters (CartPole):**

```bash
python implementation.py --paper QTRL

```

**Run on MiniGrid with a specific learning rate:**

```bash
python implementation.py --paper QTRL --env_name MiniGrid --final_output_size 441 --lr 0.005

```

**List available options:**

```bash
python implementation.py --paper QTRL --help

```

## ## Implementation Details

* **Optimization**: The `optimizer` updates both the parameters of the quantum circuit (interferometer phases) and the classical mapping weights simultaneously.
* **Discounted Returns**: Returns are computed recursively ($G_t = r_t + \gamma G_{t+1}$) and standardized (Z-score) to stabilize the quantum-classical gradients.
* **Clipping**: Gradient clipping is applied to prevent the "exploding gradient" problem common in hybrid training.

---

> **Note:** This project is part of a research reproduction effort. Results may vary based on the photonic backend and noise models used.