# from merlin.builder import CircuitBuilder
# from merlin import QuantumLayer, LexGrouping
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import gymnasium as gym
# import numpy as np
# import perceval as pcvl
# import random
# import os
# from gymnasium.core import ObservationWrapper
# import minigrid
# from lib.torchmps import MPS
# import torchquantum as tq
# import torchquantum.functional as tqf
# from torchquantum.device import QuantumDevice
# """
# Class QLayer, that initiate a photonic circuit with n photons and m modes.
# """
# class QLayer(nn.Module):
#     def __init__(self, Q_output_size, nb_photons, nb_modes):
#             super().__init__()
#             self.Q_output_size = Q_output_size
#             self.nb_photons = nb_photons
#             self.nb_modes = nb_modes
#             circuit = CircuitBuilder(n_modes=self.nb_modes)
#             circuit.add_entangling_layer(trainable=True, name="U3")


#             self.layer1 = QuantumLayer(
#                 builder=circuit,
#                 n_photons=self.nb_photons,
#                 dtype=torch.float32,
#                 )

#             self.layer2 = LexGrouping(self.layer1.output_size, Q_output_size)
#     def forward(self):
#         return self.layer2(self.layer1())


# """
# We use a mapping model to transform the state vectors into a matrix.
# """
# class MappingModel(nn.Module):
#     def __init__(self, input_size, hidden_sizes, output_size):
#         super().__init__()

#         dims = [input_size] + hidden_sizes + [output_size]
#         layers = []

#         for i in range(len(dims) - 1):
#             layers.append(nn.Linear(dims[i], dims[i+1]))

#             if i < len(dims) - 2:
#                 layers.append(nn.ReLU())

#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.network(x.type_as(self.network[0].weight))

# """
# We form here the hybrid model.
# """

# class HybridMLPModel(nn.Module):
#     def __init__(self, q_output_size, nb_photons, nb_modes, hidden_sizes, final_output_size):
#         super().__init__()
#         self.quantum_layer = QLayer(q_output_size, nb_photons, nb_modes)
#         self.mapping = MappingModel(q_output_size, hidden_sizes, final_output_size)

#     def forward(self):
#         return self.mapping(self.quantum_layer())


# """
# A class to replace the mapping model with Matrix product state.
# """
# class HybridMPSModel(nn.Module):
#     def __init__(self, q_output_size, nb_photons, nb_modes, bond_dim, final_output_size):
#         super().__init__()
#         self.quantum_layer = QLayer(q_output_size, nb_photons, nb_modes)
#         self.mapping = MPS(input_dim=q_output_size, output_dim=final_output_size, bond_dim=bond_dim)

#     def forward(self):
#         return self.mapping(self.quantum_layer())


# def generate_qubit_states_torch(n_qubit):
#     """
#     Create a tensor of shape (2**n_qubit, n_qubit) with all possible combinations of 0 and 1
#     """
#     all_states = torch.cartesian_prod(*[torch.tensor([-1, 1]) for _ in range(n_qubit)])
#     return all_states


# class TorchQuantumModel(nn.Module):
#     def __init__(self, q_output_size, n_qubit, q_depth, hidden_sizes, final_output_size):
#         super().__init__()
#         self.n_qubit = n_qubit
#         self.q_depth = q_depth
#         """
#         Here we replace merlin by torchquantum for the QLayer, just to compare.
#         """
#         # We create the tensor.
#         self.q_params_u3 = nn.Parameter(torch.randn(self.q_depth, self.n_qubit, 3) * 0.1)
#         self.q_params_cu3 = nn.Parameter(torch.randn(self.q_depth, self.n_qubit, 3) * 0.1)

#         # 2. Le Mapping Classique
#         self.mapping = MappingModel(
#             input_size=q_output_size,
#             hidden_sizes=hidden_sizes,
#             output_size=final_output_size
#         )

#     def forward(self):
#             # Quantum device initialisation.
#             qdev = QuantumDevice(n_wires=self.n_qubit, bsz=1, device=next(self.parameters()).device)

#             for k in range(self.q_depth):

#                 # Step 1 U3 door on all qubits
#                 for i in range(self.n_qubit):
#                     tqf.u3(qdev, wires=i, params=self.q_params_u3[k, i].unsqueeze(0))

#                 # Step 2 : CU3 layers addition
#                 for i in range(self.n_qubit):
#                     cible = (i + 1) % self.n_qubit
#                     tqf.cu3(qdev, wires=[i, cible], params=self.q_params_cu3[k, i].unsqueeze(0))

#             # output computation
#             state_mag = qdev.get_states_1d().abs()[0]
#             probs = state_mag ** 2

#             # We can only take the last probabilities to make sure we can map it.
#             probs = probs[:self.mapping.network[0].in_features]

#             return self.mapping(probs.unsqueeze(0))

# class classic_model(nn.Module):
#     def __init__(self, layer_dim, hidden_sizes, final_output_size):
#         super().__init__()

#         """
#         Here we replace QLayer with a classic model.

#         """
#         self.input_dim = layer_dim[0]

#         tab = []

#         for i in range(len(layer_dim) - 1):
#             tab.append(nn.Linear(layer_dim[i], layer_dim[i+1]))

#             if i < len(layer_dim) - 2:
#                 tab.append(nn.ReLU())

#         tab.append(nn.Softmax(dim=-1))

#         self.layer = nn.Sequential(*tab)

#         self.mapping = MappingModel(layer_dim[-1], hidden_sizes, final_output_size)

#     def forward(self):
#             device = next(self.parameters()).device
#             x = torch.ones(1, self.input_dim, device=device)
#             # Faire passer la sortie à travers le MappingModel
#             return self.mapping(self.layer(x))


# def set_global_seed(seed=42):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False


# def rl_agent_forward(state_tensor, generated_weights, input_dim=4, output_dim=2):
#     weight_matrix = generated_weights.view(output_dim, input_dim)

#     logits = F.linear(state_tensor, weight_matrix)
#     return logits


# def compute_discounted_returns(rewards, gamma=0.99):
#     returns = []
#     R = 0
#     for r in reversed(rewards):
#         R = r + gamma * R
#         returns.insert(0, R)

#     returns = torch.tensor(returns, dtype=torch.float32)
#     returns = (returns - returns.mean()) / (returns.std() + 1e-9)
#     return returns


# def train_cartpole_qtrl(hybrid_model, num_episodes=1000, learning_rate=0.001):
#     env = gym.make('CartPole-v1')

#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n
#     total_weights_needed = state_dim * action_dim

#     optimizer = optim.Adam(hybrid_model.parameters(), lr=learning_rate)


#     print("Begining CartPole training ...")

#     for episode in range(num_episodes):
#         state, _ = env.reset()

#         log_probs = []
#         rewards = []


#         raw_weights = hybrid_model()[0]

#         episode_weights = raw_weights[:total_weights_needed]

#         step_count=0
#         done = False
#         while not done:
#             state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#             logits = rl_agent_forward(state_tensor, episode_weights, state_dim, action_dim)


#             action_dist = torch.distributions.Categorical(logits=logits)
#             action = action_dist.sample()


#             next_state, reward, terminated, truncated, _ = env.step(action.item())
#             done = terminated or truncated


#             log_probs.append(action_dist.log_prob(action))
#             rewards.append(reward)

#             state = next_state
#             step_count += 1


#         discounted_returns = compute_discounted_returns(rewards)

#         policy_loss = []
#         for log_prob, R in zip(log_probs, discounted_returns):
#             policy_loss.append(-log_prob * R)
#         policy_loss = torch.cat(policy_loss).sum()

#         optimizer.zero_grad()
#         policy_loss.backward()


#         torch.nn.utils.clip_grad_norm_(hybrid_model.parameters(), max_norm=1.0)

#         optimizer.step()


#         total_reward = sum(rewards)
#         print(f"Episode {episode:4d} | rewards : {total_reward:5.1f} | Loss : {policy_loss.item():.4f}")

#     env.close()
#     return total_reward

# class MinigridImageOnlyWrapper(ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)

#         shape = self.env.observation_space.spaces['image'].shape
#         self.flat_dim = shape[0] * shape[1] * shape[2]

#         self.observation_space = gym.spaces.Box(
#             low=0.0, high=1.0, shape=(self.flat_dim,), dtype=np.float32
#         )

#     def observation(self, obs):
#         image = obs['image']
#         flat_image = image.flatten().astype(np.float32) / 10.0
#         return flat_image


# def train_minigrid_qtrl(hybrid_model, num_episodes=1000, learning_rate=0.005, seed=42):
#     base_env = gym.make('MiniGrid-Empty-5x5-v0')
#     env = MinigridImageOnlyWrapper(base_env)
#     action_dim = 3
#     state_dim = env.observation_space.shape[0]

#     total_weights_needed = state_dim * action_dim

#     print(f"MiniGrid's train begins...")
#     print(f"required parameters : {total_weights_needed} ({state_dim} entry x {action_dim} actions)")

#     optimizer = optim.Adam(hybrid_model.parameters(), lr=learning_rate)

#     for episode in range(num_episodes):
#         state, _ = env.reset(seed=seed if episode == 0 else None)

#         log_probs = []
#         rewards = []

#         raw_weights = hybrid_model()[0]

#         if raw_weights.shape[0] < total_weights_needed:
#             raise ValueError(f"CRASH: {raw_weights.shape[0]} values. We need {total_weights_needed}.")

#         episode_weights = raw_weights[:total_weights_needed]

#         done = False
#         step_count = 0

#         while not done:
#             state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

#             logits = rl_agent_forward(state_tensor, episode_weights, state_dim, action_dim)

#             logits = logits / 2.0

#             action_dist = torch.distributions.Categorical(logits=logits)
#             action = action_dist.sample()

#             next_state, reward, terminated, truncated, _ = env.step(action.item())
#             done = terminated or truncated

#             if reward == 0 and not done:
#                 reward = -0.01

#             log_probs.append(action_dist.log_prob(action))
#             rewards.append(reward)

#             state = next_state
#             step_count += 1

#         discounted_returns = compute_discounted_returns(rewards)

#         policy_loss = []
#         for log_prob, R in zip(log_probs, discounted_returns):
#             policy_loss.append(-log_prob * R)

#         if len(policy_loss) > 0:
#             policy_loss_tensor = torch.cat(policy_loss).sum()
#             optimizer.zero_grad()
#             policy_loss_tensor.backward()
#             torch.nn.utils.clip_grad_norm_(hybrid_model.parameters(), max_norm=1.0)
#             optimizer.step()

#             loss_val = policy_loss_tensor.item()
#         else:
#             loss_val = 0.0

#         total_reward = sum(rewards)

#         print(f"Episode {episode:4d} | steps: {step_count:3d} | reward: {total_reward:5.2f} | Loss: {loss_val:.4f}")

#     env.close()
#     return total_reward


# def train_environment(hybrid_model, num_episode=1000, learning_rate=0.005, env_name="CartPole-v1", seed=42):
#     # Charger l'environnement
#     if env_name == "MiniGrid-Empty-5x5-v0":
#         base_env = gym.make(env_name)
#         env = MinigridImageOnlyWrapper(base_env)
#         is_minigrid = True
#     else:
#         env = gym.make(env_name)
#         is_minigrid = False

#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n
#     total_weights_needed = state_dim * action_dim

#     print(f"🚀 Début de l'entraînement sur {env_name} | Poids requis : {total_weights_needed}")

#     optimizer = optim.Adam(hybrid_model.parameters(), lr=learning_rate)

#     for episodes in range(num_episode):
#         state, _ = env.reset(seed=seed if episodes == 0 else None)

#         log_probs = []
#         rewards = []

#         raw_weights = hybrid_model()[0]

#         if raw_weights.shape[0] < total_weights_needed:
#             raise ValueError(f"CRASH: {raw_weights.shape[0]} values. We need {total_weights_needed}.")

#         episode_weights = raw_weights[:total_weights_needed]

#         done = False
#         step_count = 0

#         while not done:
#             # CORRECTION 1 : Conversion en tenseur
#             state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

#             # CORRECTION 2 : Passer episode_weights au lieu de hybrid_model
#             logits = rl_agent_forward(state_tensor, episode_weights, input_dim=state_dim, output_dim=action_dim)

#             # Division des logits pour l'exploration (comme dans ton code original)
#             logits = logits / 2.0

#             action_dist = torch.distributions.Categorical(logits=logits)
#             action = action_dist.sample()

#             next_state, reward, terminated, truncated, _ = env.step(action.item())
#             done = terminated or truncated

#             # CORRECTION 3 : Remettre la pénalité pour MiniGrid uniquement
#             if is_minigrid and reward == 0 and not done:
#                 reward = -0.01

#             log_probs.append(action_dist.log_prob(action))
#             rewards.append(reward)

#             state = next_state
#             step_count += 1 # CORRECTION 4 : Incrémenter les steps

#         # Calcul du reward et Backpropagation
#         discount_return = compute_discounted_returns(rewards)
#         policy_loss = []
#         for log_prob, R in zip(log_probs, discount_return):
#             policy_loss.append(-log_prob * R)

#         if len(policy_loss) > 0:
#             policy_loss_tensor = torch.cat(policy_loss).sum()
#             optimizer.zero_grad()
#             policy_loss_tensor.backward()
#             torch.nn.utils.clip_grad_norm_(hybrid_model.parameters(), max_norm=1.0)
#             optimizer.step()

#             loss_val = policy_loss_tensor.item()
#         else:
#             loss_val = 0.0

#         total_reward = sum(rewards)

#         print(f"Episode {episodes:4d} | steps: {step_count:3d} | reward: {total_reward:5.2f} | Loss: {loss_val:.4f}")

#     env.close()
#     return total_reward
# # model = HybridMLPModel(10, 10, 10, [6,6], 15)
# # print("début test avec minigrid")
# # train_environment(model, env_name="MiniGrid-Empty-5x5-v0")
# # print("début test avec cartpole")
# # train_environment(model)


# def create_hybrid_model(args, total_weights_needed):
#     """
#     Instancie le bon modèle en fonction du backend choisi.
#     """
#     backend = args.get("backend", "merlin_mlp")

#     if backend == "merlin_mlp":
#         return HybridMLPModel(
#             q_output_size=args["q_output_size"],
#             nb_photons=args["nb_photons"],
#             nb_modes=args["nb_modes"],
#             hidden_sizes=args["hidden_sizes"], # Utilisé uniquement ici
#             final_output_size=total_weights_needed
#         )

#     elif backend == "merlin_mps":
#         return HybridMPSModel(
#             q_output_size=args["q_output_size"],
#             nb_photons=args["nb_photons"],
#             nb_modes=args.get("nb_modes", 3),
#             bond_dim=args.get("bond_dim", 2),         # Utilisé uniquement ici
#             final_output_size=total_weights_needed
#         )

#     elif backend == "torchquantum":
#             # On utilise .get() pour n_qubit et q_depth afin d'éviter un crash
#             # s'ils n'ont pas encore été ajoutés dans defaults.json
#             return TorchQuantumModel(
#                 q_output_size=args["q_output_size"],
#                 n_qubit=args.get("n_qubit", 4),
#                 q_depth=args.get("q_depth", 2),
#                 hidden_sizes=args["hidden_sizes"],
#                 final_output_size=total_weights_needed
#             )
#     elif backend == "classic":
#         return classic_model(
#             layer_dim=args["layer_dim"],
#             hidden_sizes=args["hidden_sizes"],
#             final_output_size=total_weights_needed
#         )

#     else:
#         raise ValueError(f"Backend inconnu : {backend}")


import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchquantum.functional as tqf
from gymnasium.core import ObservationWrapper
from lib.torchmps import MPS
from merlin import LexGrouping, QuantumLayer
from merlin.builder import CircuitBuilder
from torchquantum.device import QuantumDevice

"""
Class QLayer: Initializes a photonic circuit with a given number of photons and modes.
"""


class QLayer(nn.Module):
    """
    Photonic quantum layer using Merlin framework.

    Args:
        Q_output_size (int): Desired output size after lexical grouping.
        nb_photons (int): Number of photons in the circuit.
        nb_modes (int): Number of modes in the photonic circuit.
    """

    def __init__(self, Q_output_size: int, nb_photons: int, nb_modes: int):
        super().__init__()
        self.Q_output_size = Q_output_size
        self.nb_photons = nb_photons
        self.nb_modes = nb_modes
        circuit = CircuitBuilder(n_modes=self.nb_modes)
        circuit.add_entangling_layer(trainable=True, name="U3")

        self.layer1 = QuantumLayer(
            builder=circuit,
            n_photons=self.nb_photons,
            dtype=torch.float32,
        )

        self.layer2 = LexGrouping(self.layer1.output_size, Q_output_size)

    def forward(self):
        return self.layer2(self.layer1())


"""
MappingModel: Transforms state vectors into a matrix using a classical MLP.
"""


class MappingModel(nn.Module):
    """
    Classical neural network used to map quantum outputs to final required dimensions.

    Args:
        input_size (int): Input feature dimension.
        hidden_sizes (list[int]): List of hidden layer sizes.
        output_size (int): Final output dimension.
    """

    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()

        dims = [input_size] + hidden_sizes + [output_size]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.type_as(self.network[0].weight))


"""
HybridMLPModel: Combines a photonic quantum layer with a classical mapping network.
"""


class HybridMLPModel(nn.Module):
    """
    Hybrid model using Merlin photonic circuit + classical MLP mapping.

    Args:
        q_output_size (int): Output size of the quantum layer before mapping.
        nb_photons (int): Number of photons.
        nb_modes (int): Number of modes.
        hidden_sizes (list[int]): Hidden sizes for the mapping network.
        final_output_size (int): Final output dimension (usually number of weights needed).
    """

    def __init__(
        self,
        q_output_size: int,
        nb_photons: int,
        nb_modes: int,
        hidden_sizes: list[int],
        final_output_size: int,
    ):
        super().__init__()
        self.quantum_layer = QLayer(q_output_size, nb_photons, nb_modes)
        self.mapping = MappingModel(q_output_size, hidden_sizes, final_output_size)

    def forward(self):
        return self.mapping(self.quantum_layer())


"""
HybridMPSModel: Uses Matrix Product State (MPS) instead of MLP for the classical mapping part.
"""


class HybridMPSModel(nn.Module):
    """
    Hybrid model using Merlin photonic circuit with Matrix Product State mapping.

    Args:
        q_output_size (int): Output size of the quantum layer.
        nb_photons (int): Number of photons.
        nb_modes (int): Number of modes.
        bond_dim (int): Bond dimension for the MPS.
        final_output_size (int): Final output dimension.
    """

    def __init__(
        self,
        q_output_size: int,
        nb_photons: int,
        nb_modes: int,
        bond_dim: int,
        final_output_size: int,
    ):
        super().__init__()
        self.quantum_layer = QLayer(q_output_size, nb_photons, nb_modes)
        self.mapping = MPS(
            input_dim=q_output_size, output_dim=final_output_size, bond_dim=bond_dim
        )

    def forward(self):
        return self.mapping(self.quantum_layer())


def generate_qubit_states_torch(n_qubit: int) -> torch.Tensor:
    """
    Create a tensor of shape (2**n_qubit, n_qubit) containing all possible combinations of -1 and 1.

    Args:
        n_qubit (int): Number of qubits.

    Returns:
        torch.Tensor: Tensor with all possible qubit basis states.
    """
    all_states = torch.cartesian_prod(*[torch.tensor([-1, 1]) for _ in range(n_qubit)])
    return all_states


class TorchQuantumModel(nn.Module):
    """
    TorchQuantum based hybrid model for comparison with Merlin.

    Args:
        q_output_size (int): Output size from quantum circuit.
        n_qubit (int): Number of qubits.
        q_depth (int): Depth of the quantum circuit.
        hidden_sizes (list[int]): Hidden sizes for mapping.
        final_output_size (int): Final output dimension.
    """

    def __init__(
        self,
        q_output_size: int,
        n_qubit: int,
        q_depth: int,
        hidden_sizes: list[int],
        final_output_size: int,
    ):
        super().__init__()
        self.n_qubit = n_qubit
        self.q_depth = q_depth
        """
        Here we replace merlin by torchquantum for the QLayer, just to compare.
        """
        # We create the tensor.
        self.q_params_u3 = nn.Parameter(
            torch.randn(self.q_depth, self.n_qubit, 3) * 0.1
        )
        self.q_params_cu3 = nn.Parameter(
            torch.randn(self.q_depth, self.n_qubit, 3) * 0.1
        )

        # 2. Classical Mapping
        self.mapping = MappingModel(
            input_size=q_output_size,
            hidden_sizes=hidden_sizes,
            output_size=final_output_size,
        )

    def forward(self):
        # Quantum device initialisation.
        qdev = QuantumDevice(
            n_wires=self.n_qubit, bsz=1, device=next(self.parameters()).device
        )

        for k in range(self.q_depth):
            # Step 1: U3 gates on all qubits
            for i in range(self.n_qubit):
                tqf.u3(qdev, wires=i, params=self.q_params_u3[k, i].unsqueeze(0))

            # Step 2: Add CU3 layers
            for i in range(self.n_qubit):
                cible = (i + 1) % self.n_qubit
                tqf.cu3(
                    qdev, wires=[i, cible], params=self.q_params_cu3[k, i].unsqueeze(0)
                )

        # output computation
        state_mag = qdev.get_states_1d().abs()[0]
        probs = state_mag**2

        # We can only take the first probabilities to match mapping input size
        probs = probs[: self.mapping.network[0].in_features]

        return self.mapping(probs.unsqueeze(0))


class classic_model(nn.Module):
    """
    Pure classical baseline model (no quantum layer).

    Args:
        layer_dim (list[int]): Dimensions of the classical feature extractor layers.
        hidden_sizes (list[int]): Hidden sizes for the mapping network.
        final_output_size (int): Final output dimension.
    """

    def __init__(
        self, layer_dim: list[int], hidden_sizes: list[int], final_output_size: int
    ):
        super().__init__()

        """
        Here we replace QLayer with a classic model.
        """
        self.input_dim = layer_dim[0]

        tab = []

        for i in range(len(layer_dim) - 1):
            tab.append(nn.Linear(layer_dim[i], layer_dim[i + 1]))

            if i < len(layer_dim) - 2:
                tab.append(nn.ReLU())

        tab.append(nn.Softmax(dim=-1))

        self.layer = nn.Sequential(*tab)

        self.mapping = MappingModel(layer_dim[-1], hidden_sizes, final_output_size)

    def forward(self):
        device = next(self.parameters()).device
        x = torch.ones(1, self.input_dim, device=device)
        # Pass the output through the MappingModel
        return self.mapping(self.layer(x))


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy and PyTorch.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rl_agent_forward(
    state_tensor: torch.Tensor,
    generated_weights: torch.Tensor,
    input_dim: int = 4,
    output_dim: int = 2,
) -> torch.Tensor:
    """
    Forward pass of a linear policy using generated weights.

    Args:
        state_tensor (torch.Tensor): Current state observation.
        generated_weights (torch.Tensor): Flattened weights for the policy.
        input_dim (int): Input dimension (state size).
        output_dim (int): Output dimension (number of actions).

    Returns:
        torch.Tensor: Action logits.
    """
    weight_matrix = generated_weights.view(output_dim, input_dim)

    logits = F.linear(state_tensor, weight_matrix)
    return logits


def compute_discounted_returns(
    rewards: list[float], gamma: float = 0.99
) -> torch.Tensor:
    """
    Compute discounted returns and normalize them.

    Args:
        rewards (list[float]): List of rewards from an episode.
        gamma (float): Discount factor.

    Returns:
        torch.Tensor: Normalized discounted returns.
    """
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    return returns


def train_cartpole_qtrl(
    hybrid_model: nn.Module, num_episodes: int = 1000, learning_rate: float = 0.001
) -> float:
    """
    Train the hybrid model on CartPole-v1 using REINFORCE.

    Args:
        hybrid_model (nn.Module): The hybrid quantum-classical model.
        num_episodes (int): Number of episodes to train.
        learning_rate (float): Learning rate for Adam optimizer.

    Returns:
        float: Total reward of the last episode.
    """
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    total_weights_needed = state_dim * action_dim

    optimizer = optim.Adam(hybrid_model.parameters(), lr=learning_rate)

    print("Beginning CartPole training ...")

    for episode in range(num_episodes):
        state, _ = env.reset()

        log_probs = []
        rewards = []

        raw_weights = hybrid_model()[0]

        episode_weights = raw_weights[:total_weights_needed]

        step_count = 0
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = rl_agent_forward(
                state_tensor, episode_weights, state_dim, action_dim
            )

            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(action_dist.log_prob(action))
            rewards.append(reward)

            state = next_state
            step_count += 1

        discounted_returns = compute_discounted_returns(rewards)

        policy_loss = []
        for log_prob, R in zip(log_probs, discounted_returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()

        torch.nn.utils.clip_grad_norm_(hybrid_model.parameters(), max_norm=1.0)

        optimizer.step()

        total_reward = sum(rewards)
        print(
            f"Episode {episode:4d} | rewards : {total_reward:5.1f} | Loss : {policy_loss.item():.4f}"
        )

    env.close()
    return total_reward


class MinigridImageOnlyWrapper(ObservationWrapper):
    """
    Observation wrapper that flattens the image observation from MiniGrid.
    """

    def __init__(self, env):
        super().__init__(env)

        shape = self.env.observation_space.spaces["image"].shape
        self.flat_dim = shape[0] * shape[1] * shape[2]

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.flat_dim,), dtype=np.float32
        )

    def observation(self, obs):
        image = obs["image"]
        flat_image = image.flatten().astype(np.float32) / 10.0
        return flat_image


def train_minigrid_qtrl(
    hybrid_model: nn.Module,
    num_episodes: int = 1000,
    learning_rate: float = 0.005,
    seed: int = 42,
) -> float:
    """
    Train the hybrid model on MiniGrid-Empty-5x5-v0 using REINFORCE.

    Args:
        hybrid_model (nn.Module): The hybrid model.
        num_episodes (int): Number of training episodes.
        learning_rate (float): Learning rate.
        seed (int): Random seed.

    Returns:
        float: Total reward of the last episode.
    """
    base_env = gym.make("MiniGrid-Empty-5x5-v0")
    env = MinigridImageOnlyWrapper(base_env)
    action_dim = 3
    state_dim = env.observation_space.shape[0]

    total_weights_needed = state_dim * action_dim

    print("MiniGrid training begins...")
    print(
        f"Required parameters : {total_weights_needed} ({state_dim} inputs x {action_dim} actions)"
    )

    optimizer = optim.Adam(hybrid_model.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed if episode == 0 else None)

        log_probs = []
        rewards = []

        raw_weights = hybrid_model()[0]

        if raw_weights.shape[0] < total_weights_needed:
            raise ValueError(
                f"CRASH: {raw_weights.shape[0]} values. We need {total_weights_needed}."
            )

        episode_weights = raw_weights[:total_weights_needed]

        done = False
        step_count = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            logits = rl_agent_forward(
                state_tensor, episode_weights, state_dim, action_dim
            )

            logits = logits / 2.0

            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            if reward == 0 and not done:
                reward = -0.01

            log_probs.append(action_dist.log_prob(action))
            rewards.append(reward)

            state = next_state
            step_count += 1

        discounted_returns = compute_discounted_returns(rewards)

        policy_loss = []
        for log_prob, R in zip(log_probs, discounted_returns):
            policy_loss.append(-log_prob * R)

        if len(policy_loss) > 0:
            policy_loss_tensor = torch.cat(policy_loss).sum()
            optimizer.zero_grad()
            policy_loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(hybrid_model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = policy_loss_tensor.item()
        else:
            loss_val = 0.0

        total_reward = sum(rewards)

        print(
            f"Episode {episode:4d} | steps: {step_count:3d} | reward: {total_reward:5.2f} | Loss: {loss_val:.4f}"
        )

    env.close()
    return total_reward


def train_environment(
    hybrid_model: nn.Module,
    num_episode: int = 1000,
    learning_rate: float = 0.005,
    env_name: str = "CartPole-v1",
    seed: int = 42,
) -> float:
    """
    Generic training function for different environments (CartPole or MiniGrid).

    Args:
        hybrid_model (nn.Module): The hybrid model to train.
        num_episode (int): Number of episodes.
        learning_rate (float): Learning rate.
        env_name (str): Name of the Gym environment.
        seed (int): Random seed.

    Returns:
        float: Total reward of the last episode.
    """
    # Load environment
    if env_name == "MiniGrid-Empty-5x5-v0":
        base_env = gym.make(env_name)
        env = MinigridImageOnlyWrapper(base_env)
        is_minigrid = True
    else:
        env = gym.make(env_name)
        is_minigrid = False

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    total_weights_needed = state_dim * action_dim

    print(
        f"Starting training on {env_name} | Required weights : {total_weights_needed}"
    )

    optimizer = optim.Adam(hybrid_model.parameters(), lr=learning_rate)

    for episode in range(num_episode):
        state, _ = env.reset(seed=seed if episode == 0 else None)

        log_probs = []
        rewards = []

        raw_weights = hybrid_model()[0]

        if raw_weights.shape[0] < total_weights_needed:
            raise ValueError(
                f"CRASH: {raw_weights.shape[0]} values. We need {total_weights_needed}."
            )

        episode_weights = raw_weights[:total_weights_needed]

        done = False
        step_count = 0

        while not done:
            # Convert to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # Use generated weights for the linear policy
            logits = rl_agent_forward(
                state_tensor,
                episode_weights,
                input_dim=state_dim,
                output_dim=action_dim,
            )

            # Scale logits for better exploration
            logits = logits / 2.0

            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # Penalty for MiniGrid only
            if is_minigrid and reward == 0 and not done:
                reward = -0.01

            log_probs.append(action_dist.log_prob(action))
            rewards.append(reward)

            state = next_state
            step_count += 1

        # Compute loss and update
        discount_return = compute_discounted_returns(rewards)
        policy_loss = []
        for log_prob, R in zip(log_probs, discount_return):
            policy_loss.append(-log_prob * R)

        if len(policy_loss) > 0:
            policy_loss_tensor = torch.cat(policy_loss).sum()
            optimizer.zero_grad()
            policy_loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(hybrid_model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = policy_loss_tensor.item()
        else:
            loss_val = 0.0

        total_reward = sum(rewards)

        print(
            f"Episode {episode:4d} | steps: {step_count:3d} | reward: {total_reward:5.2f} | Loss: {loss_val:.4f}"
        )

    env.close()
    return total_reward


def create_hybrid_model(args: dict, total_weights_needed: int) -> nn.Module:
    """
    Instantiate the correct hybrid model based on the selected backend.

    Args:
        args (dict): Configuration dictionary containing model parameters.
        total_weights_needed (int): Number of weights the policy needs.

    Returns:
        nn.Module: The instantiated hybrid model.
    """
    backend = args.get("backend", "merlin_mlp")

    if backend == "merlin_mlp":
        return HybridMLPModel(
            q_output_size=args["q_output_size"],
            nb_photons=args["nb_photons"],
            nb_modes=args["nb_modes"],
            hidden_sizes=args["hidden_sizes"],
            final_output_size=total_weights_needed,
        )

    elif backend == "merlin_mps":
        return HybridMPSModel(
            q_output_size=args["q_output_size"],
            nb_photons=args["nb_photons"],
            nb_modes=args.get("nb_modes", 3),
            bond_dim=args.get("bond_dim", 2),
            final_output_size=total_weights_needed,
        )

    elif backend == "torchquantum":
        return TorchQuantumModel(
            q_output_size=args["q_output_size"],
            n_qubit=args.get("n_qubit", 4),
            q_depth=args.get("q_depth", 2),
            hidden_sizes=args["hidden_sizes"],
            final_output_size=total_weights_needed,
        )
    elif backend == "classic":
        return classic_model(
            layer_dim=args["layer_dim"],
            hidden_sizes=args["hidden_sizes"],
            final_output_size=total_weights_needed,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")
