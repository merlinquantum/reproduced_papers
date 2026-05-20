from merlin.builder import CircuitBuilder
from merlin import QuantumLayer, LexGrouping
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import perceval as pcvl
import random
import os
from gymnasium.core import ObservationWrapper
import minigrid
from lib.torchmps import MPS
import torchquantum as tq


class QLayer(nn.Module):
    def __init__(self, Q_output_size, nb_photons, nb_modes):
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



class MappingModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        
        dims = [input_size] + hidden_sizes + [output_size]
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.type_as(self.network[0].weight))



class HybridMLPModel(nn.Module):
    def __init__(self, q_output_size, nb_photons, nb_modes, hidden_sizes, final_output_size):
        super().__init__()
        # QLayer est instancié localement
        self.quantum_layer = QLayer(q_output_size, nb_photons, nb_modes)
        self.mapping = MappingModel(q_output_size, hidden_sizes, final_output_size)

    def forward(self):
        return self.mapping(self.quantum_layer())

class HybridMPSModel(nn.Module):
    def __init__(self, q_output_size, nb_photons, nb_modes, bond_dim, final_output_size):
        super().__init__()
        # QLayer est instancié localement
        self.quantum_layer = QLayer(q_output_size, nb_photons, nb_modes)
        self.mapping = MPS(input_dim=q_output_size, output_dim=final_output_size, bond_dim=bond_dim)

    def forward(self):
        return self.mapping(self.quantum_layer())




def generate_qubit_states_torch(n_qubit):
    # Create a tensor of shape (2**n_qubit, n_qubit) with all possible combinations of 0 and 1
    all_states = torch.cartesian_prod(*[torch.tensor([-1, 1]) for _ in range(n_qubit)])
    return all_states


import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf  
from torchquantum.device import QuantumDevice

class TorchQuantumModel(nn.Module):
    def __init__(self, q_output_size, n_qubit, q_depth, hidden_sizes, final_output_size):
        super().__init__()
        self.n_qubit = n_qubit
        self.q_depth = q_depth
        
        # 1. Gestion manuelle des paramètres quantiques
        # La porte U3 (et CU3) nécessite 3 paramètres (theta, phi, lambda).
        # On crée donc un tenseur de shape [profondeur, nb_qubits, 3]
        self.q_params_u3 = nn.Parameter(torch.randn(self.q_depth, self.n_qubit, 3) * 0.1)
        self.q_params_cu3 = nn.Parameter(torch.randn(self.q_depth, self.n_qubit, 3) * 0.1)
            
        # 2. Le Mapping Classique
        self.mapping = MappingModel(
            input_size=q_output_size, 
            hidden_sizes=hidden_sizes, 
            output_size=final_output_size
        )

    def forward(self):
            # Initialisation du device quantique
            qdev = QuantumDevice(n_wires=self.n_qubit, bsz=1, device=next(self.parameters()).device)
            
            # Exécution quantique avec l'API fonctionnelle
            for k in range(self.q_depth):
                
                # ÉTAPE 1 : Porte U3 sur tous les qubits
                for i in range(self.n_qubit):
                    # On ajoute .unsqueeze(0) pour simuler la dimension [batch_size=1, 3] attendue par tqf
                    tqf.u3(qdev, wires=i, params=self.q_params_u3[k, i].unsqueeze(0))
                
                # ÉTAPE 2 : Intrication circulaire avec CU3
                for i in range(self.n_qubit):
                    cible = (i + 1) % self.n_qubit
                    tqf.cu3(qdev, wires=[i, cible], params=self.q_params_cu3[k, i].unsqueeze(0))
                
            # Récupération des probabilités
            state_mag = qdev.get_states_1d().abs()[0] 
            probs = state_mag ** 2
            
            # On tronque les probabilités pour correspondre à l'entrée attendue par le Mapping
            probs = probs[:self.mapping.network[0].in_features]
            
            return self.mapping(probs.unsqueeze(0))


def set_global_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




def rl_agent_forward(state_tensor, generated_weights, input_dim=4, output_dim=2):
    weight_matrix = generated_weights.view(output_dim, input_dim)
    
    logits = F.linear(state_tensor, weight_matrix)
    return logits


def compute_discounted_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    return returns


def train_cartpole_qtrl(hybrid_model, num_episodes=1000, learning_rate=0.001):
    env = gym.make('CartPole-v1')
    

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    total_weights_needed = state_dim * action_dim
    

    optimizer = optim.Adam(hybrid_model.parameters(), lr=learning_rate)
    
    
    print("Begining of training CartPole...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        
        log_probs = []
        rewards = []
        

        raw_weights = hybrid_model()[0] 

        episode_weights = raw_weights[:total_weights_needed]
        

        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = rl_agent_forward(state_tensor, episode_weights, state_dim, action_dim)
            

            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            

            log_probs.append(action_dist.log_prob(action))
            rewards.append(reward)
            
            state = next_state
            

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
        print(f"Episode {episode:4d} | rewards : {total_reward:5.1f} | Loss : {policy_loss.item():.4f}")

    env.close()
    return total_reward

class MinigridImageOnlyWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        shape = self.env.observation_space.spaces['image'].shape
        self.flat_dim = shape[0] * shape[1] * shape[2]
        
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.flat_dim,), dtype=np.float32
        )

    def observation(self, obs):
        image = obs['image']
        flat_image = image.flatten().astype(np.float32) / 10.0
        return flat_image


def train_minigrid_qtrl(hybrid_model, num_episodes=1000, learning_rate=0.005, seed=42):
    base_env = gym.make('MiniGrid-Empty-5x5-v0')
    env = MinigridImageOnlyWrapper(base_env)
    action_dim = 3 
    state_dim = env.observation_space.shape[0]
    
    total_weights_needed = state_dim * action_dim
    
    print(f"MiniGrid's train begins...")
    print(f"required parameters : {total_weights_needed} ({state_dim} entry x {action_dim} actions)")
    
    optimizer = optim.Adam(hybrid_model.parameters(), lr=learning_rate)
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        
        log_probs = []
        rewards = []
        
        raw_weights = hybrid_model()[0] 
        
        if raw_weights.shape[0] < total_weights_needed:
            raise ValueError(f"CRASH: {raw_weights.shape[0]} values. We need {total_weights_needed}.")
            
        episode_weights = raw_weights[:total_weights_needed]
        
        done = False
        step_count = 0
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            logits = rl_agent_forward(state_tensor, episode_weights, state_dim, action_dim)

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
        
        print(f"Episode {episode:4d} | steps: {step_count:3d} | reward: {total_reward:5.2f} | Loss: {loss_val:.4f}")

    env.close()
    return total_reward



def create_hybrid_model(args, total_weights_needed):
    """
    Instancie le bon modèle en fonction du backend choisi.
    """
    backend = args.get("backend", "merlin_mlp")

    if backend == "merlin_mlp":
        return HybridMLPModel(
            q_output_size=args["q_output_size"],
            nb_photons=args["nb_photons"],
            nb_modes=args["nb_modes"],
            hidden_sizes=args["hidden_sizes"], # Utilisé uniquement ici
            final_output_size=total_weights_needed
        )
        
    elif backend == "merlin_mps":
        return HybridMPSModel(
            q_output_size=args["q_output_size"],
            nb_photons=args["nb_photons"],
            nb_modes=args["nb_modes"],
            bond_dim=args["bond_dim"],         # Utilisé uniquement ici
            final_output_size=total_weights_needed
        )
        
    elif backend == "torchquantum":
            # On utilise .get() pour n_qubit et q_depth afin d'éviter un crash 
            # s'ils n'ont pas encore été ajoutés dans defaults.json
            return TorchQuantumModel(
                q_output_size=args["q_output_size"],
                n_qubit=args.get("n_qubit", 4),
                q_depth=args.get("q_depth", 2),
                hidden_sizes=args["hidden_sizes"],
                final_output_size=total_weights_needed
            )
        
    else:
        raise ValueError(f"Backend inconnu : {backend}")