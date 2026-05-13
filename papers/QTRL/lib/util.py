import torch
import torch.nn as nn
from merlin.builder import CircuitBuilder
from merlin import QuantumLayer, LexGrouping
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import perceval as pcvl
import numpy as np
import random
import os
from gymnasium.core import ObservationWrapper
import minigrid

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



class HybridQMLModel(nn.Module):
    def __init__(self, q_output_size, nb_photons, nb_modes, hidden_sizes, final_output_size):
        super().__init__()
        
        self.nb_modes = nb_modes 

        self.quantum_layer = QLayer( 
            Q_output_size=q_output_size, 
            nb_photons=nb_photons, 
            nb_modes=nb_modes
        )
        
        self.classical_mapping = MappingModel(
            input_size=q_output_size, 
            hidden_sizes=hidden_sizes, 
            output_size=final_output_size
        )

    def forward(self):
        x = self.quantum_layer()
        x = self.classical_mapping(x)
        return x

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
        print(f"Épisode {episode:4d} | Récompense : {total_reward:5.1f} | Loss : {policy_loss.item():.4f}")

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


def train_minigrid_qtrl(hybrid_model, num_episodes=1000, learning_rate=0.001, seed=42):
    

    base_env = gym.make('MiniGrid-Empty-5x5-v0')
    

    env = MinigridImageOnlyWrapper(base_env)
    
    action_dim = 3 
    state_dim = env.observation_space.shape[0]
    
    total_weights_needed = state_dim * action_dim
    
    print(f"🚀 Début de l'entraînement sur MiniGrid 5x5...")
    print(f"📊 Paramètres requis : {total_weights_needed} ({state_dim} entrées x {action_dim} actions)")
    
    optimizer = optim.Adam(hybrid_model.parameters(), lr=learning_rate)
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        
        log_probs = []
        rewards = []
        
        raw_weights = hybrid_model()[0] 
        
        if raw_weights.shape[0] < total_weights_needed:
            raise ValueError(f"CRASH: Le modèle sort {raw_weights.shape[0]} valeurs. Il en faut {total_weights_needed}.")
            
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
        
        print(f"Épisode {episode:4d} | Étapes: {step_count:3d} | Rép: {total_reward:5.2f} | Loss: {loss_val:.4f}")

    env.close()
    return total_reward