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


class QLayer(nn.Module):
    def __init__(self, Q_output_size, nb_photons, nb_modes):
            super().__init__()
            self.Q_output_size = Q_output_size
            self.nb_photons = nb_photons
            self.nb_modes = nb_modes
            # initialisation
            circuit = CircuitBuilder(n_modes=self.nb_modes) 
            

            
            #Quantum Layer.
            circuit.add_entangling_layer(trainable=True, name="U3")
            
            
            #On forme le circuit.
            self.layer1 = QuantumLayer(
                # input_size=Q_input_size,
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
        
        # On combine toutes les dimensions : [entrée, caché1, caché2, ..., sortie]
        dims = [input_size] + hidden_sizes + [output_size]
        layers = []
        
        # On itère sur les dimensions pour créer les paires (entrée, sortie)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # On ajoute ReLU partout SAUF sur la dernière couche (output)
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        # On package tout dans un Sequential
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # type_as reste utile si tu mélanges Double et Float
        return self.network(x.type_as(self.network[0].weight))



# # --- PARAMÈTRES DE TEST UNITAIRE QLAYER ---
# N_INPUT = 4    # Nombre de modes
# N_OUTPUT = 2   # Classes de sortie
# N_PHOTONS = 2  # Nombre de photons
# BATCH_SIZE = 5 # Nombre d'échantillons parallèles

# try:
#     # 1. Instanciation
#     model = QLayer(Q_input_size=N_INPUT, 
#                    Q_output_size=N_OUTPUT, 
#                    nb_photons=N_PHOTONS, 
#                    nb_modes=N_INPUT)
#     print("✅ Modèle instancié avec succès.")

#     # 2. Création d'une entrée factice
#     # MerLin attend généralement des amplitudes ou des paramètres de phase
#     dummy_input = torch.randn(BATCH_SIZE, N_INPUT) 

#     # 3. Passe forward
#     output = model(dummy_input)

#     # 4. Vérifications
#     print(f"✅ Passe forward réussie.")
#     print(f"Forme de l'entrée : {dummy_input.shape}")
#     print(f"Forme de la sortie : {output.shape}") # Devrait être [5, 2]

#     if output.shape == (BATCH_SIZE, N_OUTPUT):
#         print("🚀 Le test de dimension est validé !")

# except Exception as e:
#     print(f"❌ Échec du test : {str(e)}")
#     # Pour débugger, on peut afficher la structure interne de layer1
#     # print(model.layer1)



class HybridQMLModel(nn.Module):
    def __init__(self, q_output_size, nb_photons, nb_modes, hidden_sizes, final_output_size):
        super().__init__()
        
        # --- AJOUT ICI ---
        self.nb_modes = nb_modes 
        # -----------------
        
        # 1. La partie Quantique (QML)
        self.quantum_layer = QLayer( 
            Q_output_size=q_output_size, 
            nb_photons=nb_photons, 
            nb_modes=nb_modes
        )
        
        # 2. La partie Classique (Mapping / Post-traitement)
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
    """Verrouille toutes les sources d'aléatoire standards."""
    # 1. Python natif
    random.seed(seed)
    # 2. Variables d'environnement Python (hachage)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 3. NumPy
    np.random.seed(seed)
    # 4. PyTorch
    torch.manual_seed(seed)
    
    # 5. Si tu utilises CUDA (GPU) plus tard
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Force des algorithmes déterministes (peut ralentir un peu l'entraînement)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# # Paramètres
# N_MODES = 4
# N_PHOTONS = 2
# INTERMEDIATE_DIM = 4 # Sortie du circuit quantique
# FINAL_OUT = 1        # Par exemple pour de la régression ou un score unique
# HIDDEN = [16, 8]     # Couches cachées du mapping

# # Instanciation
# my_hybrid_model = HybridQMLModel(
#     q_input_size=N_MODES,
#     q_output_size=INTERMEDIATE_DIM,
#     nb_photons=N_PHOTONS,
#     nb_modes=N_MODES,
#     hidden_sizes=HIDDEN,
#     final_output_size=FINAL_OUT
# )

# # Test avec un batch
# dummy_data = torch.randn(5, N_MODES)
# final_result = my_hybrid_model(dummy_data)

# print(f"Forme de la sortie finale : {final_result.shape}") # Devrait être [5, 1]




# --- 1. FONCTION DE L'AGENT CIBLE ---
def rl_agent_forward(state_tensor, generated_weights, input_dim=4, output_dim=2):
    """
    L'agent RL (sans paramètres propres). 
    Il utilise les poids générés par le modèle quantique hybride.
    """
    # On reformate les 8 valeurs générées en une matrice 2x4
    weight_matrix = generated_weights.view(output_dim, input_dim)
    
    # Produit matriciel : État -> Logits d'action
    # F.linear garantit que les gradients remonteront vers weight_matrix
    logits = F.linear(state_tensor, weight_matrix)
    return logits


# --- 2. FONCTION POUR LES RÉCOMPENSES (Discounted Return) ---
def compute_discounted_returns(rewards, gamma=0.99):
    """Calcule le retour cumulé avec un facteur d'escompte (gamma)."""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    
    # Normalisation pour stabiliser l'apprentissage
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    return returns


# --- 3. LA BOUCLE D'ENTRAÎNEMENT QTRL ---
def train_cartpole_qtrl(hybrid_model, num_episodes=1000, learning_rate=0.01):
    # Initialisation de l'environnement Gym
    env = gym.make('CartPole-v1')
    
    # Paramètres de l'environnement
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n             # 2
    total_weights_needed = state_dim * action_dim # 8
    
    # Optimiseur (il met à jour le circuit quantique ET le mapping classique)
    optimizer = optim.Adam(hybrid_model.parameters(), lr=learning_rate)
    
    # Entrée factice pour le circuit quantique (si autonome)
    # dummy_q_input = torch.ones(1, hybrid_model.nb_modes)
    
    print("🚀 Début de l'entraînement Quantum-Train RL sur CartPole...")
    
    for episode in range(num_episodes):
        # Reset de l'environnement (Gymnasium renvoie un tuple state, info)
        state, _ = env.reset()
        
        log_probs = []
        rewards = []
        
        # ==========================================
        # ÉTAPE 1 : LE CIRCUIT QUANTIQUE GÉNÈRE LES POIDS
        # ==========================================
        # On génère tous les poids pour cet épisode en UNE SEULE passe
        raw_weights = hybrid_model()[0] 
        
        # On s'assure de prendre exactement les 8 poids nécessaires
        # (Optionnel : appliquer un torch.tanh(raw_weights) ici si le MappingModel n'en a pas)
        episode_weights = raw_weights[:total_weights_needed]
        
        # ==========================================
        # ÉTAPE 2 : INTERACTION AVEC CARTPOLE
        # ==========================================
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            # L'agent évalue l'état avec les poids quantiques
            logits = rl_agent_forward(state_tensor, episode_weights, state_dim, action_dim)
            
            # Échantillonnage de l'action selon les probabilités
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            
            # On effectue l'action dans Gym
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # On sauvegarde pour la Loss
            log_probs.append(action_dist.log_prob(action))
            rewards.append(reward)
            
            state = next_state
            
        # ==========================================
        # ÉTAPE 3 : RÉTROPROPAGATION ET MISE À JOUR
        # ==========================================
        # Calcul des retours actualisés
        discounted_returns = compute_discounted_returns(rewards)
        
        # Calcul de la Loss (Policy Gradient)
        policy_loss = []
        for log_prob, R in zip(log_probs, discounted_returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        optimizer.zero_grad()
        policy_loss.backward()
        
        # Clipping des gradients pour éviter l'instabilité
        torch.nn.utils.clip_grad_norm_(hybrid_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Affichage
        total_reward = sum(rewards)
        # if episode % 20 == 0:
        print(f"Épisode {episode:4d} | Récompense : {total_reward:5.1f} | Loss : {policy_loss.item():.4f}")
            
        # CartPole est considéré comme "résolu" si on tient 500 étapes
        # if total_reward >= 500:
        #     print(f"🎉 CartPole résolu à l'épisode {episode} !")
        #     break

    env.close()
    return total_reward



