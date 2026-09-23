import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.datasets import QM9
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch.utils.tensorboard import SummaryWriter
import time
import os
from WGAN import Generator, Discriminator, gradient_penalty
from quantum_layer import PhotonicRewardModule
# Assure-toi d'avoir sauvegardé la fonction RDKit dans un fichier, ex: rdkit_utils.py
from chemical_reward import evaluate_and_reward 
from sklearn.decomposition import PCA
import numpy as np
# =========================================================================
# 1. PRÉPARATION DES DONNÉES (QM9)
# =========================================================================
print("--- Chargement de QM9 ---")
dataset = QM9(root='data/QM9')

N_nodes = 9 
N_atoms = 5
N_bonds = 5

node_tensors, adj_tensors = [], []
for i in range(10000): # Échantillon pour prototypage
    data = dataset[i]
    if data.num_nodes > N_nodes: 
        continue

    x_heavy = data.x[:, 1:5]
    x_dense, mask = to_dense_batch(x_heavy, max_num_nodes=N_nodes)
    x_dense = x_dense[0]
    
    is_empty = (~mask[0]).float().unsqueeze(1)
    x_final = torch.cat([is_empty, x_dense], dim=1) 
    
    adj_dense = to_dense_adj(data.edge_index, edge_attr=data.edge_attr, max_num_nodes=N_nodes)[0] 
    has_bond = adj_dense.sum(dim=-1)
    no_bond = (has_bond == 0).float().unsqueeze(-1)
    no_bond.diagonal().fill_(1.)
    
    adj_final = torch.cat([no_bond, adj_dense], dim=-1) 
    
    node_tensors.append(x_final)
    adj_tensors.append(adj_final)

dataloader = DataLoader(TensorDataset(torch.stack(node_tensors), torch.stack(adj_tensors)), batch_size=64, shuffle=True)

# =========================================================================
# 2. INITIALISATION DES MODÈLES ET TENSORBOARD
# =========================================================================


# =========================================================================
# 2. PRÉPARATION DU DATASET QM9
# =========================================================================
print("--- Chargement de QM9 ---")
dataset = QM9(root='data/QM9')

N_nodes = 9 
N_atoms = 5   # Vide, C, N, O, F
N_bonds = 5   # Aucune, Simple, Double, Triple, Aromatique

node_tensors, adj_tensors = [], []
print("Extraction d'un échantillon de 5000 molécules valides...")
target_count = 5000

for data in dataset:
    if data.num_nodes > N_nodes:
        continue
        
    x_heavy = data.x[:, 1:5]
    x_dense, mask = to_dense_batch(x_heavy, max_num_nodes=N_nodes)
    x_dense = x_dense[0]
    
    is_empty = (~mask[0]).float().unsqueeze(1)
    x_final = torch.cat([is_empty, x_dense], dim=1) 
    
    adj_dense = to_dense_adj(data.edge_index, edge_attr=data.edge_attr, max_num_nodes=N_nodes)[0] 
    has_bond = adj_dense.sum(dim=-1)
    no_bond = (has_bond == 0).float().unsqueeze(-1)
    no_bond.diagonal().fill_(1.)
    
    adj_final = torch.cat([no_bond, adj_dense], dim=-1) 
    
    node_tensors.append(x_final)
    adj_tensors.append(adj_final)
    
    if len(node_tensors) >= target_count:
        break

dataloader = DataLoader(TensorDataset(torch.stack(node_tensors), torch.stack(adj_tensors)), batch_size=32, shuffle=True)
# Initialisation de TensorBoard
writer = SummaryWriter("runs/MolGAN_QRL_Experiment_1")
# =========================================================================
# 3. INITIALISATION DES MODÈLES (G, D, Q)
# =========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Entraînement sur : {device}")

# Architecture GAN
z_dim = 8
g_conv_dims = [128, 256, 512]
d_conv_dims = ([128, 64], 128, [128, 64])

G = Generator(g_conv_dims, z_dim, N_nodes, N_bonds, N_atoms, 0.0).to(device)
D = Discriminator(d_conv_dims, m_dim=N_atoms, b_dim=N_bonds-1, dropout_rate=0.0).to(device)

# --- Calcul de l'ACP sur les données QM9 filtrées ---
print("Calcul de l'ACP pour le module quantique...")
# On récupère les tenseurs valides pour fitter l'ACP
sample_adj = torch.stack(adj_tensors).flatten(start_dim=1).cpu().numpy()
sample_nodes = torch.stack(node_tensors).flatten(start_dim=1).cpu().numpy()
sample_x = np.concatenate([sample_adj, sample_nodes], axis=1)

pca = PCA(n_components=6)
pca.fit(sample_x)

# Instanciation du Module Photonique avec les matrices ajustées
Q = PhotonicRewardModule(
    pca_components=pca.components_,
    pca_mean=pca.mean_,
    num_atoms=N_nodes, 
    num_bonds=N_bonds, 
    num_atom_types=N_atoms, 
    nb_modes=6
).to(device)

# =========================================================================
# 4. OPTIMISEURS ET BOUCLE D'ENTRAÎNEMENT
# =========================================================================
opt_g = torch.optim.RMSprop(G.parameters(), lr=1e-4)
opt_d = torch.optim.RMSprop(D.parameters(), lr=1e-4)
opt_q = torch.optim.RMSprop(Q.parameters(), lr=1e-4)


epochs = 200
n_critic = 5
lambda_gp = 10
tau = 1.0

# Paramètres de stabilisation
warmup_epochs = 30
ramp_epochs = 50
lambda_max = 0.5
ema_beta = 0.9
reward_ema = 0.0

print("\n--- Début de l'entraînement MolGAN-QRL ---")
global_step = 0

for epoch in range(1, epochs + 1):
    start_time = time.time()
    
    # --- DYNAMIQUE ALPHA (Rampe linéaire) ---
    if epoch <= warmup_epochs:
        current_lambda = 0.0
    else:
        progress = (epoch - warmup_epochs) / ramp_epochs
        current_lambda = min(lambda_max, progress * lambda_max)
        
    for batch_nodes, batch_adj in dataloader:
        batch_size = batch_nodes.size(0)
        batch_nodes, batch_adj = batch_nodes.to(device), batch_adj.to(device)

        # ---------------------------------------------------
        # ÉTAPE A : ENTRAÎNEMENT DU DISCRIMINATEUR WGAN (x5)
        # ---------------------------------------------------
        for _ in range(n_critic):
            opt_d.zero_grad()
            z = torch.randn(batch_size, z_dim).to(device)
            
            with torch.no_grad():
                edges_logits, nodes_logits = G(z)
                fake_adj = F.gumbel_softmax(edges_logits, tau=tau, hard=True, dim=-1)
                fake_nodes = F.gumbel_softmax(nodes_logits, tau=tau, hard=True, dim=-1)
            
            real_val = D(batch_adj, None, batch_nodes)
            fake_val = D(fake_adj, None, fake_nodes)
            
            gp = gradient_penalty(D, batch_nodes, batch_adj, fake_nodes, fake_adj, device)
            d_loss = -torch.mean(real_val) + torch.mean(fake_val) + (lambda_gp * gp)
            
            d_loss.backward()
            opt_d.step()

        # ---------------------------------------------------
        # ÉTAPE B : ENTRAÎNEMENT DU MODULE PHOTONIQUE
        # ---------------------------------------------------
        # ---------------------------------------------------
        # ÉTAPE B : ENTRAÎNEMENT DU MODULE PHOTONIQUE (Équation 5)
        # ---------------------------------------------------
        opt_q.zero_grad()
        with torch.no_grad():
            # 1. Évaluation des graphes générés (Fake)
            z = torch.randn(batch_size, z_dim).to(device)
            edges_logits, nodes_logits = G(z)
            fake_adj_hard = F.gumbel_softmax(edges_logits, tau=tau, hard=True, dim=-1)
            fake_nodes_hard = F.gumbel_softmax(nodes_logits, tau=tau, hard=True, dim=-1)
            target_rc_fake, val_ratio, uniq_ratio = evaluate_and_reward(fake_adj_hard, fake_nodes_hard)
            
            # 2. Évaluation des graphes réels (Real) - Nécessaire pour ancrer le QRL
            target_rc_real, _, _ = evaluate_and_reward(batch_adj, batch_nodes)

        pred_rq_fake = Q(fake_adj_hard, fake_nodes_hard)
        pred_rq_real = Q(batch_adj, batch_nodes)
        
        # Équation 5 : L1 Loss (Absolue) sur les Réels ET les Fakes
        q_loss = torch.mean(torch.abs(pred_rq_real - target_rc_real) + torch.abs(pred_rq_fake - target_rc_fake))
        q_loss.backward()
        opt_q.step()
        # ---------------------------------------------------
        # ÉTAPE C : ENTRAÎNEMENT DU GÉNÉRATEUR (WGAN + REWARD EMA)
        # ---------------------------------------------------
# ---------------------------------------------------
        # ÉTAPE C : ENTRAÎNEMENT DU GÉNÉRATEUR (Gradients Rétablis)
        # ---------------------------------------------------
        opt_g.zero_grad()
        z = torch.randn(batch_size, z_dim).to(device)
        edges_logits, nodes_logits = G(z)
        
        fake_adj_soft = F.gumbel_softmax(edges_logits, tau=tau, hard=False, dim=-1)
        fake_nodes_soft = F.gumbel_softmax(nodes_logits, tau=tau, hard=False, dim=-1)
        
        fake_val = D(fake_adj_soft, None, fake_nodes_soft)
        g_loss_wgan = -torch.mean(fake_val) 
        
        # Le tenseur GARDE son gradient (Pas de .item() ici)
        pred_rq_soft = Q(fake_adj_soft, fake_nodes_soft)
        instant_reward_tensor = torch.mean(pred_rq_soft)
        
        # Mise à jour de l'EMA uniquement pour le logging visuel
        if global_step == 0:
            reward_ema = instant_reward_tensor.item()
        else:
            reward_ema = ema_beta * reward_ema + (1.0 - ema_beta) * instant_reward_tensor.item()
        
        # Combinaison convexe (Équation 6) avec le tenseur intact pour la rétropropagation
        g_loss_total = (1.0 - current_lambda) * g_loss_wgan + current_lambda * (-instant_reward_tensor)
        
        g_loss_total.backward()
        opt_g.step()
        # ---------------------------------------------------
        # ÉTAPE D : LOGGING TENSORBOARD
        # ---------------------------------------------------
        writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step)
        writer.add_scalar('Loss/Generator_Total', g_loss_total.item(), global_step)
        writer.add_scalar('Loss/Generator_WGAN', g_loss_wgan.item(), global_step)
        writer.add_scalar('Photonic/MSE_Loss', q_loss.item(), global_step)
        writer.add_scalar('Reward/Average_RDKit_Rc', target_rc_fake.mean().item(), global_step)
        writer.add_scalar('Reward/EMA_Quantum_Rq', reward_ema, global_step)
        writer.add_scalar('Metrics/Validity', val_ratio, global_step)
        writer.add_scalar('Metrics/Uniqueness', uniq_ratio, global_step)
        # Ligne corrigée : utilisation de current_lambda
        writer.add_scalar('Hyperparameters/Lambda_RL', current_lambda, global_step) 
        
        global_step += 1

    elapsed = time.time() - start_time
    # Ligne corrigée : affichage de current_lambda
    print(f"Epoch {epoch:03d}/{epochs} | D_Loss: {d_loss.item():.4f} | G_Loss: {g_loss_total.item():.4f} | Val: {val_ratio:.2f} | Uniq: {uniq_ratio:.2f} | Lambda: {current_lambda:.2f} | Temps: {elapsed:.2f}s")
    torch.save(G.state_dict(), 'generator_final.pth')