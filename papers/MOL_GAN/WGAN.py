import torch
import torch.autograd as autograd
import torch.nn as nn
from layers import GraphAggregation, GraphConvolution, MultiDenseLayer

# =========================================================================
# 1. IMPORT DES COUCHES ET DÉFINITION DES MODÈLES (Fidèle à MolGAN)
# =========================================================================


class Generator(nn.Module):
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout_rate):
        super().__init__()
        self.activation_f = torch.nn.Tanh()
        self.multi_dense_layer = MultiDenseLayer(z_dim, conv_dims, self.activation_f)
        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes
        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.multi_dense_layer(x)
        edges_logits = self.edges_layer(output).view(
            -1, self.edges, self.vertexes, self.vertexes
        )
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropout(nodes_logits.view(-1, self.vertexes, self.nodes))
        return edges_logits, nodes_logits


class Discriminator(nn.Module):
    def __init__(self, conv_dim, m_dim, b_dim, dropout_rate=0.0):
        super().__init__()
        self.activation_f = torch.nn.Tanh()
        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # m_dim = atomes, b_dim = liaisons
        self.gcn_layer = GraphConvolution(
            m_dim, graph_conv_dim, b_dim, False, 0, dropout_rate
        )
        self.agg_layer = GraphAggregation(
            graph_conv_dim[-1] + m_dim,
            aux_dim,
            self.activation_f,
            False,
            0,
            dropout_rate,
        )
        self.multi_dense_layer = MultiDenseLayer(
            aux_dim, linear_dim, self.activation_f, dropout_rate=dropout_rate
        )
        # self.output_layer = nn.Linear(linear_dim[-1], 1)
        self.output_layer = nn.Linear(linear_dim[-1], 1, bias=False)

    def forward(self, adj, hidden, node):
        # On ignore le canal 0 ("Pas de liaison") pour les convolutions
        adj_conv = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        h_1 = self.gcn_layer(node, adj_conv, hidden)
        h = self.agg_layer(h_1, node, hidden)
        h = self.multi_dense_layer(h)
        output = self.output_layer(h)
        return output


# # =========================================================================
# # 2. PRÉPARATION DU DATASET QM9
# # =========================================================================
# print("--- Chargement de QM9 ---")
# dataset = QM9(root='data/QM9')

# N_nodes = 9
# N_atoms = 5   # Vide, C, N, O, F
# N_bonds = 5   # Aucune, Simple, Double, Triple, Aromatique

# node_tensors, adj_tensors = [], []
# print("Extraction d'un échantillon (10 000 molécules pour la vitesse)...")
# for i in range(10000):
#     data = dataset[i]
#     if data.num_nodes > N_nodes: continue

#     x_heavy = data.x[:, 1:5]
#     x_dense, mask = to_dense_batch(x_heavy, max_num_nodes=N_nodes)
#     x_dense = x_dense[0]

#     is_empty = (~mask[0]).float().unsqueeze(1)
#     x_final = torch.cat([is_empty, x_dense], dim=1)

#     adj_dense = to_dense_adj(data.edge_index, edge_attr=data.edge_attr, max_num_nodes=N_nodes)[0]
#     has_bond = adj_dense.sum(dim=-1)
#     no_bond = (has_bond == 0).float().unsqueeze(-1)
#     no_bond.diagonal().fill_(1.)

#     adj_final = torch.cat([no_bond, adj_dense], dim=-1)

#     node_tensors.append(x_final)
#     adj_tensors.append(adj_final)

# dataloader = DataLoader(TensorDataset(torch.stack(node_tensors), torch.stack(adj_tensors)), batch_size=128, shuffle=True)

# # =========================================================================
# # 3. INITIALISATION (Paramètres MolGAN)
# # =========================================================================
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Entraînement sur : {device}")

# z_dim = 8
# g_conv_dims = [128, 256, 512]
# d_conv_dims = ([128, 64], 128, [128, 64])

# G = Generator(g_conv_dims, z_dim, N_nodes, N_bonds, N_atoms, 0.0).to(device)
# D = Discriminator(d_conv_dims, m_dim=N_atoms, b_dim=N_bonds-1, dropout_rate=0.0).to(device)

# opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
# opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))


# # =========================================================================
# # 4. BOUCLE D'ENTRAÎNEMENT EXACTE (MolGAN WGAN-GP)
# # =========================================================================
def gradient_penalty(D, real_nodes, real_adj, fake_nodes, fake_adj, device):
    alpha = torch.rand(real_nodes.size(0), 1, 1).to(device)
    alpha_adj = alpha.unsqueeze(-1)

    int_nodes = (alpha * real_nodes + ((1 - alpha) * fake_nodes)).requires_grad_(True)
    int_adj = (alpha_adj * real_adj + ((1 - alpha_adj) * fake_adj)).requires_grad_(True)

    d_interpolates = D(int_adj, None, int_nodes)

    fake = torch.ones(real_nodes.shape[0], 1).to(device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=(int_nodes, int_adj),
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )

    grad_nodes = gradients[0].view(gradients[0].size(0), -1)
    grad_adj = gradients[1].view(gradients[1].size(0), -1)
    grad_norm = torch.sqrt(
        torch.sum(grad_nodes**2, dim=1) + torch.sum(grad_adj**2, dim=1)
    )
    return ((grad_norm - 1) ** 2).mean()


# print("\n--- Début de l'entraînement WGAN ---")
# epochs = 50
# n_critic = 5
# lambda_gp = 10
# tau = 1.0

# for epoch in range(1, epochs + 1):
#     start_time = time.time()

#     for batch_nodes, batch_adj in dataloader:
#         batch_size = batch_nodes.size(0)
#         batch_nodes, batch_adj = batch_nodes.to(device), batch_adj.to(device)

#         # ---------------------------------------------------
#         # ÉTAPE A : ENTRAÎNEMENT DU CRITIQUE (x5)
#         # ---------------------------------------------------
#         for _ in range(n_critic):
#             opt_D.zero_grad()
#             z = torch.randn(batch_size, z_dim).to(device)

#             with torch.no_grad():
#                 edges_logits, nodes_logits = G(z)
#                 # Astuce MolGAN : Gumbel-Softmax pour avoir des probabilités "dures"
#                 fake_adj = F.gumbel_softmax(edges_logits, tau=tau, hard=True, dim=-1)
#                 fake_nodes = F.gumbel_softmax(nodes_logits, tau=tau, hard=True, dim=-1)

#             real_val = D(batch_adj, None, batch_nodes)
#             fake_val = D(fake_adj, None, fake_nodes)

#             gp = gradient_penalty(D, batch_nodes, batch_adj, fake_nodes, fake_adj)
#             d_loss = -torch.mean(real_val) + torch.mean(fake_val) + (lambda_gp * gp)

#             d_loss.backward()
#             opt_D.step()

#         # ---------------------------------------------------
#         # ÉTAPE B : ENTRAÎNEMENT DU GÉNÉRATEUR (x1)
#         # ---------------------------------------------------
#         opt_G.zero_grad()
#         z = torch.randn(batch_size, z_dim).to(device)
#         edges_logits, nodes_logits = G(z)

#         # Le générateur utilise Gumbel-Softmax mais conserve le gradient (hard=False)
#         fake_adj = F.gumbel_softmax(edges_logits, tau=tau, hard=False, dim=-1)
#         fake_nodes = F.gumbel_softmax(nodes_logits, tau=tau, hard=False, dim=-1)

#         fake_val = D(fake_adj, None, fake_nodes)

#         # Le but du Générateur : Pousser le Critique dans le positif
#         g_loss = -torch.mean(fake_val)

#         g_loss.backward()
#         opt_G.step()

#     if epoch % 5 == 0:
#         elapsed = time.time() - start_time
#         print(f"Epoch {epoch:03d}/{epochs} | D_Loss: {d_loss.item():.4f} | G_Loss: {g_loss.item():.4f} | Temps: {elapsed:.2f}s")
