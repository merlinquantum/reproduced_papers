import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. GÉNÉRATEUR (Fidèle à l'original, sans forçage de symétrie)
# ==============================================================================
class Generator(nn.Module):
    def __init__(self, z_dim, vertexes, edges, nodes, hidden_dims=[128, 256, 512]):
        super().__init__()
        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes
        
        layers = []
        in_dim = z_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.Tanh()])
            in_dim = h_dim
        self.mlp = nn.Sequential(*layers)
        
        self.edge_head = nn.Linear(in_dim, vertexes * vertexes * edges)
        self.node_head = nn.Linear(in_dim, vertexes * nodes)

    def forward(self, z):
        h = self.mlp(z)
        edge_logits = self.edge_head(h).view(-1, self.vertexes, self.vertexes, self.edges)
        node_logits = self.node_head(h).view(-1, self.vertexes, self.nodes)
        
        # Le Gumbel-Softmax reste pur (encodage one-hot validé)
        edge_hat = F.gumbel_softmax(edge_logits, tau=1.0, hard=True, dim=-1)
        node_hat = F.gumbel_softmax(node_logits, tau=1.0, hard=True, dim=-1)
        return edge_hat, node_hat


# ==============================================================================
# 2. R-GCN (Avec Skip Connections)
# ==============================================================================
class ExactGraphConv(nn.Module):
    def __init__(self, in_features, out_features, num_edge_types):
        super().__init__()
        self.num_edge_types = num_edge_types
        self.edge_linears = nn.ModuleList([
            nn.Linear(in_features, out_features) for _ in range(num_edge_types - 1)
        ])
        self.self_loop = nn.Linear(in_features, out_features)

    def forward(self, annotations, edge_features):
        out = self.self_loop(annotations)
        for i in range(self.num_edge_types - 1):
            A_e = edge_features[:, :, :, i+1] # On saute la non-liaison
            hw = self.edge_linears[i](annotations)
            out = out + torch.matmul(A_e, hw)
        return torch.tanh(out)


# ==============================================================================
# 3. DISCRIMINATEUR (Gated Sum + Mini-Batch)
# ==============================================================================
class Discriminator(nn.Module):
    def __init__(self, vertexes, edges, nodes, conv_dims=[128, 128], dense_dims=[128, 64], batch_disc=True):
        super().__init__()
        self.batch_disc = batch_disc
        
        # GCN + Skip connections
        self.gcn_layers = nn.ModuleList()
        in_dim = nodes
        for out_dim in conv_dims:
            self.gcn_layers.append(ExactGraphConv(in_dim, out_dim, edges))
            in_dim = out_dim + nodes 
            
        # Gated Pooling
        agg_in = conv_dims[-1]
        agg_out = dense_dims[0]
        self.agg_i = nn.Linear(agg_in, agg_out)
        self.agg_j = nn.Linear(agg_in, agg_out)
        
        # Multi Dense
        self.mlp = nn.ModuleList()
        curr_dim = agg_out
        for u in dense_dims[1:]:
            self.mlp.append(nn.Linear(curr_dim, u))
            curr_dim = u
            
        # Mini-Batch Discrimination
        if self.batch_disc:
            mb_dim = max(curr_dim // 8, 1)
            # CORRECTION : mb_1 doit prendre agg_out (128) en entrée, pas curr_dim
            self.mb_1 = nn.Linear(agg_out, mb_dim)
            self.mb_2 = nn.Linear(mb_dim, mb_dim)
            curr_dim = curr_dim + mb_dim    
            
        self.classifier = nn.Linear(curr_dim, 1)

    def forward(self, edge_hat, node_hat):
        hidden_tensor = None
        
        # Convolutions
        for gcn in self.gcn_layers:
            annotations = torch.cat((hidden_tensor, node_hat), dim=-1) if hidden_tensor is not None else node_hat
            hidden_tensor = gcn(annotations, edge_hat)
            
        # Gated Aggregation
        i = torch.sigmoid(self.agg_i(hidden_tensor))
        j = torch.tanh(self.agg_j(hidden_tensor))
        outputs0 = torch.sum(i * j, dim=1)
        
        # Dense Layers
        out = outputs0
        for layer in self.mlp:
            out = torch.tanh(layer(out))
            
        # Batch Discrimination
        if self.batch_disc:
            out_b = torch.tanh(self.mb_1(outputs0))
            out_b = torch.tanh(self.mb_2(out_b.mean(dim=0, keepdim=True)))
            out_b = out_b.repeat(out.size(0), 1)
            out = torch.cat((out, out_b), dim=-1)
            
        return self.classifier(out), out


# ==============================================================================
# 4. LE CORRECTIF WGAN-GP (La clé de l'explosion)
# ==============================================================================
def compute_gradient_penalty(critic, real_edges, real_nodes, fake_edges, fake_nodes, device="cpu"):
    B = real_edges.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    alpha_nodes = alpha.squeeze(-1) 

    # IL EST OBLIGATOIRE DE FAIRE .detach() SUR LES DONNÉES GÉNÉRÉES
    interpolated_edges = (alpha * real_edges + (1 - alpha) * fake_edges.detach()).requires_grad_(True)
    interpolated_nodes = (alpha_nodes * real_nodes + (1 - alpha_nodes) * fake_nodes.detach()).requires_grad_(True)

    d_interpolates, _ = critic(interpolated_edges, interpolated_nodes)

    gradients = torch.autograd.grad(
        outputs=d_interpolates.sum(),
        inputs=[interpolated_edges, interpolated_nodes],
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )

    grad_edges = gradients[0].view(B, -1)
    grad_nodes = gradients[1].view(B, -1)
    all_grads = torch.cat([grad_edges, grad_nodes], dim=1)

    grad_norm = all_grads.norm(2, dim=1)
    return torch.mean((grad_norm - 1.0) ** 2)
# ==============================================================================
# BOUCLE D'ENTRAÎNEMENT WGAN
# ==============================================================================
import torch

# 1. Dimensions du problème
BATCH_SIZE = 16
Z_DIM = 32
VERTEXES = 9
EDGES = 4
NODES = 5

# 2. Instanciation des modèles (G et D doivent être définis au préalable)
G = Generator(z_dim=Z_DIM, vertexes=VERTEXES, edges=EDGES, nodes=NODES)
D = Discriminator(vertexes=VERTEXES, edges=EDGES, nodes=NODES)

optimizer_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))
lambda_gp = 10.0

# ==========================================================
# SIMULATION DU DATALOADER (Les vraies molécules)
# Dans la réalité, ceci vient de ton dataset (ex: QM9)
# ==========================================================
# (B, V, V, E) - Graphe fully connected factice
real_edges_indices = torch.randint(0, EDGES, (BATCH_SIZE, VERTEXES, VERTEXES))
real_edges = torch.nn.functional.one_hot(real_edges_indices, num_classes=EDGES).float()
# Rendre symétrique
real_edges = (torch.triu(real_edges.permute(0, 3, 1, 2)) + torch.triu(real_edges.permute(0, 3, 1, 2)).transpose(2,3)).permute(0, 2, 3, 1)

# (B, V, N) - Noeuds factices
real_nodes_indices = torch.randint(0, NODES, (BATCH_SIZE, VERTEXES))
real_nodes = torch.nn.functional.one_hot(real_nodes_indices, num_classes=NODES).float()
# ==========================================================

epochs = 10
n_critic = 5 # WGAN nécessite d'entraîner le critique plus souvent que le générateur

for epoch in range(epochs):
    # --- ENTRAÎNEMENT DU CRITIQUE (WGAN-GP) ---
    for _ in range(n_critic):
        optimizer_D.zero_grad()
        
        z = torch.randn(BATCH_SIZE, Z_DIM)
        fake_edges, fake_nodes = G(z)
        
        real_scores, _ = D(real_edges, real_nodes)
        fake_scores, _ = D(fake_edges.detach(), fake_nodes.detach())
        
        d_loss = torch.mean(fake_scores) - torch.mean(real_scores)
        gp = compute_gradient_penalty(D, real_edges, real_nodes, fake_edges, fake_nodes)
        
        total_d_loss = d_loss + lambda_gp * gp
        total_d_loss.backward()
        optimizer_D.step()

    # --- ENTRAÎNEMENT DU GÉNÉRATEUR ---
    optimizer_G.zero_grad()
    
    z = torch.randn(BATCH_SIZE, Z_DIM)
    fake_edges, fake_nodes = G(z)
    
    gen_fake_scores, _ = D(fake_edges, fake_nodes)
    g_loss_wgan = -torch.mean(gen_fake_scores)
    
    g_loss_wgan.backward()
    optimizer_G.step()
    
    print(f"Epoch {epoch+1} | D Loss: {total_d_loss.item():.4f} | G Loss: {g_loss_wgan.item():.4f}")