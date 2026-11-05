import torch
import torch.autograd as autograd
import torch.nn as nn
from lib.layers import GraphAggregation, GraphConvolution, MultiDenseLayer


class Generator(nn.Module):
    """
    MolGAN Generator network that transforms latent noise vectors into dense molecular
    adjacency and node feature matrices.
    """

    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout_rate):
        """
        Initializes the generator architecture layers.

        Args:
            conv_dims (list of int): Hidden dimensions for the multi-dense intermediate layers.
            z_dim (int): Dimension of the latent noise vector z.
            vertexes (int): Maximum number of nodes (atoms) per molecule.
            edges (int): Number of bond types.
            nodes (int): Number of atom species types.
            dropout_rate (float): Dropout probability.
        """
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
        """
        Forward pass generating symmetric adjacency logits and node categorical logits.

        Args:
            x (torch.Tensor): Latent noise tensor of shape (batch_size, z_dim).

        Returns:
            tuple of torch.Tensor:
                - edges_logits: Generated edge adjacency logits of shape (batch_size, vertexes, vertexes, edges).
                - nodes_logits: Generated node feature logits of shape (batch_size, vertexes, nodes).
        """
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
    """
    WGAN Discriminator network utilizing graph convolutions and graph aggregation
    to evaluate the realism of molecular graphs.
    """

    def __init__(self, conv_dim, m_dim, b_dim, dropout_rate=0.0):
        """
        Initializes the graph convolutional blocks and output linear layers of the discriminator.

        Args:
            conv_dim (tuple): Tuple containing (graph_conv_dim, aux_dim, linear_dim) configurations.
            m_dim (int): Number of atom types / node feature dimension.
            b_dim (int): Number of bond types considered in convolutions.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        self.activation_f = torch.nn.Tanh()
        graph_conv_dim, aux_dim, linear_dim = conv_dim
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
        self.output_layer = nn.Linear(linear_dim[-1], 1, bias=False)

    def forward(self, adj, hidden, node):
        """
        Forward pass evaluating molecular structures and outputting a scalar Wasserstein critic score.

        Args:
            adj (torch.Tensor): Adjacency tensor of shape (batch_size, num_nodes, num_nodes, total_bonds).
            hidden (torch.Tensor, optional): Hidden state tensor.
            node (torch.Tensor): Node feature matrix of shape (batch_size, num_nodes, m_dim).

        Returns:
            torch.Tensor: Critic score tensor of shape (batch_size, 1).
        """
        adj_conv = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        h_1 = self.gcn_layer(node, adj_conv, hidden)
        h = self.agg_layer(h_1, node, hidden)
        h = self.multi_dense_layer(h)
        output = self.output_layer(h)
        return output


def gradient_penalty(D, real_nodes, real_adj, fake_nodes, fake_adj, device):
    """
    Computes the WGAN-GP gradient penalty enforcing the 1-Lipschitz constraint
    on interpolated samples between real and fake molecules.

    Args:
        D (nn.Module): The discriminator / critic model instance.
        real_nodes (torch.Tensor): Real node feature tensor.
        real_adj (torch.Tensor): Real adjacency tensor.
        fake_nodes (torch.Tensor): Generated/fake node feature tensor.
        fake_adj (torch.Tensor): Generated/fake adjacency tensor.
        device (torch.device): Computation device (CPU or CUDA).

    Returns:
        torch.Tensor: Scalar gradient penalty loss term.
    """
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
