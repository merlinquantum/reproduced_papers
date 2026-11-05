import torch
import torch.nn as nn
from torch.nn.modules.module import Module


class GraphConvolutionLayer(Module):
    """
    Performs a single graph convolution operation handling multiple bond/edge types.
    """

    def __init__(self, in_features, u, activation, edge_type_num, dropout_rate=0.0):
        """
        Initializes the graph convolution layer.

        Args:
            in_features (int): Number of input features per node.
            u (int): Number of output hidden units.
            activation (callable, optional): Activation function to apply.
            edge_type_num (int): Number of distinct bond/edge types.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        self.edge_type_num = edge_type_num
        self.u = u
        self.adj_list = nn.ModuleList()
        for _ in range(self.edge_type_num):
            self.adj_list.append(nn.Linear(in_features, u))
        self.linear_2 = nn.Linear(in_features, u)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, n_tensor, adj_tensor, h_tensor=None):
        """
        Forward pass for the graph convolution layer.

        Args:
            n_tensor (torch.Tensor): Node feature tensor of shape (batch_size, num_nodes, in_features).
            adj_tensor (torch.Tensor): Adjacency tensor of shape (batch_size, edge_type_num, num_nodes, num_nodes).
            h_tensor (torch.Tensor, optional): Optional hidden state tensor for residual connections.

        Returns:
            torch.Tensor: Processed node feature tensor of shape (batch_size, num_nodes, u).
        """
        if h_tensor is not None:
            annotations = torch.cat((n_tensor, h_tensor), -1)
        else:
            annotations = n_tensor

        output = torch.stack(
            [self.adj_list[i](annotations) for i in range(self.edge_type_num)], 1
        )
        output = torch.matmul(adj_tensor, output)
        out_sum = torch.sum(output, 1)
        out_linear_2 = self.linear_2(annotations)
        output = out_sum + out_linear_2
        output = self.activation(output) if self.activation is not None else output
        output = self.dropout(output)
        return output


class MultiGraphConvolutionLayers(Module):
    """
    Sequentially stacks multiple GraphConvolutionLayers to process graph structures deeply.
    """

    def __init__(
        self,
        in_features,
        units,
        activation,
        edge_type_num,
        with_features=False,
        f=0,
        dropout_rate=0.0,
    ):
        """
        Initializes the sequence of graph convolution layers.

        Args:
            in_features (int): Number of initial input node features.
            units (list of int): List specifying output dimensions for each convolutional layer.
            activation (callable): Activation function.
            edge_type_num (int): Number of bond/edge types.
            with_features (bool): Whether auxiliary node features are concatenated.
            f (int): Dimension of auxiliary node features.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        self.conv_nets = nn.ModuleList()
        self.units = units
        in_units = []
        if with_features:
            for _i in range(len(self.units)):
                in_units = [x + in_features for x in self.units]
            for u0, u1 in zip([in_features + f] + in_units[:-1], self.units):
                self.conv_nets.append(
                    GraphConvolutionLayer(
                        u0, u1, activation, edge_type_num, dropout_rate
                    )
                )
        else:
            for _i in range(len(self.units)):
                in_units = [x + in_features for x in self.units]
            for u0, u1 in zip([in_features] + in_units[:-1], self.units):
                self.conv_nets.append(
                    GraphConvolutionLayer(
                        u0, u1, activation, edge_type_num, dropout_rate
                    )
                )

    def forward(self, n_tensor, adj_tensor, h_tensor=None):
        """
        Forward pass passing inputs sequentially through all internal convolution layers.

        Args:
            n_tensor (torch.Tensor): Node feature tensor.
            adj_tensor (torch.Tensor): Adjacency tensor.
            h_tensor (torch.Tensor, optional): Optional hidden tensor.

        Returns:
            torch.Tensor: Final hidden representation tensor after all convolutions.
        """
        hidden_tensor = h_tensor
        for conv_idx in range(len(self.units)):
            hidden_tensor = self.conv_nets[conv_idx](
                n_tensor, adj_tensor, hidden_tensor
            )
        return hidden_tensor


class GraphConvolution(Module):
    """
    High-level wrapper module initializing and executing multi-layer graph convolutions with Tanh activation.
    """

    def __init__(
        self,
        in_features,
        graph_conv_units,
        edge_type_num,
        with_features=False,
        f_dim=0,
        dropout_rate=0.0,
    ):
        """
        Initializes the wrapper graph convolution network.

        Args:
            in_features (int): Input feature size.
            graph_conv_units (list of int): Layer dimensionalities.
            edge_type_num (int): Number of edge types.
            with_features (bool): Flag for auxiliary features.
            f_dim (int): Auxiliary feature dimension.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.in_features = in_features
        self.graph_conv_units = graph_conv_units
        self.activation_f = torch.nn.Tanh()
        self.multi_graph_convolution_layers = MultiGraphConvolutionLayers(
            in_features,
            self.graph_conv_units,
            self.activation_f,
            edge_type_num,
            with_features,
            f_dim,
            dropout_rate,
        )

    def forward(self, n_tensor, adj_tensor, h_tensor=None):
        """
        Forward pass executing the multi-layer graph convolution sequence.

        Args:
            n_tensor (torch.Tensor): Node feature tensor.
            adj_tensor (torch.Tensor): Adjacency tensor.
            h_tensor (torch.Tensor, optional): Hidden tensor.

        Returns:
            torch.Tensor: Output hidden tensor.
        """
        output = self.multi_graph_convolution_layers(n_tensor, adj_tensor, h_tensor)
        return output


class GraphConvolution2(Module):
    """
    Alternative two-step graph convolution implementation using Einstein summation (einsum).
    """

    def __init__(self, in_features, out_feature_list, b_dim, dropout):
        """
        Initializes alternative graph convolution components.

        Args:
            in_features (int): Input feature size.
            out_feature_list (list of int): Two output dimensions for the sequential steps.
            b_dim (int): Bond dimension / number of edge types.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.in_features = in_features
        self.out_feature_list = out_feature_list

        self.linear1 = nn.Linear(in_features, out_feature_list[0])
        self.linear2 = nn.Linear(out_feature_list[0], out_feature_list[1])

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, adj, activation=None):
        """
        Forward pass utilizing einsum contractions across batch, node, and edge dimensions.

        Args:
            inputs (torch.Tensor): Input feature tensor (e.g., shape batch_size x num_nodes x num_nodes).
            adj (torch.Tensor): Adjacency tensor (e.g., shape batch_size x b_dim x num_nodes x num_nodes).
            activation (callable, optional): Activation function.

        Returns:
            torch.Tensor: Convolution output tensor.
        """
        hidden = torch.stack([self.linear1(inputs) for _ in range(adj.size(1))], 1)
        hidden = torch.einsum("bijk,bikl->bijl", (adj, hidden))
        hidden = torch.sum(hidden, 1) + self.linear1(inputs)
        hidden = activation(hidden) if activation is not None else hidden
        hidden = self.dropout(hidden)

        output = torch.stack([self.linear2(hidden) for _ in range(adj.size(1))], 1)
        output = torch.einsum("bijk,bikl->bijl", (adj, output))
        output = torch.sum(output, 1) + self.linear2(hidden)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)

        return output


class GraphAggregation(Module):
    """
    Aggregates node-level representations into a graph-level vector using an attention-like mechanism.
    """

    def __init__(
        self,
        in_features,
        aux_units,
        activation,
        with_features=False,
        f_dim=0,
        dropout_rate=0.0,
    ):
        """
        Initializes gating/attention layers for aggregation.

        Args:
            in_features (int): Input feature dimensions.
            aux_units (int): Hidden units for aggregation layers.
            activation (callable): Activation function.
            with_features (bool): Whether auxiliary features are included.
            f_dim (int): Auxiliary feature dimension.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        self.with_features = with_features
        self.activation = activation
        if self.with_features:
            self.i = nn.Sequential(
                nn.Linear(in_features + f_dim, aux_units), nn.Sigmoid()
            )
            j_layers = [nn.Linear(in_features + f_dim, aux_units)]
            if self.activation is not None:
                j_layers.append(self.activation)
            self.j = nn.Sequential(*j_layers)
        else:
            self.i = nn.Sequential(nn.Linear(in_features, aux_units), nn.Sigmoid())
            j_layers = [nn.Linear(in_features, aux_units)]
            if self.activation is not None:
                j_layers.append(self.activation)
            self.j = nn.Sequential(*j_layers)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, n_tensor, out_tensor, h_tensor=None):
        """
        Forward pass combining node annotations with sigmoid gating and summing over nodes.

        Args:
            n_tensor (torch.Tensor): Initial node features.
            out_tensor (torch.Tensor): Convoluted node features.
            h_tensor (torch.Tensor, optional): Hidden state tensor.

        Returns:
            torch.Tensor: Graph-level aggregated feature vector.
        """
        if h_tensor is not None:
            annotations = torch.cat((out_tensor, h_tensor, n_tensor), -1)
        else:
            annotations = torch.cat((out_tensor, n_tensor), -1)

        i = self.i(annotations)
        j = self.j(annotations)
        output = torch.sum(torch.mul(i, j), 1)
        if self.activation is not None:
            output = self.activation(output)
        output = self.dropout(output)

        return output


class GraphAggregation2(Module):
    """
    Alternative aggregation mechanism utilizing parallel Sigmoid and Tanh gating layers.
    """

    def __init__(self, in_features, out_features, b_dim, dropout):
        """
        Initializes sigmoid and tanh linear sequences.

        Args:
            in_features (int): Input feature size.
            out_features (int): Output feature size.
            b_dim (int): Bond dimension.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.sigmoid_linear = nn.Sequential(
            nn.Linear(in_features + b_dim, out_features), nn.Sigmoid()
        )
        self.tanh_linear = nn.Sequential(
            nn.Linear(in_features + b_dim, out_features), nn.Tanh()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, activation):
        """
        Forward pass scaling tanh representations with sigmoid gates and reducing across nodes.

        Args:
            inputs (torch.Tensor): Input tensor.
            activation (callable, optional): Optional activation function.

        Returns:
            torch.Tensor: Aggregated graph-level vector.
        """
        i = self.sigmoid_linear(inputs)
        j = self.tanh_linear(inputs)
        output = torch.sum(torch.mul(i, j), 1)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)

        return output


class MultiDenseLayer(Module):
    """
    A utility container for sequential fully-connected (dense) layers with dropout and activation.
    """

    def __init__(self, aux_unit, linear_units, activation=None, dropout_rate=0.0):
        """
        Initializes a stack of dense layers.

        Args:
            aux_unit (int): Input dimension size for the first layer.
            linear_units (list of int): Dimensions for successive dense layers.
            activation (callable, optional): Activation function applied after layers.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        layers = []
        for c0, c1 in zip([aux_unit] + linear_units[:-1], linear_units):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Dropout(dropout_rate))
            if activation is not None:
                layers.append(activation)
        self.linear_layer = nn.Sequential(*layers)

    def forward(self, inputs):
        """
        Forward pass through the multi-dense layer stack.

        Args:
            inputs (torch.Tensor): Input feature tensor.

        Returns:
            torch.Tensor: Processed output tensor.
        """
        h = self.linear_layer(inputs)
        return h
