"""GRIT models and graph-convolution baseline used by the reproduction.

GRIT (Ma et al. 2023) is a fairly heavy model — multiple attention heads, an
RWP-aware update rule, EGT-style edge attention. We re-implement the parts
relevant to consuming a positional encoding tensor of shape (B, N, N, K) and
producing a graph-level prediction, without depending on PyTorch Geometric.

``GRIT`` is a dense-batch adaptation of the original GRIT/GraphGPS layer. It
keeps the original node-pair update, signed-square-root edge modulation,
edge-enhanced values, and degree scaling while avoiding a GraphGym dependency.
``GRITLite`` remains available for inexpensive smoke tests.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalFeatureEncoder(nn.Module):
    """Embed and sum one or more categorical feature columns.

    Parameters
    ----------
    vocabulary_sizes : tuple[int, ...]
        Number of categories in each input feature column.
    output_dim : int
        Dimension of the summed embedding.

    Raises
    ------
    ValueError
        If the input column count or a categorical value is invalid.
    """

    def __init__(self, vocabulary_sizes: tuple[int, ...], output_dim: int):
        super().__init__()
        if not vocabulary_sizes:
            raise ValueError("categorical encoders require at least one vocabulary")
        self.vocabulary_sizes = vocabulary_sizes
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, output_dim) for size in vocabulary_sizes]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != len(self.embeddings):
            raise ValueError(
                f"expected {len(self.embeddings)} categorical columns, "
                f"got {features.shape[-1]}"
            )
        encoded_features = []
        for column, (embedding, vocabulary_size) in enumerate(
            zip(self.embeddings, self.vocabulary_sizes)
        ):
            values = features[..., column].long()
            if values.numel() and (values.min() < 0 or values.max() >= vocabulary_size):
                raise ValueError(
                    f"categorical column {column} contains a value outside "
                    f"[0, {vocabulary_size - 1}]"
                )
            encoded_features.append(embedding(values))
        return torch.stack(encoded_features, dim=0).sum(dim=0)


class EdgeAttnLayer(nn.Module):
    """One transformer layer with edge-feature-modulated attention.

    Attention scores are
        a_{ij} = (Q_i · K_j) / sqrt(d) + W_e · e_{ij}
    where ``e_{ij}`` is the PE edge feature. After softmax, values are mixed
    with a per-edge gate term inspired by GRIT.
    """

    def __init__(
        self, dim: int, edge_dim: int, num_heads: int = 4, dropout: float = 0.0
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        # Edge-attention bias from PE features.
        self.edge_bias = nn.Linear(edge_dim, num_heads)
        # Per-edge gate that multiplies value contributions.
        self.edge_gate = nn.Linear(edge_dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, e: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Args:
        x: (B, N, D) node features.
        e: (B, N, N, K) edge features (positional encoding).
        mask: (B, N) boolean valid-node mask.
        """
        B, N, D = x.shape
        H = self.num_heads
        Hd = self.head_dim
        q = self.q(x).view(B, N, H, Hd).permute(0, 2, 1, 3)  # (B, H, N, Hd)
        k = self.k(x).view(B, N, H, Hd).permute(0, 2, 1, 3)
        v = self.v(x).view(B, N, H, Hd).permute(0, 2, 1, 3)
        # Attention scores.
        scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(Hd)
        # Add edge-feature bias.
        e_bias = self.edge_bias(e).permute(0, 3, 1, 2)  # (B, H, N, N)
        scores = scores + e_bias
        # Mask out invalid (j) nodes.
        mask_j = mask.view(B, 1, 1, N)
        scores = scores.masked_fill(~mask_j, float("-inf"))
        # And invalid (i) rows for stability.
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # Edge gating: multiplies each attention link before mixing values.
        gate = torch.sigmoid(self.edge_gate(e)).permute(0, 3, 1, 2)
        gated_attn = attn * gate
        out = (
            torch.einsum("bhij,bhjd->bhid", gated_attn, v)
            .permute(0, 2, 1, 3)
            .reshape(B, N, D)
        )
        out = self.out(out)
        # Residual + norms.
        x = self.norm1(x + self.dropout(out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        # Zero out padded nodes so they don't leak through pooling.
        x = x * mask.unsqueeze(-1).float()
        return x


class GRITLite(nn.Module):
    """Minimal GRIT-style encoder over (A, PE) → graph- or node-level output.

    Inputs:
        A:    (B, Nmax, Nmax) adjacency  (used only to derive node degree
              as the trivial node feature when no other features are passed).
        PE:   (B, Nmax, Nmax, K) edge positional encoding tensor.
        mask: (B, Nmax) boolean node mask.

    Heads:
        ``head="graph_class"`` → (B, num_classes) logits via mean pool.
        ``head="graph_reg"``   → (B,) scalar via mean pool.
        ``head="node_class"``  → (B, Nmax, num_classes) node logits.
        ``head="node_feat"``   → (B, Nmax, dim) node features (no pooling).
    """

    def __init__(
        self,
        edge_dim: int,
        node_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        num_classes: int = 2,
        head: str = "graph_class",
        node_in_dim: int = 1,
        edge_in_dim: int = 0,
        node_feature_type: str = "continuous",
        edge_feature_type: str = "continuous",
        node_vocab_sizes: tuple[int, ...] = (),
        edge_vocab_sizes: tuple[int, ...] = (),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.head_type = head
        self.edge_in_dim = edge_in_dim
        if node_feature_type == "continuous":
            self.node_embed = nn.Linear(node_in_dim, node_dim)
        elif node_feature_type == "categorical":
            self.node_embed = CategoricalFeatureEncoder(node_vocab_sizes, node_dim)
        else:
            raise ValueError(f"unknown node feature type: {node_feature_type}")
        if edge_in_dim == 0:
            self.edge_feature_encoder = None
        elif edge_feature_type == "continuous":
            self.edge_feature_encoder = nn.Linear(edge_in_dim, edge_dim, bias=False)
        elif edge_feature_type == "categorical":
            self.edge_feature_encoder = CategoricalFeatureEncoder(
                edge_vocab_sizes, edge_dim
            )
        else:
            raise ValueError(f"unknown edge feature type: {edge_feature_type}")
        self.layers = nn.ModuleList(
            [
                EdgeAttnLayer(node_dim, edge_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(depth)
            ]
        )
        if head == "graph_class":
            self.out = nn.Linear(node_dim, num_classes)
        elif head == "graph_reg":
            self.out = nn.Linear(node_dim, 1)
        elif head == "node_class":
            self.out = nn.Linear(node_dim, num_classes)
        elif head == "node_feat":
            self.out = nn.Identity()
        else:
            raise ValueError(f"unknown head: {head}")

    def forward(
        self,
        PE: torch.Tensor,
        mask: torch.Tensor,
        node_features: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, _ = mask.shape if mask.dim() == 3 else (*mask.shape, 0)
        if node_features is None:
            # Use a trivial all-ones node feature (matches the "uniform" init
            # used by the paper for the synthetic experiments).
            node_features = mask.float().unsqueeze(-1)
        x = self.node_embed(node_features)
        edge_inputs = PE
        if self.edge_feature_encoder is not None:
            if edge_features is None:
                raise ValueError("this model requires edge features")
            if edge_features.shape[-1] != self.edge_in_dim:
                raise ValueError(
                    f"expected {self.edge_in_dim} edge feature columns, "
                    f"got {edge_features.shape[-1]}"
                )
            if edge_mask is None:
                raise ValueError("edge_mask is required when edge features are used")
            encoded_edge_features = self.edge_feature_encoder(edge_features)
            encoded_edge_features = encoded_edge_features * edge_mask.unsqueeze(-1)
            edge_inputs = edge_inputs + encoded_edge_features
        for layer in self.layers:
            x = layer(x, edge_inputs, mask)
        if self.head_type == "graph_class":
            denom = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = x.sum(dim=1) / denom
            return self.out(pooled)
        if self.head_type == "graph_reg":
            denom = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = x.sum(dim=1) / denom
            return self.out(pooled).squeeze(-1)
        return self.out(x)


def _masked_batch_norm(
    features: torch.Tensor, valid_entries: torch.Tensor, norm: nn.BatchNorm1d
) -> torch.Tensor:
    """Apply batch normalization only to real nodes or node pairs."""
    normalized = torch.zeros_like(features)
    normalized[valid_entries] = norm(features[valid_entries])
    return normalized


class GRITAttentionLayer(nn.Module):
    """Apply one faithful GRIT node and node-pair update.

    This is the dense equivalent of ``GritTransformerLayer`` in the original
    GRIT repository. For target node ``i`` and source node ``j``, the layer
    constructs ``Q_i + K_j``, modulates it with learned pair weights and
    biases, and updates both node and pair representations.

    Parameters
    ----------
    dim : int
        Hidden node and node-pair dimension.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout applied to node, edge, and feed-forward updates.
    attention_dropout : float
        Dropout applied only to normalized attention coefficients.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        attention_dropout: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("GRIT hidden dimension must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Linear(dim, dim, bias=True)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.pair_projection = nn.Linear(dim, 2 * dim, bias=True)
        self.attention_projection = nn.Parameter(torch.empty(self.head_dim, num_heads))
        self.edge_value_projection = nn.Parameter(
            torch.empty(self.head_dim, num_heads, self.head_dim)
        )
        self.node_output = nn.Linear(dim, dim)
        self.pair_output = nn.Linear(dim, dim)
        self.degree_coefficients = nn.Parameter(torch.empty(1, 1, dim, 2))
        self.node_norm1 = nn.BatchNorm1d(dim)
        self.pair_norm = nn.BatchNorm1d(dim)
        self.node_feed_forward1 = nn.Linear(dim, 2 * dim)
        self.node_feed_forward2 = nn.Linear(2 * dim, dim)
        self.node_norm2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        nn.init.xavier_normal_(self.query.weight)
        nn.init.xavier_normal_(self.key.weight)
        nn.init.xavier_normal_(self.value.weight)
        nn.init.xavier_normal_(self.pair_projection.weight)
        nn.init.xavier_normal_(self.attention_projection)
        nn.init.xavier_normal_(self.edge_value_projection)
        nn.init.xavier_normal_(self.degree_coefficients)

    def forward(
        self,
        node_states: torch.Tensor,
        pair_states: torch.Tensor,
        node_mask: torch.Tensor,
        log_degree: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update node and node-pair states.

        Parameters
        ----------
        node_states : torch.Tensor
            Node tensor with shape ``(batch, nodes, dim)``.
        pair_states : torch.Tensor
            Dense node-pair tensor with shape ``(batch, nodes, nodes, dim)``.
        node_mask : torch.Tensor
            Boolean mask identifying real nodes.
        log_degree : torch.Tensor
            ``log(degree + 1)`` for every real node.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated node and node-pair tensors.
        """
        batch_size, num_nodes, _ = node_states.shape
        query = self.query(node_states).view(
            batch_size, num_nodes, self.num_heads, self.head_dim
        )
        key = self.key(node_states).view(
            batch_size, num_nodes, self.num_heads, self.head_dim
        )
        value = self.value(node_states).view(
            batch_size, num_nodes, self.num_heads, self.head_dim
        )
        pair_weight, pair_bias = self.pair_projection(pair_states).chunk(2, dim=-1)
        pair_weight = pair_weight.view(
            batch_size, num_nodes, num_nodes, self.num_heads, self.head_dim
        )
        pair_bias = pair_bias.view_as(pair_weight)

        # Pair axes follow PyG edge_index: the first is source and the second is
        # target. Attention normalizes over sources and scatters into targets.
        pair_scores = key[:, :, None] + query[:, None, :]
        modulated_scores = pair_scores * pair_weight
        signed_sqrt_epsilon = 1e-8
        modulated_scores = torch.sqrt(
            F.relu(modulated_scores) + signed_sqrt_epsilon
        ) - torch.sqrt(F.relu(-modulated_scores) + signed_sqrt_epsilon)
        modulated_scores = F.relu(modulated_scores + pair_bias)

        attention_logits = torch.einsum(
            "bijhd,dh->bijh", modulated_scores, self.attention_projection
        ).clamp(min=-5.0, max=5.0)
        valid_pairs = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        attention_logits = attention_logits.masked_fill(
            ~valid_pairs.unsqueeze(-1),
            torch.finfo(attention_logits.dtype).min,
        )
        attention = torch.softmax(attention_logits, dim=1)
        attention = torch.where(
            valid_pairs.unsqueeze(-1), attention, torch.zeros_like(attention)
        )
        attention = self.attention_dropout(attention)

        node_messages = torch.einsum("bsth,bshd->bthd", attention, value)
        edge_messages = torch.einsum("bsthd,bsth->bthd", modulated_scores, attention)
        edge_messages = torch.einsum(
            "bihd,dhc->bihc", edge_messages, self.edge_value_projection
        )
        node_update = (node_messages + edge_messages).reshape(
            batch_size, num_nodes, self.dim
        )
        degree_scaled = torch.stack(
            [node_update, node_update * log_degree.unsqueeze(-1)], dim=-1
        )
        node_update = (degree_scaled * self.degree_coefficients).sum(dim=-1)
        node_update = self.node_output(self.dropout(node_update))

        pair_update = self.pair_output(
            self.dropout(modulated_scores.reshape(batch_size, num_nodes, num_nodes, -1))
        )
        node_states = node_states + node_update
        pair_states = pair_states + pair_update
        node_states = _masked_batch_norm(node_states, node_mask, self.node_norm1)
        pair_states = _masked_batch_norm(pair_states, valid_pairs, self.pair_norm)

        residual = node_states
        node_states = self.node_feed_forward1(node_states)
        node_states = F.relu(node_states)
        node_states = self.dropout(node_states)
        node_states = residual + self.node_feed_forward2(node_states)
        node_states = _masked_batch_norm(node_states, node_mask, self.node_norm2)
        return node_states, pair_states


class GRIT(nn.Module):
    """Full GRIT encoder for paper-scale graph and node experiments.

    Parameters
    ----------
    edge_dim : int
        Positional-encoding dimension.
    node_dim : int
        Hidden node and node-pair dimension. Default value is 64.
    depth : int
        Number of GRIT layers. Default value is 10.
    num_heads : int
        Number of attention heads. Default value is 8.
    num_classes : int
        Number of output classes. Default value is 2.
    head : str
        One of ``graph_class``, ``graph_reg``, or ``node_class``. Default value
        is ``graph_class``.
    pooling : str
        Graph pooling operation: ``mean``, ``sum``, or ``none`` for node-level
        heads. Default value is ``mean``.
    node_in_dim : int
        Number of input node feature columns. Default value is 1.
    edge_in_dim : int
        Number of input edge feature columns. Default value is 0.
    node_feature_type : str
        Node encoder type, ``continuous`` or ``categorical``. Default value is
        ``continuous``.
    edge_feature_type : str
        Edge encoder type, ``continuous`` or ``categorical``. Default value is
        ``continuous``.
    node_vocab_sizes : tuple[int, ...]
        Vocabulary sizes for categorical node columns. Default value is ().
    edge_vocab_sizes : tuple[int, ...]
        Vocabulary sizes for categorical edge columns. Default value is ().
    dropout : float
        Feature dropout probability. Default value is 0.0.
    attention_dropout : float
        Attention-coefficient dropout probability. Default value is 0.0.

    Raises
    ------
    ValueError
        If an encoder, head, pooling mode, or required input is invalid.
    """

    def __init__(
        self,
        edge_dim: int,
        node_dim: int = 64,
        depth: int = 10,
        num_heads: int = 8,
        num_classes: int = 2,
        head: str = "graph_class",
        pooling: str = "mean",
        node_in_dim: int = 1,
        edge_in_dim: int = 0,
        node_feature_type: str = "continuous",
        edge_feature_type: str = "continuous",
        node_vocab_sizes: tuple[int, ...] = (),
        edge_vocab_sizes: tuple[int, ...] = (),
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        if pooling not in {"mean", "sum", "none"}:
            raise ValueError(f"unknown pooling mode: {pooling}")
        if pooling == "none" and head != "node_class":
            raise ValueError("pooling='none' requires head='node_class'")
        self.head_type = head
        self.pooling = pooling
        self.edge_in_dim = edge_in_dim
        if node_feature_type == "continuous":
            self.node_encoder = nn.Linear(node_in_dim, node_dim)
        elif node_feature_type == "categorical":
            self.node_encoder = CategoricalFeatureEncoder(node_vocab_sizes, node_dim)
        else:
            raise ValueError(f"unknown node feature type: {node_feature_type}")
        if edge_in_dim == 0:
            self.edge_encoder = None
        elif edge_feature_type == "continuous":
            self.edge_encoder = nn.Linear(edge_in_dim, node_dim, bias=False)
        elif edge_feature_type == "categorical":
            self.edge_encoder = CategoricalFeatureEncoder(edge_vocab_sizes, node_dim)
        else:
            raise ValueError(f"unknown edge feature type: {edge_feature_type}")
        self.node_position_encoder = nn.Linear(edge_dim, node_dim, bias=False)
        self.pair_position_encoder = nn.Linear(edge_dim, node_dim, bias=False)
        self.layers = nn.ModuleList(
            [
                GRITAttentionLayer(
                    node_dim,
                    num_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )
        if head in {"graph_class", "graph_reg"}:
            output_dim = num_classes if head == "graph_class" else 1
            self.output_head = nn.Sequential(
                nn.Linear(node_dim, node_dim // 2),
                nn.ReLU(),
                nn.Linear(node_dim // 2, node_dim // 4),
                nn.ReLU(),
                nn.Linear(node_dim // 4, output_dim),
            )
        elif head == "node_class":
            self.output_head = nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.ReLU(),
                nn.Linear(node_dim, node_dim),
                nn.ReLU(),
                nn.Linear(node_dim, num_classes),
            )
        else:
            raise ValueError(f"unknown head: {head}")

    def forward(
        self,
        PE: torch.Tensor,
        mask: torch.Tensor,
        node_features: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict graph or node targets from positional and input features.

        Parameters
        ----------
        PE : torch.Tensor
            Dense positional encoding with shape ``(B, N, N, K)``.
        mask : torch.Tensor
            Boolean valid-node mask with shape ``(B, N)``.
        node_features : torch.Tensor | None
            Raw node features. This input is required.
        edge_features : torch.Tensor | None
            Raw edge features. Required when ``edge_in_dim`` is nonzero.
        edge_mask : torch.Tensor | None
            Boolean original-edge mask. This input is required for degree
            scaling and whenever edge features are encoded.

        Returns
        -------
        torch.Tensor
            Graph predictions or padded node predictions.

        Raises
        ------
        ValueError
            If a required feature tensor is missing or malformed.
        """
        if node_features is None:
            raise ValueError("full GRIT requires node_features")
        if edge_mask is None:
            raise ValueError("full GRIT requires edge_mask for degree scaling")
        node_states = self.node_encoder(node_features)
        diagonal_positions = torch.diagonal(PE, dim1=1, dim2=2).movedim(-1, -2)
        node_states = node_states + self.node_position_encoder(diagonal_positions)
        node_states = node_states * mask.unsqueeze(-1)
        pair_states = self.pair_position_encoder(PE)
        if self.edge_encoder is not None:
            if edge_features is None:
                raise ValueError("this GRIT model requires edge features")
            if edge_features.shape[-1] != self.edge_in_dim:
                raise ValueError(
                    f"expected {self.edge_in_dim} edge feature columns, "
                    f"got {edge_features.shape[-1]}"
                )
            pair_states = pair_states + self.edge_encoder(
                edge_features
            ) * edge_mask.unsqueeze(-1)
        valid_pairs = mask.unsqueeze(2) & mask.unsqueeze(1)
        pair_states = pair_states * valid_pairs.unsqueeze(-1)
        log_degree = torch.log1p(edge_mask.sum(dim=1).float())
        for layer in self.layers:
            node_states, pair_states = layer(node_states, pair_states, mask, log_degree)
        if self.head_type == "node_class":
            return self.output_head(node_states)
        pooled = node_states.sum(dim=1)
        if self.pooling == "mean":
            pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1)
        output = self.output_head(pooled)
        return output.squeeze(-1) if self.head_type == "graph_reg" else output


# ---------------------------------------------------------------------------
# GCN baseline (Sec. 5.2): a minimal GCN that consumes node features only.
# ---------------------------------------------------------------------------


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, padded_nodes, feature_dim = x.shape
        flat_features = x.reshape(batch_size * padded_nodes, feature_dim)
        valid_nodes = mask.reshape(-1)
        source_nodes, target_nodes = edge_index
        self_nodes = torch.nonzero(valid_nodes, as_tuple=False).flatten()
        source_nodes = torch.cat((source_nodes, self_nodes))
        target_nodes = torch.cat((target_nodes, self_nodes))
        degrees = torch.zeros(
            batch_size * padded_nodes,
            dtype=flat_features.dtype,
            device=flat_features.device,
        )
        degrees.index_add_(
            0,
            target_nodes,
            torch.ones_like(target_nodes, dtype=flat_features.dtype),
        )
        normalization = (
            degrees[source_nodes].clamp_min(1).rsqrt()
            * degrees[target_nodes].clamp_min(1).rsqrt()
        )
        aggregated = torch.zeros_like(flat_features)
        aggregated.index_add_(
            0,
            target_nodes,
            flat_features[source_nodes] * normalization.unsqueeze(-1),
        )
        out = F.relu(self.linear(aggregated.reshape_as(x)))
        return out * mask.unsqueeze(-1).float()


class GCN(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 32,
        num_classes: int = 2,
        depth: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        d = node_in_dim
        for _ in range(depth):
            self.layers.append(GCNLayer(d, hidden_dim))
            d = hidden_dim
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        A: torch.Tensor | None,
        mask: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index is None:
            if A is None:
                raise ValueError("GCN requires edge_index or a dense adjacency tensor")
            edge_index = self._edge_index_from_dense(A)
        for layer in self.layers:
            x = layer(x, mask, edge_index)
        denom = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = x.sum(dim=1) / denom
        return self.out(pooled)

    @staticmethod
    def _edge_index_from_dense(A: torch.Tensor) -> torch.Tensor:
        """Convert a dense adjacency batch into flattened directed edges."""
        batch_indices, source_nodes, target_nodes = torch.nonzero(A, as_tuple=True)
        padded_nodes = A.shape[1]
        return torch.stack(
            (
                batch_indices * padded_nodes + source_nodes,
                batch_indices * padded_nodes + target_nodes,
            )
        )

    @staticmethod
    def _norm(A: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = A.shape
        eye = torch.eye(N, device=A.device).expand(B, N, N)
        A_tilde = A + eye
        deg = A_tilde.sum(dim=-1).clamp_min(1.0)
        d_inv_sqrt = deg.pow(-0.5)
        d_mat = d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(-2)
        A_hat = A_tilde * d_mat
        return A_hat * (mask.unsqueeze(-1) & mask.unsqueeze(-2)).float()


class SparseGatedGCNLayer(nn.Module):
    """One sparse edge-gated graph-convolution layer."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.node_update = nn.Linear(hidden_dim, hidden_dim)
        self.edge_update = nn.Linear(3 * hidden_dim, hidden_dim)

    def forward(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update flattened node states and aligned directed-edge states."""
        source_nodes, target_nodes = edge_index
        gated_messages = torch.sigmoid(self.gate(edge_states)) * self.message(
            node_states[source_nodes]
        )
        aggregated_messages = torch.zeros_like(node_states)
        aggregated_messages.index_add_(0, target_nodes, gated_messages)
        target_degrees = torch.zeros(
            node_states.shape[0],
            dtype=node_states.dtype,
            device=node_states.device,
        )
        target_degrees.index_add_(
            0,
            target_nodes,
            torch.ones_like(target_nodes, dtype=node_states.dtype),
        )
        aggregated_messages = aggregated_messages / target_degrees.clamp_min(
            1
        ).unsqueeze(-1)
        updated_nodes = F.relu(self.node_update(node_states) + aggregated_messages)
        updated_edges = F.relu(
            self.edge_update(
                torch.cat(
                    (
                        node_states[source_nodes],
                        node_states[target_nodes],
                        edge_states,
                    ),
                    dim=-1,
                )
            )
        )
        return updated_nodes, updated_edges


class SparseGatedGCN(nn.Module):
    """Sparse GatedGCN baseline consuming RRWP as relative edge features."""

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 32,
        num_classes: int = 2,
        depth: int = 5,
    ):
        super().__init__()
        if edge_in_dim < 1:
            raise ValueError("SparseGatedGCN requires edge features")
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [SparseGatedGCNLayer(hidden_dim) for _ in range(depth)]
        )
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return graph logits from a padded sparse batch."""
        batch_size, padded_nodes, _ = node_features.shape
        flat_node_states = self.node_encoder(node_features).reshape(
            batch_size * padded_nodes, -1
        )
        edge_states = self.edge_encoder(edge_features)
        for layer in self.layers:
            flat_node_states, edge_states = layer(
                flat_node_states, edge_states, edge_index
            )
        node_states = flat_node_states.reshape(batch_size, padded_nodes, -1)
        node_states = node_states * node_mask.unsqueeze(-1)
        pooled_states = node_states.sum(dim=1) / node_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        return self.output(pooled_states)
