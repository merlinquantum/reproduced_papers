"""Functional fully connected networks used as classical baselines (paper Sec. 2, footnote 5).

The parameter vector of each sample is a flat row of ``theta`` laid out like
``torch.nn.Module.parameters()`` of a stack of ``nn.Linear`` layers: for every
layer, the weight matrix ``(out, in)`` in row-major order, then the bias ``(out,)``
when biases are enabled. That keeps the parameter ordering identical to the
authors' ``ClassicalNeuralNetwork`` so per-entry Fisher matrices can be compared.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

ACTIVATIONS = {
    "leaky_relu": F.leaky_relu,
    "relu": torch.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}
# "softmax_tanh" is the head of the authors' Fisher code (softmax(tanh(z)));
# "softmax" is the head of their Iris training code (logits into CrossEntropyLoss).
HEADS = ("softmax_tanh", "softmax")


def parameter_count(sizes: list[int], bias: bool) -> int:
    """Return ``d`` for layer sizes ``[s_in, h_1, ..., s_out]``."""
    return sum(
        sizes[i] * sizes[i + 1] + (sizes[i + 1] if bias else 0)
        for i in range(len(sizes) - 1)
    )


def enumerate_topologies(
    input_size: int,
    output_size: int,
    target_d: int,
    max_hidden_layers: int,
    bias: bool = False,
) -> list[list[int]]:
    """List every layer-size list with exactly ``target_d`` parameters.

    Parameters
    ----------
    input_size, output_size : int
        ``s_in`` and ``s_out``.
    target_d : int
        Required number of trainable parameters.
    max_hidden_layers : int
        Upper bound on the number of hidden layers (0 allows the linear model).
    bias : bool
        Count biases. Default value is False.

    Returns
    -------
    list of list of int
        Topologies ``[s_in, h_1, ..., h_k, s_out]`` sorted by depth, then sizes.
    """
    found: list[list[int]] = []

    def extend(prefix: list[int], used: int, hidden_left: int) -> None:
        previous = prefix[-1]
        closing = previous * output_size + (output_size if bias else 0)
        if used + closing == target_d:
            found.append(prefix + [output_size])
        if hidden_left == 0:
            return
        width = 1
        while True:
            cost = previous * width + (width if bias else 0)
            # Cheapest way to close a layer of this width: directly to s_out, or through
            # one more width-1 hidden layer (biases only add parameters).
            minimum_rest = min(width * output_size, width + output_size)
            if used + cost + minimum_rest > target_d:
                break
            extend(prefix + [width], used + cost, hidden_left - 1)
            width += 1

    extend([input_size], 0, max_hidden_layers)
    return sorted(found, key=lambda sizes: (len(sizes), sizes))


class FeedForwardNetwork:
    """Fully connected network evaluated with a per-sample parameter matrix.

    Parameters
    ----------
    sizes : list of int
        ``[s_in, h_1, ..., h_k, s_out]``.
    activation : str
        Hidden activation, one of ``ACTIVATIONS``.
    head : str
        Output head, one of ``HEADS``.
    bias : bool
        Use biases. Default value is False (the paper's networks).
    """

    def __init__(
        self, sizes: list[int], activation: str, head: str, bias: bool = False
    ):
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation {activation!r}; expected one of {sorted(ACTIVATIONS)}"
            )
        if head not in HEADS:
            raise ValueError(f"Unsupported head {head!r}; expected one of {HEADS}")
        self.sizes = list(sizes)
        self.activation = ACTIVATIONS[activation]
        self.head = head
        self.bias = bias
        self.input_size = sizes[0]
        self.output_size = sizes[-1]
        self.d = parameter_count(self.sizes, bias)

    def logits(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Return the pre-head outputs ``(B, s_out)``."""
        if theta.shape[1] != self.d:
            raise ValueError(f"Expected theta (B, {self.d}); got {tuple(theta.shape)}")
        hidden = x
        offset = 0
        n_layers = len(self.sizes) - 1
        for layer in range(n_layers):
            fan_in, fan_out = self.sizes[layer], self.sizes[layer + 1]
            weight = theta[:, offset : offset + fan_out * fan_in].reshape(
                -1, fan_out, fan_in
            )
            offset += fan_out * fan_in
            hidden = torch.einsum("boi,bi->bo", weight, hidden)
            if self.bias:
                hidden = hidden + theta[:, offset : offset + fan_out]
                offset += fan_out
            if layer < n_layers - 1:
                hidden = self.activation(hidden)
        return hidden

    def probabilities(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Return class probabilities ``(B, s_out)`` through the configured head."""
        outputs = self.logits(x, theta)
        if self.head == "softmax_tanh":
            outputs = torch.tanh(outputs)
        return torch.softmax(outputs, dim=-1)
