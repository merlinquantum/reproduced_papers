"""Adam with per-agent update masks (Keras / TensorFlow 2.3 update rule).

Agents trained in one process update at different environment steps
(``update_model_every`` counts steps inside each agent's own episode). A
standard optimiser would still move the moments and parameters of agents that
receive a zero gradient, so this optimiser applies the update only to the
agents selected by a boolean mask and keeps a step counter per agent.

The update reproduces ``tf.keras.optimizers.Adam`` as used by the reference
code (``epsilon`` added to ``sqrt(v)`` before bias correction of the step
size; optional AMSGrad):

.. code-block:: text

    t     <- t + 1
    m     <- beta1 m + (1 - beta1) g
    v     <- beta2 v + (1 - beta2) g^2
    vhat  <- max(vhat, v)                      (AMSGrad only)
    lr_t  =  lr sqrt(1 - beta2^t) / (1 - beta1^t)
    p     <- p - lr_t m / (sqrt(vhat or v) + epsilon)
"""

from __future__ import annotations

import torch


class MaskedAdam:
    """Per-agent masked Adam over tensors with a leading agent dimension.

    Parameters
    ----------
    groups : list of tuple
        ``(tensors, learning_rate, amsgrad)`` per parameter group. Every tensor
        has shape ``(A, ...)``.
    beta1, beta2, epsilon : float
        Adam constants (Keras defaults 0.9, 0.999, 1e-7).
    """

    def __init__(self, groups, beta1: float, beta2: float, epsilon: float):
        if not groups:
            raise ValueError("MaskedAdam needs at least one parameter group")
        self.groups = []
        for tensors, learning_rate, amsgrad in groups:
            if learning_rate is None:
                raise ValueError(
                    "Every trainable parameter group needs a learning rate"
                )
            state = []
            for tensor in tensors:
                state.append(
                    {
                        "m": torch.zeros_like(tensor),
                        "v": torch.zeros_like(tensor),
                        "vhat": torch.zeros_like(tensor) if amsgrad else None,
                    }
                )
            self.groups.append(
                {
                    "tensors": list(tensors),
                    "lr": float(learning_rate),
                    "amsgrad": bool(amsgrad),
                    "state": state,
                }
            )
        n_agents = self.groups[0]["tensors"][0].shape[0]
        device = self.groups[0]["tensors"][0].device
        self.steps = torch.zeros(n_agents, dtype=torch.float64, device=device)
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon

    def zero_grad(self) -> None:
        """Clear gradients of every managed tensor."""
        for group in self.groups:
            for tensor in group["tensors"]:
                tensor.grad = None

    @torch.no_grad()
    def step(self, agent_mask: torch.Tensor) -> None:
        """Apply one update to the agents where ``agent_mask`` is True.

        Parameters
        ----------
        agent_mask : torch.Tensor
            Boolean tensor of shape ``(A,)``.
        """
        if not bool(agent_mask.any()):
            return
        self.steps = self.steps + agent_mask.to(self.steps.dtype)
        bias_step = torch.where(agent_mask, self.steps, torch.ones_like(self.steps))
        step_scale = torch.sqrt(1 - self.beta2**bias_step) / (1 - self.beta1**bias_step)
        for group in self.groups:
            for tensor, state in zip(group["tensors"], group["state"]):
                if tensor.grad is None:
                    continue
                shape = (-1,) + (1,) * (tensor.dim() - 1)
                mask = agent_mask.view(shape)
                gradient = tensor.grad
                new_m = self.beta1 * state["m"] + (1 - self.beta1) * gradient
                new_v = self.beta2 * state["v"] + (1 - self.beta2) * gradient * gradient
                state["m"] = torch.where(mask, new_m, state["m"])
                state["v"] = torch.where(mask, new_v, state["v"])
                denominator_source = state["v"]
                if group["amsgrad"]:
                    state["vhat"] = torch.where(
                        mask, torch.maximum(state["vhat"], state["v"]), state["vhat"]
                    )
                    denominator_source = state["vhat"]
                lr_t = (group["lr"] * step_scale).to(tensor.dtype).view(shape)
                update = (
                    lr_t * state["m"] / (torch.sqrt(denominator_source) + self.epsilon)
                )
                tensor.sub_(torch.where(mask, update, torch.zeros_like(update)))
