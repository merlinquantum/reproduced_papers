from collections import deque
from typing import Optional, Union

import torch


class FeedbackLayer(torch.nn.Module):
    """
    A trainable feedback layer that maintains a sliding window history of past inputs.
    This layer computes a scalar feedback value 'r_t' (reflectivity) based on a weighted moving average of past quantum
    output probabilities, transformed by a sigmoid function. It contains trainable parameters `a` and `b`.
    """

    def __init__(self, memory_size: int):
        """
        Initialize the FeedbackLayer.

        Args:
            memory_size (int): The maximum size of the sliding window history (deque).
        """
        super().__init__()
        self.memory_size = memory_size
        self.a = torch.nn.Parameter(torch.randn(1))
        self.b = torch.nn.Parameter(torch.randn(1))
        self.mem = deque(maxlen=memory_size)
        self.last_feedback: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """
        Resets the internal memory and feedback state.
        """
        self.mem.clear()
        self.last_feedback = None

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Computes the updated feedback value based on new input probabilities.

        Args:
            p (torch.Tensor): Quantum output probabilities of shape (batch_size, output_dim).

        Returns:
            torch.Tensor: The updated scalar reflectivity 'r_t'.
        """
        # We assume the relevant probability for memory update is at index 2
        p_mem = p[:, 2].mean()

        # update memory
        self.mem.append(p_mem.detach())
        mem_tensor = torch.stack(list(self.mem))

        # Calculate feedback: sigmoid(mean(a * history + b))
        r_raw = (self.a * mem_tensor + self.b).mean()
        r_t = torch.sigmoid(r_raw)

        self.last_feedback = r_t  # this is R_t now

        return r_t


class FeedbackLayerNARMA(torch.nn.Module):
    """
    A specific feedback layer implementation for the NARMA task, simulating a memristive exponential decay.
    The update rule follows: R(t) = R(t-1) + (P(t) - R(t-1)) / tau, where 'tau' corresponds to the memory size.
    """

    def __init__(
        self,
        memory_size: int,
        r_0: float = 0.5,
        update_index: int = 1,
        eps: float = 1e-6,
    ):
        """
        Initialize the NARMA Feedback Layer.

        Args:
            memory_size (int): The time constant (tau) for the decay.
            r_0 (float, optional): Initial reflectivity value. Defaults to 0.5.
            update_index (int, optional): Index of the probability vector 'p' to use for updates. Defaults to 1.
            eps (float, optional): Epsilon for numerical stability when clamping values. Defaults to 1e-6.
        """
        super().__init__()
        self.update_index = update_index
        self.r_0 = float(r_0)
        self.eps = eps
        self.register_buffer("memory_size", torch.tensor(float(memory_size)))
        self.R: Optional[torch.Tensor] = None  # Internal state storage

    def reset(self) -> None:
        """
        Resets the internal memristor state R to None.
        """
        self.R = None

    def get_R(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Retrieves the current reflectivity value R.
        Initializes it to r_0 if it hasn't been set yet.

        Args:
            device (torch.device, optional): Device to create the tensor on if initializing.
            dtype (torch.dtype, optional): Data type for the tensor.

        Returns:
            torch.Tensor: The current reflectivity R.
        """
        if self.R is None:
            return torch.tensor(self.r_0, device=device, dtype=dtype)
        return self.R

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Updates the internal memristor state based on input probability p using an exponential moving average rule.

        Args:
            p (torch.Tensor): Quantum output probabilities of shape (batch_size, output_dim).

        Returns:
            torch.Tensor: The NEW reflectivity value R_next (Scalar or Batch).
        """
        # 1. Extract the probability driving the memory (usually index 0 or 2)
        p_update = p[:, self.update_index].mean()

        if self.update_index == 2:
            p_update = 1.0 - p_update

        # 2. Get previous R
        R_prev = self.get_R(device=p_update.device, dtype=p_update.dtype)

        # 3. NARMA Update Equation: R(t) = R(t-1) + (P(t) - R(t-1)) / tau
        R_next = R_prev + (p_update - R_prev) / self.memory_size
        R_next = torch.clamp(R_next, self.eps, 1.0 - self.eps)

        # 4. Save state
        self.R = R_next.detach()

        return R_next
