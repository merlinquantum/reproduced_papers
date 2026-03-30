import torch
import torch.nn as nn


class ClassicalBenchmark(nn.Module):
    r"""
    Implements four specific classical benchmark models for time-series comparison.
    These models range from simple linear regressions to polynomial models with short-term memory (looking back one
    time step).

    The supported model types are:

    1. **'L' (Linear)**:
       Output depends linearly on the current input `u_t`.
       Formula: $Out = a \cdot u_t + b$

    2. **'Q' (Cubic Polynomial)**:
       Output depends non-linearly on the current input `u_t` up to degree 3.
       Formula: $Out = a \cdot u_t^3 + b \cdot u_t^2 + c \cdot u_t + d$

    3. **'L+M' (Linear + Memory)**:
       Output depends linearly on current input `u_t` and previous input `u_{t-1}`.
       Formula: $Out = a \cdot u_t + b \cdot u_{t-1} + c$

    4. **'Q+M' (Cubic Polynomial + Memory)**:
       A partial 3rd-degree polynomial involving current `u_t` and previous `u_{t-1}`.
       It explicitly uses the following 8 terms (plus bias):
       $u_t^3, u_t^2 u_{t-1}, u_t u_{t-1}^2, u_t^2, u_t u_{t-1}, u_{t-1}^2, u_t, u_{t-1}$.
    """

    def __init__(self, model_type: str = "L", input_dim: int = 1):
        """
        Initialize the classical benchmark model.

        Args:
            model_type (str, optional): The architecture identifier ('L', 'Q', 'L+M', 'Q+M').
                Defaults to "L".
            input_dim (int, optional): The size of the input feature dimension.
                Defaults to 1.

        Raises:
            ValueError: If an unknown `model_type` is provided.
        """
        super().__init__()
        self.model_type = model_type
        self.input_dim = input_dim

        # Define the input dimension for the final linear layer
        if model_type == "L":
            # Features: u_t (1 term)
            self.net = nn.Linear(input_dim, 1)

        elif model_type == "Q":
            # Features: u_t^3, u_t^2, u_t (3 terms)
            self.net = nn.Linear(input_dim * 3, 1)

        elif model_type == "L+M":
            # Features: u_t, u_{t-1} (2 terms)
            self.net = nn.Linear(input_dim * 2, 1)

        elif model_type == "Q+M":
            # Features:
            # 1. u_t^3
            # 2. u_t^2 * u_{t-1}
            # 3. u_t * u_{t-1}^2
            # 4. u_t^2
            # 5. u_t * u_{t-1}
            # 6. u_{t-1}^2
            # 7. u_t
            # 8. u_{t-1}
            # Total = 8 weights + 1 bias
            self.net = nn.Linear(input_dim * 8, 1)

        else:
            raise ValueError(f"Unknown classical model type: {model_type}")

    def forward(self, u_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for processing a sequence of inputs.
        Constructs the feature vector based on the selected `model_type` and applies the learnable linear layer. For
        memory-based models ('+M'), it automatically constructs the time-shifted previous step `u_{t-1}`, padding with
        zero for the first time step.

        Args:
            u_seq (torch.Tensor): Input sequence tensor of shape (Batch, Seq_Len, Input_Dim).

        Returns:
            torch.Tensor: Output prediction sequence of shape (Batch, Seq_Len, 1).
        """
        batch_size, seq_len, _ = u_seq.shape
        device = u_seq.device

        # 1. Prepare u_t
        u_t = u_seq  # (B, T, 1)

        # 2. Prepare u_{t-1} (Shifted, padded with 0 at t=0)
        u_tm1 = torch.cat(
            [torch.zeros(batch_size, 1, 1, device=device), u_seq[:, :-1, :]], dim=1
        )

        # 3. Construct Feature Vector based on Model Type
        if self.model_type == "L":
            features = u_t

        elif self.model_type == "Q":
            features = torch.cat([u_t**3, u_t**2, u_t], dim=2)

        elif self.model_type == "L+M":
            features = torch.cat([u_t, u_tm1], dim=2)

        elif self.model_type == "Q+M":
            features = torch.cat(
                [
                    u_t**3,
                    (u_t**2) * u_tm1,
                    u_t * (u_tm1**2),
                    u_t**2,
                    u_t * u_tm1,
                    u_tm1**2,
                    u_t,
                    u_tm1,
                ],
                dim=2,
            )

        # 4. Apply Linear Layer (Weights + Bias)
        return self.net(features)
