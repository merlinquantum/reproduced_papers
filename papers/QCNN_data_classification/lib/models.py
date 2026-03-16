"""Model definitions for the QCNN data classification reproduction."""

from __future__ import annotations

import torch
import torch.nn as nn
from utils.circuit import build_quantum_kernel_layer, build_single_gi_layer


class SingleGI(nn.Module):
    """Photonic layer applying a single Gaussian interferometer."""

    def __init__(
        self,
        n_modes: int,
        n_features: int,
        n_photons: int,
        reservoir_mode: bool,
        state_pattern: str,
        required_inputs: int,
        input_dim: int,
    ) -> None:
        super().__init__()
        # Enforce that the upstream preprocessing matches the optical circuit width.
        if input_dim != required_inputs:
            raise ValueError(
                f"Quantum layer expects {required_inputs} features but received {input_dim}. "
                "Adjust preprocessing or circuit parameters so they match exactly."
            )
        self.K = required_inputs
        # `build_single_gi_layer` returns a Merlin QuantumLayer configured for the
        # requested interferometer topology (modes/features/photons/etc.).
        self.q_layer = build_single_gi_layer(
            n_modes=n_modes,
            n_features=n_features,
            n_photons=n_photons,
            reservoir_mode=reservoir_mode,
            state_pattern=state_pattern,
        )

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        # Accept both single samples and batched inputs for convenience.
        if angles.dim() == 1:
            angles = angles.unsqueeze(0)
        if angles.shape[-1] != self.K:
            raise ValueError(
                f"Expected input with last dimension {self.K}, got {angles.shape[-1]}."
            )
        return self.q_layer(angles)


class QuantumPatchKernel(nn.Module):
    """Wrapper that applies a QuantumLayer to sliding PCA patches."""

    def __init__(self, q_layer: nn.Module, patch_dim: int) -> None:
        super().__init__()
        self.q_layer = q_layer
        self.patch_dim = patch_dim
        self.output_keys = self._extract_output_keys(q_layer)

    @staticmethod
    def _extract_output_keys(q_layer: nn.Module) -> list[str]:
        keys = getattr(q_layer, "output_keys", None)
        if keys is not None:
            return list(keys)
        for child in q_layer.children():
            child_keys = getattr(child, "output_keys", None)
            if child_keys is not None:
                return list(child_keys)
        return []

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        # Kernels are evaluated on flattened PCA patches, so enforce the width.
        if patch.dim() == 1:
            patch = patch.unsqueeze(0)
        if patch.shape[-1] != self.patch_dim:
            raise ValueError(
                f"Expected patch with last dimension {self.patch_dim}, got {patch.shape[-1]}."
            )
        return self.q_layer(patch)


class QConvModel(nn.Module):
    """Pseudo-convolutional model mixing classical or quantum kernels."""

    def __init__(
        self,
        input_dim: int,
        n_kernels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        kernel_modules: list[nn.Module] | None = None,
        amplitudes_encoding: bool = False,
    ) -> None:
        super().__init__()
        if n_kernels <= 0:
            raise ValueError("n_kernels must be a positive integer.")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer.")
        if stride <= 0:
            raise ValueError("stride must be a positive integer.")

        self.input_dim = input_dim
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_quantum = kernel_modules is not None
        self.amplitudes_encoding = amplitudes_encoding

        if self.use_quantum:
            # Quantum mode: we receive a pre-built list of QuantumPatchKernel modules.
            if kernel_modules is None or len(kernel_modules) != n_kernels:
                raise ValueError("kernel_modules must contain one module per kernel.")
            self.kernel_modules = nn.ModuleList(kernel_modules)
            self.space_dim = len(self.kernel_modules[0].output_keys)
            if self.amplitudes_encoding and self.space_dim == 0:
                raise ValueError(
                    "Amplitude encoding requires quantum kernels that expose non-empty output_keys."
                )
            if self.amplitudes_encoding:
                for kernel in self.kernel_modules:
                    kernel.patch_dim = self.space_dim
            self.kernel_output_dim = 2  # Each quantum kernel outputs 2 features.
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            self.classical_conv = None
        else:
            # Classical fallback: delegate to Conv1d for efficiency and correctness.
            self.classical_conv = nn.Conv1d(
                in_channels=1,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
            )
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            self.kernel_modules = None
            self.kernel_output_dim = 1

        self.output_features = self.output_dim(input_dim)
        self.head = nn.Linear(self.output_features, 2)

    def _compute_num_windows(self, input_len: int) -> int:
        # Mirror torch.nn.Conv1d logic: number of sliding positions per feature vector.
        if input_len < self.kernel_size:
            raise ValueError(
                f"Input length {input_len} is smaller than kernel size {self.kernel_size}."
            )
        return 1 + (input_len - self.kernel_size) // self.stride

    def output_dim(self, input_len: int) -> int:
        num_windows = self._compute_num_windows(input_len)
        return num_windows * self.n_kernels * self.kernel_output_dim

    def _apply_classical(self, x: torch.Tensor) -> torch.Tensor:
        # Apply a learnable 1D convolution to match the quantum receptive field.
        conv_in = x.unsqueeze(1)  # (batch, channels=1, features)
        out = self.classical_conv(conv_in)
        return out.view(out.size(0), -1)

    def _apply_quantum(
        self, patches: torch.Tensor, num_windows: int, batch_size: int
    ) -> torch.Tensor:
        # Evaluate every patch with each quantum kernel module and stack responses.
        patches_flat = patches.contiguous().view(-1, patches.size(-1))
        outputs = []
        for kernel in self.kernel_modules:
            y = kernel(patches_flat)
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            y = y.view(batch_size, num_windows, -1)
            outputs.append(y)
        out = torch.stack(outputs, dim=1)
        return out.view(out.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError(
                "QConvModel expects inputs of shape (batch, features) or a single feature vector."
            )

        num_windows = self._compute_num_windows(x.size(-1))

        if self.use_quantum:
            patches = x.unfold(dimension=-1, size=self.kernel_size, step=self.stride)
            if self.amplitudes_encoding:
                patch_width = patches.shape[-1]
                if patch_width > self.space_dim:
                    raise ValueError(
                        "Amplitude encoding patches cannot exceed the computation space dimension."
                    )
                pad = self.space_dim - patch_width

                if pad:
                    pad_shape = patches.shape[:-1] + (pad,)
                    zeros = patches.new_zeros(*pad_shape)
                    patches = torch.cat((patches, zeros), dim=-1)
            features = self._apply_quantum(patches, num_windows, x.size(0))
        else:
            features = self._apply_classical(x)
        return self.head(features)


def build_quantum_kernels(
    n_kernels: int,
    kernel_size: int,
    kernel_modes: int,
    n_photons: int,
    state_pattern: str,
    reservoir_mode: bool,
    amplitudes_encoding: bool,
    show_circuit: bool = False,
) -> list[QuantumPatchKernel]:
    """Build a list of QuantumPatchKernel modules with shared configuration."""
    modules: list[QuantumPatchKernel] = []
    for _ in range(n_kernels):
        layer = build_quantum_kernel_layer(
            kernel_modes=kernel_modes,
            kernel_features=kernel_size,
            n_photons=n_photons,
            state_pattern=state_pattern,
            reservoir_mode=reservoir_mode,
            show_circuit=show_circuit,
            amplitudes_encoding=amplitudes_encoding,
        )
        modules.append(QuantumPatchKernel(layer, patch_dim=kernel_size))
    return modules
