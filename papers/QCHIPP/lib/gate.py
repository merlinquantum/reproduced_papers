# Copyright 2021, IBM Corporation.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Torch translation of IBM Q-CHIPP v1.0.0's unshared QRAC pyramid.

Source: commit 9f72bd9fc87522bc45c806f9849d07210a64e48d,
``libraries/qcnn_lib.py`` and ``qrac/qrac_parametervector.py``.
Only circuit structure is compiled; Qiskit is not a runtime dependency.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def _pyramid(n):
    operations, keys = [], set()

    def unit(kind, a, b, prefix, index):
        parameters = tuple((prefix, index + j) for j in range(3))
        keys.update(parameters)
        operations.append((kind, a, b, parameters))

    def convolution(width, offset, prefix, passes):
        index = 0
        for start in passes:
            for a in range(start, width - 1, 2):
                unit("conv", offset + a, offset + a + 1, prefix, index)
                index += 3

    center, width, layer = n // 2, n, 0
    while width > 2:
        offset = center - width // 2
        if width % 4 == 0:
            convolution(width, offset, f"c{layer}", (0, 1))
        else:
            convolution(width, offset, f"cs{layer}", (1,))
            convolution(width, offset, f"cf{layer}", (0,))
        midpoint = (width // 2 // 2) * 2
        endpoint = midpoint * 2
        sources = list(range(0, midpoint - 1, 2)) + list(
            range(midpoint + 1, endpoint, 2)
        )
        sinks = list(range(1, midpoint, 2)) + list(range(midpoint, endpoint - 1, 2))
        offset = center - endpoint // 2
        for i, (a, b) in enumerate(zip(sources, sinks)):
            unit("pool", offset + a, offset + b, f"p{layer}", 3 * i)
        swap_size = math.ceil((math.ceil(width / 2) - 2) / 2) * 2
        for offset in (center - swap_size - 1, center + 1):
            wires = list(range(offset, offset + swap_size))
            while wires:
                for a, b in zip(wires[::2], wires[1::2]):
                    operations.append(("swap", a, b, ()))
                wires = wires[1:-1]
        width = math.ceil(math.ceil(width / 2) / 2) * 2
        layer += 1
    convolution(2, center - 1, f"c{layer}", (0, 1))
    unit("pool", center - 1, center, f"p{layer}", 0)
    ordered = tuple(sorted(keys))
    binding = {key: i for i, key in enumerate(ordered)}
    return tuple(
        (kind, a, b, tuple(binding[k] for k in params))
        for kind, a, b, params in operations
    ), tuple(f"{p}[{i}]" for p, i in ordered)


def source_objective(scores, targets01):
    """Return the exact signed-label source scalar cross-entropy value.

    Parameters
    ----------
    scores : torch.Tensor
        Signed source scores, shape (B,).
    targets01 : torch.Tensor
        Binary labels, shape (B,), explicitly mapped to {-1,+1} here.

    Returns
    -------
    torch.Tensor
        Mean ``-(2*target-1)*log2(clamp_min(score, 1e-10))``.
        Autograd differentiates this value, not QML's softmax gradient surrogate.
    """
    if scores.ndim != 1 or targets01.shape != scores.shape:
        raise ValueError("scores and targets01 must have identical shape (B,)")
    targets = targets01.to(device=scores.device, dtype=scores.dtype)
    return (-(2 * targets - 1) * torch.log2(scores.clamp_min(1e-10))).mean()


def corrected_objective(scores, targets01):
    """Return explicitly repaired Bernoulli BCE in bits, not the source loss.

    Parameters
    ----------
    scores : torch.Tensor
        Signed source scores of shape (B,).
    targets01 : torch.Tensor
        Binary labels of shape (B,).

    Returns
    -------
    torch.Tensor
        Mean base-two BCE with physical p=0.5+score. Both outcome probabilities
        are independently lower-clipped at 1e-10, including in float32.
    """
    if scores.ndim != 1 or targets01.shape != scores.shape:
        raise ValueError("scores and targets01 must have identical shape (B,)")
    targets = targets01.to(device=scores.device, dtype=scores.dtype)
    probability = 0.5 + scores
    return -(
        targets * torch.log2(probability.clamp_min(1e-10))
        + (1 - targets) * torch.log2((1 - probability).clamp_min(1e-10))
    ).mean()


class GateQCNN(nn.Module):
    """Source pyramid with globally paired (2,1)-QRAC input.

    Parameters
    ----------
    n_bits : int, optional
        Input width, divisible by four and at least eight. Default is 20.
    seed : int, optional
        NumPy default_rng seed for source QML uniform [0,1) weights. Default 49.
    dtype : torch.dtype, optional
        Real parameter precision, float32 or float64. Default torch.float32.

    Notes
    -----
    ``weights`` follows Qiskit's sorted ansatz.parameters order, not gate order.
    Constructor supports structural inspection at 46 qubits without a statevector.
    Uniform initialization matches QML 0.9's distribution, not standard_normal.
    The seed here is explicit: the release seeds qiskit_algorithms globals, while
    QML 0.9 uses its own globals; seed propagation is not guaranteed in Qiskit 2.
    """

    def __init__(self, n_bits=20, seed=49, dtype=torch.float32):
        super().__init__()
        if n_bits < 8 or n_bits % 4:
            raise ValueError("n_bits must be divisible by four and at least eight")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self.n_bits, self.n_qubits = n_bits, n_bits // 2
        self.operations, self.parameter_names = _pyramid(self.n_qubits)
        self.weights = nn.Parameter(
            torch.tensor(
                np.random.default_rng(seed).random(len(self.parameter_names)),
                dtype=dtype,
            )
        )
        self._execution = []
        for kind, a, b, params in self.operations:
            axes = list(range(self.n_qubits + 1))
            if kind == "swap":
                axes[self.n_qubits - a], axes[self.n_qubits - b] = (
                    axes[self.n_qubits - b],
                    axes[self.n_qubits - a],
                )
                self._execution.append((kind, tuple(axes), (), params))
            else:
                axes.remove(self.n_qubits - a)
                axes.remove(self.n_qubits - b)
                axes += [self.n_qubits - b, self.n_qubits - a]
                inverse = tuple(axes.index(i) for i in range(self.n_qubits + 1))
                self._execution.append((kind, tuple(axes), inverse, params))

    def probabilities(self, bits):
        """Return physical positive-class probability, ``0.5 + forward(bits)``.

        Parameters
        ----------
        bits : torch.Tensor
            Inputs of shape (B, n_bits).

        Returns
        -------
        torch.Tensor
            Positive-class probabilities of shape (B,).
        """
        return 0.5 + self(bits)

    def forward(self, bits):
        """Return signed source scores ``0.5*<Z[n_qubits//2]>``.

        Parameters
        ----------
        bits : torch.Tensor
            Batch of shape (B, n_bits), binary or continuous QRAC inputs.

        Returns
        -------
        torch.Tensor
            Signed scores of shape (B,), in the parameter precision.
        """
        if bits.ndim != 2 or bits.shape[1] != self.n_bits:
            raise ValueError(f"bits must have shape (B, {self.n_bits})")
        if self.n_qubits > 20:
            raise ValueError(
                "Statevector evaluation above 20 qubits is prohibited; structural inspection only"
            )
        bits = bits.to(device=self.weights.device, dtype=self.weights.dtype)
        batch, n = bits.shape[0], self.n_qubits
        angles = (
            math.pi / 4
            + math.pi / 2 * (bits[:, 1::2] - bits[:, ::2])
            + math.pi * bits[:, 1::2] * bits[:, ::2]
        )
        local = torch.stack((torch.cos(angles / 2), torch.sin(angles / 2)), dim=-1)
        state = torch.ones((batch, 1), device=bits.device, dtype=bits.dtype)
        for q in reversed(range(n)):
            state = (state.unsqueeze(-1) * local[:, q].unsqueeze(1)).flatten(1)
        complex_dtype = (
            torch.complex64 if bits.dtype == torch.float32 else torch.complex128
        )
        shape = (batch,) + (2,) * n
        state = state.to(complex_dtype).reshape(shape)
        eye = torch.eye(2, dtype=complex_dtype, device=bits.device)
        cx10 = torch.eye(4, dtype=complex_dtype, device=bits.device)[[0, 1, 3, 2]]
        cx01 = torch.eye(4, dtype=complex_dtype, device=bits.device)[[0, 3, 2, 1]]
        zstart = torch.kron(
            torch.diag(torch.exp(-0.5j * bits.new_tensor([-math.pi / 2, math.pi / 2]))),
            eye,
        )
        zend = torch.kron(
            eye,
            torch.diag(torch.exp(-0.5j * bits.new_tensor([math.pi / 2, -math.pi / 2]))),
        )
        cos, sin = torch.cos(self.weights / 2), torch.sin(self.weights / 2)
        ry = (
            torch.stack((cos, -sin, sin, cos), dim=-1)
            .reshape(-1, 2, 2)
            .to(complex_dtype)
        )
        phases = torch.exp(-0.5j * self.weights)
        rz = torch.diag_embed(torch.stack((phases, phases.conj()), dim=-1))
        for kind, axes, inverse, params in self._execution:
            if kind == "swap":
                state = state.permute(axes)
                continue
            p0, p1, p2 = params
            unitary = (
                torch.kron(ry[p2].contiguous(), eye)
                @ cx01
                @ torch.kron(ry[p1].contiguous(), rz[p0].contiguous())
                @ cx10
                @ zstart
            )
            if kind == "conv":
                unitary = zend @ cx10 @ unitary
            transformed = state.permute(axes).reshape(batch, -1, 4) @ unitary.T
            state = transformed.reshape(shape).permute(inverse)
        # Qiskit little-endian Z label measures physical n/2, not n/2-1.
        probability = state.abs().square()
        measured = probability.movedim(n - n // 2, -1).reshape(batch, -1, 2).sum(dim=1)
        return 0.5 * (measured[:, 0] - measured[:, 1])
