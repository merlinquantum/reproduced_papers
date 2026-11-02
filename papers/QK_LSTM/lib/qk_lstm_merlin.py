"""Photonic (MerLin) fidelity kernel for QK-LSTM: a Phase 4 *adaptation*.

This is not part of the paper. It replaces the 4-qubit fidelity kernel of
``lib/quantum_kernel.py`` with a linear-optical fidelity kernel

    k(z_t, z_j) = |<s| U^dagger(z_j) U(z_t) |s>|^2

computed by ``merlin.FidelityKernel`` (SLOS simulation, exact autograd), with
``U(z) = M_out . Phi(z) . M_in``:

- ``M_in``, ``M_out``: fixed (non-trainable) MZI meshes generated from recorded
  seeds, so the feature map adds no trainable parameters (the paper's 209
  count has none);
- ``Phi(z)``: angle encoding of the compressed features, one phase per mode
  (``scale`` = 1, i.e. the identity angle map used in Phase 3);
- ``|s>``: the input Fock state, e.g. ``[1, 0, 1, 0]`` (2 photons, 4 modes).

With two or more indistinguishable photons the overlap contains multi-photon
interference (matrix permanents) and does not factorise over modes, unlike the
paper's text-circuit qubit kernel.

Computation space: ``FOCK``. In merlin 0.4.1 ``FidelityKernel`` with
``UNBUNCHED`` renormalises by the collision-free probability mass, which gives
an asymmetric, postselected quantity rather than the fidelity of Eq. 7. With
``FOCK`` the kernel is the exact fidelity. The detector model is still
threshold detection without postselection: for a collision-free input state
the click pattern of |s> arises only from the Fock outcome |s>, so its
unrenormalised probability equals this kernel. See LOG.md, Phase 4 Amendment.
"""

from __future__ import annotations

from typing import Any

import merlin as ml
import torch
from torch import nn


class PhotonicFidelityKernel(nn.Module):
    """Differentiable photonic cross-kernel ``k(z, z_reference)``.

    Parameters
    ----------
    n_modes : int
        Number of optical modes; must equal the compressed dimension (one
        encoded feature per mode).
    input_state : list of int
        Input Fock state of length ``n_modes``.
    mesh_seeds : list of int
        Seeds of the fixed input and output MZI meshes (length 2).
    encoding_scale : float
        Angle-encoding scale. Default value is 1.0.
    computation_space : str
        MerLin computation space name. Default value is ``"fock"`` (exact
        fidelity; see the module docstring).

    Raises
    ------
    ValueError
        If the input state does not match ``n_modes`` or has no photon.
    """

    def __init__(
        self,
        n_modes: int,
        input_state: list[int],
        mesh_seeds: list[int],
        *,
        encoding_scale: float = 1.0,
        computation_space: str = "fock",
    ):
        super().__init__()
        input_state = [int(v) for v in input_state]
        if len(input_state) != n_modes:
            raise ValueError(
                f"input_state {input_state} must have {n_modes} entries (one per mode)"
            )
        if sum(input_state) < 1:
            raise ValueError("input_state must contain at least one photon")
        if max(input_state) > 1:
            # The threshold-detector equivalence (see module docstring) needs it.
            raise ValueError(
                "input_state must be collision-free (at most one photon per mode)"
            )
        if len(mesh_seeds) != 2:
            raise ValueError("mesh_seeds must give the input and output mesh seeds")
        self.n_modes = n_modes
        self.input_state = input_state
        self.n_photons = sum(input_state)
        self.mesh_seeds = [int(s) for s in mesh_seeds]
        self.encoding_scale = float(encoding_scale)
        self.computation_space = ml.ComputationSpace.coerce(computation_space)

        # Building the kernel must not consume the global torch RNG, so seeded
        # model initialisation stays identical to the qubit and RBF variants.
        with torch.random.fork_rng():
            builder = ml.CircuitBuilder(n_modes=n_modes)
            builder.add_entangling_layer(trainable=False, seed=self.mesh_seeds[0])
            builder.add_angle_encoding(
                modes=list(range(n_modes)), name="x", scale=self.encoding_scale
            )
            builder.add_entangling_layer(trainable=False, seed=self.mesh_seeds[1])
            feature_map = ml.FeatureMap(
                builder=builder, input_size=n_modes, input_parameters="x"
            )
            # force_psd only applies to square Gram matrices; cross-kernels are rectangular.
            self.kernel = ml.FidelityKernel(
                feature_map,
                input_state=input_state,
                computation_space=self.computation_space,
                force_psd=False,
            )

    def forward(self, z: torch.Tensor, z_reference: torch.Tensor) -> torch.Tensor:
        """Return the ``(B, N)`` kernel between ``z`` ``(B, n_modes)`` and references ``(N, n_modes)``."""
        return self.kernel(z, z_reference)

    def describe(self) -> dict[str, Any]:
        """Hardware-aware description of the photonic configuration."""
        return {
            "computation_space": self.computation_space.name,
            "detector_model": "threshold (collision-free input state: click pattern of |s> == Fock outcome |s>; no photon-number resolution needed)",
            "n_photons": self.n_photons,
            "n_modes": self.n_modes,
            "input_state": self.input_state,
            "encoding": f"angle encoding, one phase per mode (modes 0-{self.n_modes - 1}), scale={self.encoding_scale}",
            "circuit": f"fixed MZI mesh (seed {self.mesh_seeds[0]}) -> encoding -> fixed MZI mesh (seed {self.mesh_seeds[1]})",
            "measurement_strategy": "fidelity kernel: probability of returning to the input state |s> under U^dagger(z_j) U(z_t)",
            "postselection": "none (no renormalisation over the unbunched subspace)",
            "simulator": f"MerLin {ml.__version__} FidelityKernel (SLOS), analytic, shots=0",
            "shots": 0,
            "gradients": "exact autograd through simulation",
            "trainable_circuit_parameters": int(
                sum(p.numel() for p in self.parameters() if p.requires_grad)
            ),
        }
