"""Photonic (MerLin) adaptation of the virtual quantum classifier — NOT the original.

Phase 4 adaptation of paper Sec. III B (LOG.md, "Phase 4 Adaptation Plan").
Each VQNN branch is a linear-optical circuit instead of a 2-qubit gate circuit::

    trainable 4-mode MZI mesh -> phase encoding of (x1, x2) * scale on modes 0, 1
    -> trainable 4-mode MZI mesh,   input |1,0,1,0>  (2 photons)

Modes 2-3 carry no data phase and act as the phase reference. This matters:
photon-number conservation makes the output invariant under a common phase on
all modes, so encoding (x1, x2, x1, x2) on all four modes would make every
prediction depend on x1 - x2 only (LOG.md, Phase 4 amendment 2).

measured with photon-number-resolving detection (``ComputationSpace.FOCK``,
required by MerLin for partial-distinguishability noise). Hardware noise is
partial photon distinguishability (``pcvl.NoiseModel(indistinguishability=η)``)
instead of gate depolarizing noise. The branch readout is the photon-number
parity of the first half of the modes, ``z = sum_k p_k (-1)^(n0 + n1)`` in
``[-1, 1]``, the photonic stand-in for ``<Z_0>``. A parity (second-moment)
observable is required: any readout linear in the mean mode occupations
``<n_j> = sum_i |U_ji|^2 <n_i>`` is independent of photon distinguishability
and of multi-photon interference, so it would reduce the model to independent
single photons (LOG.md, Phase 4 amendment). The VQNN combination is unchanged:
``y = (1 + c1 z1 + c2 z2) / 2`` with ``c1 = (1 + γ)/2`` and ``c2 = -(γ - 1)/2``.
"""

from __future__ import annotations

import torch
from torch import nn

from .vqem import budget_coefficients


def _computation_space(name: str):
    import merlin as ml

    spaces = {
        "fock": ml.ComputationSpace.FOCK,
        "unbunched": ml.ComputationSpace.UNBUNCHED,
    }
    if name not in spaces:
        raise ValueError(
            f"Unknown computation space {name!r}; expected one of {sorted(spaces)}"
        )
    return spaces[name]


def build_branch(
    n_modes: int,
    input_state: list[int],
    encoding_modes: list[int],
    encoding_scale: float,
    computation_space: str,
    indistinguishability: float | None,
    dtype: torch.dtype,
    device: torch.device,
):
    """Build one photonic branch as a ``merlin.QuantumLayer``.

    Parameters
    ----------
    n_modes : int
        Number of optical modes (plan: 4).
    input_state : list[int]
        Fock input state, e.g. ``[1, 0, 1, 0]`` (2 photons).
    encoding_modes : list[int]
        Modes receiving one data phase each (plan: ``[0, 1]`` for ``(x1, x2)``).
    encoding_scale : float
        Phase per unit feature in ``add_angle_encoding`` (plan: pi).
    computation_space : str
        ``"fock"`` (photon-number-resolving) or ``"unbunched"`` (threshold).
    indistinguishability : float | None
        Photon indistinguishability for ``pcvl.NoiseModel``; ``None`` means a
        noise-free layer.
    dtype : torch.dtype
        Real dtype of the layer.
    device : torch.device
        Torch device.

    Returns
    -------
    merlin.QuantumLayer
        Layer mapping ``(batch, n_modes)`` inputs to output probabilities.

    Raises
    ------
    ValueError
        If the input state does not have ``n_modes`` entries or fewer than two
        photons (the photonic-translation policy forbids single-photon models).
    """
    import merlin as ml
    import perceval as pcvl

    if len(input_state) != n_modes:
        raise ValueError(f"input_state {input_state} does not have {n_modes} modes")
    n_photons = int(sum(input_state))
    if n_photons < 2:
        raise ValueError("A meaningful MerLin adaptation requires at least two photons")
    builder = ml.CircuitBuilder(n_modes=n_modes)
    builder.add_entangling_layer()
    builder.add_angle_encoding(modes=list(encoding_modes), scale=float(encoding_scale))
    builder.add_entangling_layer()
    noise = (
        None
        if indistinguishability is None
        else pcvl.NoiseModel(indistinguishability=float(indistinguishability))
    )
    return ml.QuantumLayer(
        input_size=len(encoding_modes),
        builder=builder,
        input_state=list(input_state),
        n_photons=n_photons,
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=_computation_space(computation_space)
        ),
        noise=noise,
        dtype=dtype,
        device=device,
    )


def parity_weights(output_keys, n_modes: int, dtype: torch.dtype) -> torch.Tensor:
    """Weights ``(-1)^(photons in the first half of the modes)`` per Fock outcome."""
    half = n_modes // 2
    return torch.tensor([(-1.0) ** sum(key[:half]) for key in output_keys], dtype=dtype)


def occupation_contrast_weights(
    output_keys, n_modes: int, dtype: torch.dtype
) -> torch.Tensor:
    """Linear readout ``(n_first_half - n_second_half) / n``; kept only for the test
    showing that such readouts are blind to photon distinguishability."""
    half = n_modes // 2
    return torch.tensor(
        [
            (sum(k[:half]) - sum(k[half:])) / sum(k) if sum(k) else 0.0
            for k in output_keys
        ],
        dtype=dtype,
    )


class PhotonicVirtualClassifier(nn.Module):
    """Two-branch photonic VQNN classifier; ``budget = 1`` is the conventional one.

    Parameters
    ----------
    budget : float
        Sampling-overhead budget ``γ``.
    n_modes : int
        Modes per branch.
    input_state : list[int]
        Fock input state per branch.
    encoding_modes : list[int]
        Modes carrying the data phases, one per feature (plan: ``[0, 1]``).
    encoding_scale : float
        Phase-encoding scale.
    computation_space : str
        ``"fock"`` or ``"unbunched"``.
    indistinguishability : float | None
        Noise level; ``None`` for the noise-free control.
    dtype : torch.dtype
        Real dtype.
    device : torch.device
        Torch device.
    """

    def __init__(
        self,
        budget: float,
        n_modes: int,
        input_state: list[int],
        encoding_modes: list[int],
        encoding_scale: float,
        computation_space: str,
        indistinguishability: float | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.budget = float(budget)
        self.c1, self.c2 = budget_coefficients(self.budget)
        self.n_modes = n_modes
        n_branches = 1 if self.c2 == 0.0 else 2
        self.branches = nn.ModuleList(
            build_branch(
                n_modes,
                input_state,
                encoding_modes,
                encoding_scale,
                computation_space,
                indistinguishability,
                dtype,
                device,
            )
            for _ in range(n_branches)
        )
        weights = parity_weights(self.branches[0].output_keys, n_modes, dtype)
        self.register_buffer("readout_weights", weights.to(device))

    def branch_readouts(self, x: torch.Tensor) -> list[torch.Tensor]:
        """First-half photon-number parity ``z_i`` of each branch, shape ``(batch,)``."""
        return [branch(x) @ self.readout_weights for branch in self.branches]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions ``y`` of shape ``(batch,)``."""
        coefficients = (self.c1, self.c2)[: len(self.branches)]
        combination = sum(c * z for c, z in zip(coefficients, self.branch_readouts(x)))
        return 0.5 * (1.0 + combination)

    def n_parameters(self) -> int:
        """Trainable phases of the branches that carry nonzero weight."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
