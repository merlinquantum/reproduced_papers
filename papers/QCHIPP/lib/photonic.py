"""Hybrid measured-feature photonic QCNN adaptation, not coherent gate pooling."""

import math

import merlin as ml
import torch
from torch import nn


def is_strict_int(value) -> bool:
    """Return True for ints, rejecting bools (which subclass int)."""
    return isinstance(value, int) and not isinstance(value, bool)


def residue_angles(bits: torch.Tensor) -> torch.Tensor:
    """Convert B x 20 bits to B x 4 x 3 residue-local QRAC angles.

    Parameters
    ----------
    bits : torch.Tensor
        Four ordered five-bit residue codes, flattened per sample.

    Returns
    -------
    torch.Tensor
        Three angles per residue, after appending a zero bit locally.
    """
    if (
        not isinstance(bits, torch.Tensor)
        or bits.ndim != 2
        or bits.shape[1] != 20
        or bits.shape[0] == 0
    ):
        raise ValueError("bits must have nonempty shape (B, 20)")
    if bits.is_complex() or not torch.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must contain only finite binary values 0 and 1")
    residues = bits.reshape(-1, 4, 5).to(dtype=torch.get_default_dtype())
    pairs = torch.nn.functional.pad(residues, (0, 1)).reshape(-1, 4, 3, 2)
    b0, b1 = pairs[..., 0], pairs[..., 1]
    return math.pi / 4 + math.pi / 2 * (b1 - b0) + math.pi * b1 * b0


def _validate_optical_state(n_photons, input_state):
    if not is_strict_int(n_photons) or n_photons != 2:
        raise ValueError(
            "n_photons must be integer 2; one-photon ablation is unsupported"
        )
    if not isinstance(input_state, (list, tuple)) or len(input_state) != 6:
        raise ValueError("input_state must contain exactly six integer occupations")
    if any(not is_strict_int(value) or value < 0 for value in input_state):
        raise ValueError("input_state occupations must be nonnegative integers")
    if sum(input_state) != n_photons:
        raise ValueError("input_state photon count must match n_photons")
    if tuple(input_state) != (1, 0, 0, 1, 0, 0):
        raise ValueError("input_state must be [1, 0, 0, 1, 0, 0] for this adaptation")


def _validate_shots(module, shots):
    if not is_strict_int(shots) or shots < 0:
        raise ValueError("shots must be a nonnegative integer (0 means analytic)")
    if shots and any(child.training for child in module.modules()):
        raise ValueError("finite shots require eval() on every cell")
    if shots and torch.is_grad_enabled():
        raise ValueError("finite shots require torch.no_grad()")


class PhotonicCell(nn.Module):
    """Six-mode mesh/angle/mesh filter with classical measured pooling.

    Parameters
    ----------
    n_photons : int
        Must be 2. Default value is 2; one-photon ablations are unsupported.
    input_state : list of int or tuple of int
        Must be (1, 0, 0, 1, 0, 0). Default value is (1, 0, 0, 1, 0, 0).
    """

    def __init__(self, n_photons=2, input_state=(1, 0, 0, 1, 0, 0)):
        super().__init__()
        _validate_optical_state(n_photons, input_state)
        builder = ml.CircuitBuilder(n_modes=6)
        builder.add_entangling_layer()
        builder.add_angle_encoding(modes=list(range(6)), scale=1.0)
        builder.add_entangling_layer()
        self.quantum = ml.QuantumLayer(
            input_size=6,
            builder=builder,
            input_state=list(input_state),
            n_photons=n_photons,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.FOCK
            ),
        )
        self.projection = nn.Linear(self.quantum.output_size, 3)

    def forward(self, angles, shots=0):
        """Measure all Fock outcomes and project to three bounded features.

        Parameters
        ----------
        angles : torch.Tensor
            Shape (N, 6), in radians; the optical encoding scale is 1.
        shots : int
            Shots per local cell, or 0 for analytic execution. Default is 0.

        Returns
        -------
        torch.Tensor
            Shape (N, 3), with tanh-bounded classical measured features.

        Raises
        ------
        ValueError
            If shots is invalid or sampling lacks eval() and torch.no_grad().
        """
        _validate_shots(self, shots)
        return torch.tanh(self.projection(self.quantum(angles, shots=shots)))


class PhotonicQCNN(nn.Module):
    """Two-stage local hybrid QCNN with shared filters within each stage.

    Parameters
    ----------
    n_photons : int
        Must be 2. Default value is 2; other photon counts are unsupported.
    input_state : sequence of int
        Fixed six-mode occupations. Default value is (1, 0, 0, 1, 0, 0).

    Notes
    -----
    This is a PARTIAL_MERLIN_TRANSLATION: pooling measures, projects and
    re-encodes classical features, rather than preserving a coherent state.
    Full Fock probabilities assume ideal number-resolving detectors and no
    postselection. There are three cell executions per peptide.
    """

    def __init__(self, n_photons=2, input_state=(1, 0, 0, 1, 0, 0)):
        super().__init__()
        _validate_optical_state(n_photons, input_state)
        self.n_photons = n_photons
        self.input_state = tuple(input_state)
        self.stage1 = PhotonicCell(n_photons, self.input_state)
        self.stage2 = PhotonicCell(n_photons, self.input_state)
        self.readout = nn.Linear(3, 1)

    def forward(self, bits, shots=0):
        """Return B binary logits from B x 20 residue bits.

        Parameters
        ----------
        bits : torch.Tensor
            Four ordered five-bit residue codes per peptide.
        shots : int
            Shots per cell, with 0 for analytic execution. Default value is 0.
            Positive values require eval() and torch.no_grad(); each peptide
            uses three cell measurements, hence 3 * shots total shots.

        Returns
        -------
        torch.Tensor
            One logit per peptide, shape (B,).

        Raises
        ------
        ValueError
            If bits or shots is invalid, or finite-shot execution would be
            silently analytic because training or autograd is enabled.
        """
        _validate_shots(self, shots)
        angles = residue_angles(bits).to(self.readout.weight)
        local = self.stage1(angles.reshape(-1, 6), shots=shots)
        pooled_angles = local.reshape(bits.shape[0], 6) * math.pi
        return self.readout(self.stage2(pooled_angles, shots=shots)).squeeze(-1)


class ReferencePhotonicCell(PhotonicCell):
    """Encode six phases against a seventh, unencoded optical reference mode."""

    def __init__(self):
        nn.Module.__init__(self)
        builder = ml.CircuitBuilder(n_modes=7)
        builder.add_entangling_layer()
        builder.add_angle_encoding(modes=list(range(6)), scale=1.0)
        builder.add_entangling_layer()
        # Deliberate seven-mode state, not a silent repair of the legacy state.
        self.quantum = ml.QuantumLayer(
            input_size=6,
            builder=builder,
            input_state=[1, 0, 0, 1, 0, 0, 0],
            n_photons=2,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.FOCK
            ),
        )
        self.projection = nn.Linear(self.quantum.output_size, 3)


class ReferencePhotonicQCNN(PhotonicQCNN):
    """Information-corrected partial adaptation with two-photon reference cells.

    Notes
    -----
    Encoding every mode loses a common input phase: distinct A/L categorical
    codes can alias exactly in the legacy six-mode architecture. Mode6 is left
    unencoded here and lies in the first mesh's photon light cone, so the six
    input phases are relative to a physical reference. Measured pooling still
    changes coherent representational scale; the translation remains partial.
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.n_photons = 2
        self.input_state = (1, 0, 0, 1, 0, 0, 0)
        self.stage1 = ReferencePhotonicCell()
        self.stage2 = ReferencePhotonicCell()
        self.readout = nn.Linear(3, 1)
