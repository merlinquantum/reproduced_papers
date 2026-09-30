"""Photonic (MerLin) counterpart of the paper's ZFeatureMap + RealAmplitudes QNN.

The source paper is written for the gate model (Qiskit). This module implements
the *photonic adaptation* used throughout this reproduction. The translation
rules are recorded in ``LOG.md`` under "Hardware Equivalence Mapping"; the
short version is:

===========================  ==================================================
Gate-model construct         Photonic / MerLin counterpart
===========================  ==================================================
n features -> n qubits       n features -> n modes
``H`` column of ZFeatureMap  fixed 50:50 beam-splitter column (superposition)
``P(2*x_i)``                 phase shifter of angle ``encoding_scale * x_i``
``RY(theta_i)`` of RA        trainable phase shifter on mode ``i``
CNOT entangler of RA         fixed (non-trainable) beam-splitter mesh
computational-basis readout  photon-counting probabilities over Fock outcomes
class assignment             lexicographic grouping of the outcome vector
===========================  ==================================================

Two deviations are forced by photonic measurement semantics and are documented
in the README and ``LOG.md``:

1. ``RealAmplitudes`` ends on a rotation layer. A trailing column of phase
   shifters is invisible to photon-counting detectors, so every trainable phase
   column here is followed by a mixing column.
2. The entangling columns are non-trainable so that the trainable parameter
   count stays at ``n_modes * n_layers``, matching
   ``RealAmplitudes(n_qubits, reps=n_layers - 1)``.

Circuits are built through :mod:`lib.circuit_ir` rather than directly with
``CircuitBuilder``, so that the original model and every mutant share one code
path. ``tests/test_smoke.py`` asserts that the IR path is numerically identical
to the equivalent ``CircuitBuilder`` circuit.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import merlin as ml
import perceval as pcvl
import torch
from lib.circuit_ir import (
    BLOCK_ANSATZ_MIX,
    BLOCK_ANSATZ_PHASE,
    CircuitIR,
    Element,
    build_original_ir,
    set_weights,
)
from merlin.models.qcnn import QCNNClassifier
from torch import nn

logger = logging.getLogger(__name__)

DEFAULT_ENCODING_SCALE = math.pi


@dataclass(frozen=True)
class PhotonicQnnSpec:
    """Structural description of the photonic QNN.

    Attributes
    ----------
    n_features : int
        Number of encoded classical features (one phase shifter each).
    n_classes : int
        Number of output classes.
    n_modes : int
        Number of optical modes. The photonic analogue of the paper's
        ``n features -> n qubits`` rule is ``n features -> n modes``.
    n_photons : int
        Number of input photons. Must be at least 2 for the circuit to be a
        non-trivial (interfering) photonic model.
    n_layers : int
        Number of ansatz layers, where one layer is a trainable phase column
        followed by a fixed mixing column. This is the photonic counterpart of
        one ``RealAmplitudes`` rotation layer plus its entangler.
    encoding_scale : float
        Multiplier applied to a scaled feature to obtain its encoding phase.
    computation_space : str
        MerLin computation space; ``unbunched`` corresponds to threshold
        detectors.
    input_state : list[int]
        Explicit photon occupation of the input modes.
    """

    n_features: int
    n_classes: int
    n_modes: int
    n_photons: int
    n_layers: int
    encoding_scale: float
    computation_space: str = "unbunched"
    input_state: list[int] = field(default_factory=list)

    def resolved_input_state(self) -> list[int]:
        """Return entry state or generate a new state."""

        if self.input_state:
            state = list(self.input_state)
            if len(state) != self.n_modes:
                raise ValueError(
                    f"incorrect dimensions : the entry state {state} needs {len(state)} modes, "
                    f"the circuit has {self.n_modes} modes."
                )
            if sum(state) != self.n_photons:
                raise ValueError(
                    f"Photon number incorrect : state {state} contains {sum(state)} photons, "
                    f"but we need {self.n_photons} photons."
                )
            return state

        if self.n_photons > self.n_modes:
            raise ValueError(
                f"impossible to distribute {self.n_photons} photons in "
                f"{self.n_modes} modes without bunch."
            )

        state = [0] * self.n_modes
        for i in range(self.n_photons):
            index = int(i * (self.n_modes / self.n_photons))
            state[index] = 1

        return state

    def as_dict(self) -> dict:
        """Returns a JSON description of the model."""
        return {
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "n_modes": self.n_modes,
            "n_photons": self.n_photons,
            "n_layers": self.n_layers,
            "encoding_scale": self.encoding_scale,
            "computation_space": self.computation_space,
            "input_state": self.resolved_input_state(),
        }


def build_spec_ir(spec: PhotonicQnnSpec) -> CircuitIR:
    """Return the IR of the unmutated circuit described by ``spec``."""

    return build_original_ir(
        n_modes=spec.n_modes,
        n_features=spec.n_features,
        n_layers=spec.n_layers,
    )


class PhotonicQnnClassifier(nn.Module):
    """Photonic QNN classifier built from a circuit IR.

    Parameters
    ----------
    spec : PhotonicQnnSpec
        Structural description used for the measurement, the input state, and
        the encoding scale.
    ir : CircuitIR | None
        Circuit to realise. If omitted, the unmutated circuit of ``spec`` is
        built. Mutants pass their own IR here. Default value is None.

    Raises
    ------
    ValueError
        If fewer than two photons are requested, if the outcome-space size is
        not divisible by the class count, or if the IR does not encode the
        number of features declared by ``spec``.
    """

    def __init__(self, spec: PhotonicQnnSpec, ir: CircuitIR | None = None) -> None:
        super().__init__()
        if spec.n_photons < 2:
            raise ValueError(
                "A photonic reproduction needs at least 2 photons; a "
                "single-photon circuit is a trivial linear-optical baseline."
            )
        self.spec = spec
        self.ir = ir.copy() if ir is not None else build_spec_ir(spec)
        # Mutants may drop an encoding element, so the model binds exactly the
        # feature columns the circuit still consumes rather than assuming
        # range(n_features).
        self.encoding_order = self.ir.encoding_feature_order()
        if not self.encoding_order:
            raise ValueError(
                "The circuit encodes no input feature; it cannot classify data."
            )
        if max(self.encoding_order) >= spec.n_features:
            raise ValueError(
                f"IR references feature {max(self.encoding_order)} but the spec "
                f"declares only {spec.n_features} features."
            )

        circuit, weight_names, _ = self.ir.to_pcvl()
        self.quantum = ml.QuantumLayer(
            circuit=circuit,
            input_state=spec.resolved_input_state(),
            n_photons=spec.n_photons,
            trainable_parameters=["theta"] if weight_names else [],
            input_parameters=["px"],
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.coerce(spec.computation_space),
            ),
        )
        self._load_weights_from_ir()

        outcomes = self.quantum.output_size
        if outcomes % spec.n_classes != 0:
            raise ValueError(
                f"LexGrouping needs n_classes to divide the outcome count; "
                f"{outcomes} outcomes cannot be folded into {spec.n_classes} "
                "classes without dropping outcomes."
            )
        self.grouping = ml.LexGrouping(outcomes, spec.n_classes)

    def _load_weights_from_ir(self) -> None:
        """Copy the IR's weight angles into the layer's parameter tensor."""

        if not self.ir.weights_initialised():
            # Freshly built original circuit: keep MerLin's seeded phase
            # initialisation instead of overwriting it with placeholders.
            return
        weights = self.ir.weights()
        with torch.no_grad():
            for _, parameter in self.quantum.named_parameters():
                parameter.copy_(torch.tensor(weights, dtype=parameter.dtype))
                break

    def export_ir(self) -> CircuitIR:
        """Return the circuit IR carrying the model's current weights."""

        if not self.ir.weight_positions():
            return self.ir.copy()
        for _, parameter in self.quantum.named_parameters():
            return set_weights(self.ir, parameter.detach().flatten().tolist())
        return self.ir.copy()

    def num_trainable_parameters(self) -> int:
        """Return the number of trainable photonic phases."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, shots: int = 0) -> torch.Tensor:
        """Return per-class probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Scaled features of shape ``(batch, n_features)``. They are
            multiplied by ``spec.encoding_scale`` to obtain encoding phases,
            which is the photonic counterpart of the ZFeatureMap's ``P(2*x)``.
        shots : int
            Number of samples drawn from the outcome distribution. ``0`` means
            the analytic distribution. Sampling is only honoured in eval mode
            under ``torch.no_grad()``. Default value is 0.

        Returns
        -------
        torch.Tensor
            Class probabilities of shape ``(batch, n_classes)``.
        """

        phases = x[:, self.encoding_order] * self.spec.encoding_scale
        outcome_probs = (
            self.quantum(phases) if shots == 0 else self.quantum(phases, shots=shots)
        )
        return self.grouping(outcome_probs)

    def hardware_report(self, *, shots: int) -> dict:
        """Return the hardware-aware fields required for MerLin result tables.

        Parameters
        ----------
        shots : int
            Shot budget used when producing the reported predictions.

        Returns
        -------
        dict
            Hardware-aware reporting fields.
        """

        return {
            "computation_space": self.spec.computation_space,
            "detector_model": "threshold (unbunched outcome space)",
            "n_photons": self.spec.n_photons,
            "n_modes": self.spec.n_modes,
            "input_state": self.spec.resolved_input_state(),
            "encoding": (
                f"angle encoding on modes 0-{self.spec.n_features - 1}, "
                f"scale={self.spec.encoding_scale:.6g} rad, preceded by a fixed "
                "50:50 beam-splitter column"
            ),
            "measurement_strategy": (
                "MeasurementStrategy.probs(computation_space="
                f"{self.spec.computation_space}) + LexGrouping"
                f"({self.quantum.output_size}, {self.spec.n_classes})"
            ),
            "postselection": "none",
            "simulator": "MerLin CPU simulator (Perceval backend)",
            "shots": shots,
            "outcome_space_size": self.quantum.output_size,
            "trainable_parameters": self.num_trainable_parameters(),
            "n_components": len(self.ir.elements),
            "n_encoded_features": len(self.encoding_order),
        }


def gate_model_parameter_count(n_qubits: int, n_rotation_layers: int) -> int:
    """Return the parameter count of the equivalent ``RealAmplitudes`` ansatz.

    Used to document that the photonic adaptation is iso-parameter with the
    gate-model ansatz it replaces.

    Parameters
    ----------
    n_qubits : int
        Number of qubits of the gate-model circuit.
    n_rotation_layers : int
        Number of RY rotation layers (``reps + 1`` in Qiskit's convention).

    Returns
    -------
    int
        Number of trainable parameters.
    """

    return n_qubits * n_rotation_layers


class MutatableQCNN(QCNNClassifier):
    """QCNN Classifier wrapped to support topological mutation testing via IR."""

    def __init__(self, input_shape, num_classes, stages=None, ir_dict=None):
        super().__init__(
            input_shape=input_shape, num_classes=num_classes, stages=stages
        )

        self.ir_dict = ir_dict

        if self.ir_dict is not None:
            self._apply_mutated_ir()

    def export_ir(self) -> dict:
        """Export the optical structure into a dict"""
        if self.ir_dict is not None:
            return self.ir_dict

        extracted_ir = {}
        for idx, layer in enumerate(self.layers):
            target_circuit = None
            if hasattr(layer, "circuit") and layer.circuit is not None:
                target_circuit = layer.circuit
            elif hasattr(layer, "processor") and layer.processor.circuit is not None:
                target_circuit = layer.processor.circuit

            if target_circuit is None:
                continue

            elements = []
            for modes, component in target_circuit._components:
                if isinstance(component, pcvl.PS):
                    val = (
                        float(component.assign({}).phi)
                        if hasattr(component, "phi")
                        else 0.0
                    )
                    elements.append(
                        Element(
                            kind="ps",
                            modes=modes,
                            role="weight",
                            value=val,
                            feature_index=None,
                            block=BLOCK_ANSATZ_PHASE,
                            layer=0,
                        )
                    )

                elif isinstance(component, pcvl.BS):
                    val = (
                        float(component.assign({}).theta)
                        if hasattr(component, "theta")
                        else 0.0
                    )
                    elements.append(
                        Element(
                            kind="bs",
                            modes=modes,
                            role="fixed",
                            value=val,
                            feature_index=None,
                            block=BLOCK_ANSATZ_MIX,
                            layer=0,
                        )
                    )

            extracted_ir[idx] = CircuitIR(n_modes=target_circuit.m, elements=elements)

        return extracted_ir

    def forward(self, x, shots=None, **kwargs):
        """
        Extension of the forward method to intercept the 'shots' argument.
        MerLin's QCNN analytically calculates exact probabilities by default.
        """
        logits = super().forward(x)

        return logits

    def num_trainable_parameters(self) -> int:
        """Returns the total number of trainable parameters of the PyTorch model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _apply_mutated_ir(self):
        """Replaces the native optical circuits with the mutant's."""
        for layer_idx, ir_obj in self.ir_dict.items():
            layer = self.layers[int(layer_idx)]

            mutated_circuit, _, _ = ir_obj.to_pcvl()

            if hasattr(layer, "circuit"):
                layer.circuit = mutated_circuit
            elif hasattr(layer, "processor"):
                layer.processor.set_circuit(mutated_circuit)
            else:
                raise AttributeError(f"No circuit on layer {layer_idx}")

    def hardware_report(self, *, shots: int) -> dict:
        """Return hardware metadatas"""
        return {
            "architecture": "Photonic QCNN (Mutatable)",
            "simulator": "MerLin CPU simulator (Perceval backend)",
            "shots": shots,
            "trainable_parameters": self.num_trainable_parameters(),
            "n_layers": len(self.layers),
            "measurement_strategy": "QCNN native grouping",
            "postselection": "none",
        }
