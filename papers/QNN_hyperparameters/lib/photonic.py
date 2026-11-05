"""Photonic-native (MerLin) classifier — Phase 4 adaptation, not the paper.

The model asks the paper's question (which hyperparameters of a variational
classifier matter, and is the "beta initialization" effect real) with
photonic hyperparameters, instead of imitating the Qiskit circuits (LOG.md,
"Amendment — photonic-native design"):

- ``m = 2d`` modes for ``d`` features; ``n`` photons on modes
  ``floor(k * m / n)``;
- data reuploading: a trainable MZI mesh, then ``depth`` x [phase
  ``pi * x_i`` on mode ``2i``, trainable mesh]. All meshes share the
  parameter prefix ``mesh``, so MerLin registers one phase tensor;
- readout: per-mode photon expectations under threshold detection
  (``ComputationSpace.UNBUNCHED``, ``m`` values) into a trainable linear head;
- training: Adam on exact expectations, softmax cross-entropy.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .qnn import TrainingResult, initial_point

logger = logging.getLogger(__name__)

# Encoded phase = ENCODING_SCALE * x: MinMax inputs in [0, 1] span [0, pi].
ENCODING_SCALE = math.pi
PHOTONIC_INITS = ("merlin_default", "beta", "zeros")


def spread_input_state(n_modes: int, n_photons: int) -> list[int]:
    """One photon on each mode ``floor(k * n_modes / n_photons)``."""

    if not 2 <= n_photons <= n_modes:
        raise ValueError(
            f"need 2 <= n_photons <= n_modes, got {n_photons} photons in {n_modes} modes"
        )
    state = [0] * n_modes
    for photon in range(n_photons):
        state[photon * n_modes // n_photons] = 1
    return state


def build_photonic_layer(
    n_features: int, n_photons: int, depth: int, device: torch.device
):
    """Reuploading circuit on ``2 * n_features`` modes with mode-expectation readout."""

    import merlin as ml

    n_modes = 2 * n_features
    builder = ml.CircuitBuilder(n_modes=n_modes)
    builder.add_entangling_layer(name="mesh")
    for _ in range(depth):
        # One encoder name across repetitions: MerLin numbers the inputs
        # contiguously in builder order, so the input is x tiled ``depth`` times.
        builder.add_angle_encoding(
            modes=list(range(0, n_modes, 2)), name="px", scale=ENCODING_SCALE
        )
        builder.add_entangling_layer(name="mesh")
    input_state = spread_input_state(n_modes, n_photons)
    layer = ml.QuantumLayer(
        input_size=n_features * depth,
        builder=builder,
        input_state=input_state,
        n_photons=n_photons,
        measurement_strategy=ml.MeasurementStrategy.mode_expectations(
            ml.ComputationSpace.UNBUNCHED
        ),
        device=device,
    )
    # merlin 0.4.1: mode_expectations keeps its marginalisation mask on the
    # CPU even when the layer is built on CUDA (device mismatch in
    # ``marginalize_per_mode``). TODO: drop once MerLin moves the mask.
    layer.measurement_mapping.mask = layer.measurement_mapping.mask.to(device)
    return layer, input_state


class PhotonicClassifier:
    """Photonic reuploading classifier with a linear readout head.

    Parameters
    ----------
    model_cfg : dict
        ``init`` (one of ``PHOTONIC_INITS``).
    photonic_cfg : dict
        ``photons`` and ``depth``.
    n_features : int
        Number of input features; the circuit has twice as many modes.
    device : str
        Torch device for the MerLin simulation, e.g. ``cpu`` or ``cuda:0``.
    rng : numpy.random.Generator
        Source of the ``beta`` initial phases.
    """

    def __init__(
        self,
        model_cfg: dict,
        photonic_cfg: dict,
        n_features: int,
        device: str,
        rng: np.random.Generator,
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"device {device!r} requested but CUDA is not available")
        if model_cfg["init"] not in PHOTONIC_INITS:
            raise ValueError(
                f"photonic init must be one of {PHOTONIC_INITS}, got {model_cfg['init']!r}"
            )
        self.init = model_cfg["init"]
        self.n_features = n_features
        self.depth = int(photonic_cfg["depth"])
        self.rng = rng
        self.layer, self.input_state = build_photonic_layer(
            n_features, int(photonic_cfg["photons"]), self.depth, self.device
        )
        self.phases = self.layer.get_parameter("mesh")
        self.num_phases = self.phases.numel()
        self.n_modes = len(self.input_state)
        self.n_photons = sum(self.input_state)
        self.head = None
        self.classes = None
        self.progress_hook = None

    @property
    def num_weights(self) -> int:
        """Trainable phases plus head weights (head exists after ``fit``)."""

        head = sum(p.numel() for p in self.head.parameters()) if self.head else 0
        return self.num_phases + head

    def _inputs(self, X: np.ndarray) -> torch.Tensor:
        tiled = np.tile(np.asarray(X, dtype=float), (1, self.depth))
        return torch.as_tensor(tiled, dtype=self.phases.dtype, device=self.device)

    def logits(self, inputs: torch.Tensor) -> torch.Tensor:
        """Class logits, shape ``(batch, num_classes)``."""

        return self.head(self.layer(inputs))

    def _initialise_phases(self, X: np.ndarray) -> dict:
        if self.init == "merlin_default":
            # MerLin already drew randn * pi from the seeded torch generator.
            return {"distribution": "randn * pi (MerLin default)"}
        start, details = initial_point(self.init, self.num_phases, 1, X, self.rng)
        with torch.no_grad():
            self.phases.copy_(torch.as_tensor(start, dtype=self.phases.dtype))
        return details

    def fit(self, X: np.ndarray, y: np.ndarray, training_cfg: dict) -> TrainingResult:
        """Adam on softmax cross-entropy; log every step."""

        if training_cfg["optimizer"] != "Adam":
            raise ValueError(
                f"photonic runs train with Adam, got {training_cfg['optimizer']!r}"
            )
        self.classes = np.unique(y)
        self.head = torch.nn.Linear(self.n_modes, len(self.classes)).to(
            device=self.device, dtype=self.phases.dtype
        )
        details = self._initialise_phases(X)
        result = TrainingResult(
            weights=self.phases.detach().cpu().numpy().astype(float),
            initial_point_details=details,
            classes=self.classes,
        )
        inputs = self._inputs(X)
        targets = torch.as_tensor(np.searchsorted(self.classes, y), device=self.device)
        steps = int(training_cfg["steps"])
        optimizer = torch.optim.Adam(
            [self.phases, *self.head.parameters()], lr=float(training_cfg["lr"])
        )
        began = time.perf_counter()
        for _ in range(steps):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(self.logits(inputs), targets)
            loss.backward()
            optimizer.step()
            result.evaluations += 1
            result.losses.append(float(loss.detach()))
            logger.info(
                "TRAIN_STEP_COMPLETED | step=%d | budget=%d | loss=%.6f",
                result.evaluations,
                steps,
                result.losses[-1],
            )
            if self.progress_hook is not None:
                self.progress_hook(result.evaluations)
        result.fit_seconds = time.perf_counter() - began
        result.weights = self.phases.detach().cpu().numpy().astype(float)
        return result

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Accuracy and weighted F1 from exact expectations."""

        with torch.no_grad():
            logits = self.logits(self._inputs(X)).cpu().numpy()
        predictions = self.classes[np.argmax(logits, axis=1)]
        return {
            "accuracy": float(accuracy_score(y, predictions)),
            "f1_weighted": float(f1_score(y, predictions, average="weighted")),
        }
