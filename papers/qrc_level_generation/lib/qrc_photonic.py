"""Photonic reservoir built with MerLin for the level-generation paper.

The reservoir replaces the gate-based ``QubitQRC`` while preserving the
input/feedback structure (per-step embedding of ``x_t`` and ``h_t`` followed by
a fixed random linear-optical mesh whose measurement probabilities are fed
back). The encoding is angle-based via :func:`merlin.CircuitBuilder.add_angle_encoding`,
which closely mirrors the paper's ``Ry`` parametrisation.

Computation space: ``UNBUNCHED`` with threshold detection (default), matching
near-term photonic hardware. The output probability vector has dimension
``C(n_modes, n_photons)``.
"""

from __future__ import annotations

import merlin as ml
import numpy as np
import torch


class PhotonicQRC:
    """Photonic reservoir computing layer.

    Both the input feature and the hidden state are angle-encoded onto the
    same modes after fixed random linear projections (the reservoir analogue
    of the paper's fixed gate-based encoding). A trainable-looking MerLin
    layer is used in eval mode with parameters fixed at construction time
    so that only the downstream FNN learns (consistent with reservoir
    computing).
    """

    def __init__(
        self,
        num_features: int,
        n_modes: int = 6,
        n_photons: int = 3,
        input_scale: float = 1.0,
        feedback_scale: float = 1.0,
        seed: int = 0,
        n_post_layers: int = 2,
    ):
        if n_modes <= 1:
            raise ValueError("Need at least 2 modes for an entangling mesh")
        if n_post_layers < 1:
            raise ValueError("Need at least one post-encoding entangling layer")
        self.num_features = int(num_features)
        self.n_modes = int(n_modes)
        self.n_photons = int(n_photons)
        self.input_scale = float(input_scale)
        self.feedback_scale = float(feedback_scale)

        rng = np.random.default_rng(seed)

        # Photons distributed across modes (dual-rail style when possible).
        input_state = [0] * n_modes
        for k in range(n_photons):
            input_state[(2 * k) % n_modes] = 1
        # Ensure exactly ``n_photons`` photons after wrapping.
        while sum(input_state) < n_photons:
            for i in range(n_modes):
                if input_state[i] == 0:
                    input_state[i] = 1
                    if sum(input_state) >= n_photons:
                        break
        while sum(input_state) > n_photons:
            for i in range(n_modes):
                if input_state[i] == 1:
                    input_state[i] = 0
                    if sum(input_state) <= n_photons:
                        break
        self.input_state = input_state

        builder = ml.CircuitBuilder(n_modes=n_modes)
        builder.add_entangling_layer()
        builder.add_angle_encoding(scale=float(np.pi))
        # Post-encoding mesh depth is configurable. For a frozen reservoir
        # this is not an expressivity knob: consecutive passive meshes with
        # no data injection between them compose into a single equivalent
        # interferometer, so n_post_layers only changes the random draw.
        # The default of 2 matches the committed results; a 1-vs-2
        # comparison (see README) confirms metrics agree within seed noise.
        for _ in range(int(n_post_layers)):
            builder.add_entangling_layer()

        # merlin >= 0.4: the computation space is owned by the measurement
        # strategy factory and the photon count is inferred from input_state.
        self._layer = ml.QuantumLayer(
            builder=builder,
            input_state=input_state,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.UNBUNCHED,
            ),
        )
        # Freeze: this is a reservoir.
        for p in self._layer.parameters():
            p.requires_grad = False
            with torch.no_grad():
                p.copy_(
                    torch.tensor(
                        rng.uniform(-np.pi, np.pi, size=p.shape), dtype=p.dtype
                    )
                )

        self.output_dim = int(self._layer.output_size)

        # Fixed random projections from one-hot feature (num_features) and
        # hidden state (output_dim) to the n_modes angle vector.
        self.input_projection = rng.normal(
            0.0, 1.0, size=(self.num_features, self.n_modes)
        )
        self.hidden_projection = rng.normal(
            0.0, 1.0, size=(self.output_dim, self.n_modes)
        )

    def initial_hidden(self) -> np.ndarray:
        return np.full(self.output_dim, 1.0 / self.output_dim, dtype=np.float64)

    def step(self, x_t: int, h_t: np.ndarray) -> np.ndarray:
        onehot = np.zeros(self.num_features, dtype=np.float64)
        onehot[int(x_t)] = 1.0
        angles = self.input_scale * (onehot @ self.input_projection)
        angles += self.feedback_scale * (h_t @ self.hidden_projection)
        # Wrap to [-pi, pi] before passing into MerLin (which scales by pi internally).
        angles_wrapped = np.mod(angles + np.pi, 2 * np.pi) - np.pi
        angles_norm = angles_wrapped / np.pi

        with torch.no_grad():
            x = torch.tensor(angles_norm, dtype=torch.float32).unsqueeze(0)
            probs = self._layer(x).detach().cpu().numpy()[0]

        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if total > 0:
            probs /= total
        return probs.astype(np.float64)
