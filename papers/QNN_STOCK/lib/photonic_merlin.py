"""MerLin photonic ADAPTATION of the context-conditional predictor (Phase 4).

This is **not** the paper's method. The paper passes a superposition of all
contexts through a gate circuit and trains with a SWAP-test loss (QBGU).
None of that is kept here: the model below only keeps the task of claim C1,
``q(up | context)``, and feeds the ``2**T`` contexts as one classical batch.

Circuit (``T + 1`` modes, MerLin ``CircuitBuilder``)::

    U0 -> phases pi * c -> U1 -> phases pi * c -> ... -> U_L

- ``U_l``: trainable MZI meshes.
- Phases: context bit ``c_i`` (oldest step first) puts a phase ``pi * c_i``
  on mode ``i + 1``. Mode 0 never carries a data phase; it is the reference
  the other modes interfere with.
- Readout: ``q(up | c)`` is the probability that the detector on mode 0
  detects at least one photon. No trainable classical layer follows.

The readout is a detection event on purpose. The mean photon number of a
mode equals the sum of the single-photon probabilities of the input photons,
so a readout linear in the occupations gives the same model for one photon,
for distinguishable photons and for indistinguishable photons.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any

import merlin as ml
import numpy as np
import perceval as pcvl
import torch
from torch import nn

from .baselines import context_features

logger = logging.getLogger(__name__)

COMPUTATION_SPACES = {
    "fock": ml.ComputationSpace.FOCK,
    "unbunched": ml.ComputationSpace.UNBUNCHED,
}
READOUT_MODE = 0


def spread_input_state(n_modes: int, n_photons: int) -> list[int]:
    """Collision-free Fock input with the photons spread evenly over the modes.

    Photon ``k`` enters mode ``round(k * n_modes / n_photons)``.

    Raises
    ------
    ValueError
        If the photons do not fit in distinct modes.
    """

    if n_photons < 1 or n_photons > n_modes:
        raise ValueError(
            f"n_photons must be in [1, n_modes={n_modes}], got {n_photons}"
        )
    positions = [round(k * n_modes / n_photons) for k in range(n_photons)]
    if len(set(positions)) != n_photons or max(positions) >= n_modes:
        raise ValueError(
            f"Cannot spread {n_photons} photons over {n_modes} modes: {positions}"
        )
    state = [0] * n_modes
    for position in positions:
        state[position] = 1
    return state


def _build_circuit(n_modes: int, reuploads: int) -> ml.CircuitBuilder:
    builder = ml.CircuitBuilder(n_modes=n_modes)
    for layer in range(reuploads):
        builder.add_entangling_layer(name=f"U{layer}")
        builder.add_angle_encoding(
            modes=list(range(1, n_modes)), name=f"x{layer}", scale=math.pi
        )
    builder.add_entangling_layer(name=f"U{reuploads}")
    return builder


class PhotonicContextModel(nn.Module):
    """Photonic circuit returning ``q(up | context)`` for binary contexts.

    Parameters
    ----------
    context_length : int
        Number of context bits ``T``; the circuit has ``T + 1`` modes.
    n_photons : int
        Number of input photons. One photon is an ablation (wave interference
        only), not a photonic result on its own.
    reuploads : int
        Number of times the context phases are applied.
    computation_space : str
        ``"fock"`` (photon-number-resolving detection, no postselection) or
        ``"unbunched"`` (threshold detection with collision-free
        postselection, which MerLin renormalizes).
    distinguishable : bool
        If True, the photons are made fully distinguishable with a Perceval
        noise model (indistinguishability 0). This removes two-photon
        interference and keeps everything else. Requires ``"fock"``.
    dtype : torch.dtype
        Simulation dtype.

    Notes
    -----
    MerLin draws the mesh phases from the global torch RNG at construction;
    seed it with ``torch.manual_seed`` immediately before building the model.
    """

    def __init__(
        self,
        context_length: int,
        n_photons: int,
        reuploads: int,
        computation_space: str,
        distinguishable: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if computation_space not in COMPUTATION_SPACES:
            raise ValueError(
                f"computation_space must be one of {sorted(COMPUTATION_SPACES)}, "
                f"got {computation_space!r}"
            )
        if reuploads < 1:
            raise ValueError(f"reuploads must be >= 1, got {reuploads}")
        if distinguishable and computation_space != "fock":
            raise ValueError(
                "distinguishable photons need computation_space 'fock' "
                "(source noise is not defined on the unbunched space)"
            )
        self.context_length = context_length
        self.n_modes = context_length + 1
        self.n_photons = n_photons
        self.reuploads = reuploads
        self.computation_space = computation_space
        self.distinguishable = distinguishable
        self.real_dtype = dtype
        self.input_state = spread_input_state(self.n_modes, n_photons)
        self.quantum = self._layer(computation_space, distinguishable)
        # Outcomes with at least one photon on the readout mode.
        self.register_buffer(
            "detects_readout_mode",
            torch.tensor(
                [key[READOUT_MODE] >= 1 for key in self.quantum.output_keys],
                dtype=torch.bool,
            ),
            # Derived from the layer's outcome list, which depends on the
            # computation space; it is not part of the trained state.
            persistent=False,
        )

    def _layer(self, computation_space: str, distinguishable: bool) -> ml.QuantumLayer:
        options: dict[str, Any] = {}
        if distinguishable:
            options["noise"] = pcvl.NoiseModel(indistinguishability=0.0)
        return ml.QuantumLayer(
            input_size=self.context_length * self.reuploads,
            builder=_build_circuit(self.n_modes, self.reuploads),
            input_state=self.input_state,
            n_photons=self.n_photons,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=COMPUTATION_SPACES[computation_space]
            ),
            dtype=self.real_dtype,
            **options,
        )

    def encode(self, context_bits: torch.Tensor) -> torch.Tensor:
        """Repeat the context bits once per re-upload: shape ``(batch, T * L)``."""

        if context_bits.ndim != 2 or context_bits.shape[1] != self.context_length:
            raise ValueError(
                f"Expected context bits of shape (batch, {self.context_length}), "
                f"got {tuple(context_bits.shape)}"
            )
        return context_bits.repeat(1, self.reuploads)

    def forward(self, context_bits: torch.Tensor) -> torch.Tensor:
        """``q(up | c)`` for every row of ``context_bits``, shape ``(batch,)``."""

        probabilities = self.quantum(self.encode(context_bits))
        return probabilities[:, self.detects_readout_mode].sum(dim=1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def keep_rate(self, context_bits: torch.Tensor) -> torch.Tensor:
        """Probability that an ideal shot survives postselection, per context.

        For ``"unbunched"`` this is the collision-free probability, computed
        with a FOCK-space twin that shares the circuit parameters. For
        ``"fock"`` no outcome is discarded.
        """

        if self.computation_space == "fock":
            return torch.ones(context_bits.shape[0], dtype=self.real_dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            twin = self._layer("fock", False)
        twin.load_state_dict(self.quantum.state_dict())
        probabilities = twin(self.encode(context_bits))
        collision_free = torch.tensor([max(key) <= 1 for key in twin.output_keys])
        return probabilities[:, collision_free].sum(dim=1)


def fit_photonic_model(
    counts: np.ndarray,
    photonic_cfg: dict[str, Any],
    seed: int,
    dtype: torch.dtype,
    stage: str,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Train one photonic model on the context counts of one asset.

    The loss is the negative log-likelihood of the next move weighted by the
    context counts, the objective of baseline B2. The counts are a sufficient
    statistic, so this equals training on every window.

    Parameters
    ----------
    counts : numpy.ndarray
        Training counts of shape ``(2**T, 2)``: columns are down and up.
    photonic_cfg : dict
        The ``photonic`` config section: ``n_photons``, ``reuploads``,
        ``computation_space``, ``distinguishable``, ``steps``, ``lr`` and
        ``monitor_every``.
    seed : int
        Seed for the mesh initialization.
    dtype : torch.dtype
        Simulation dtype.
    stage : str
        Label of this training in the history rows and log lines.
    history : list of dict
        Receives one row per monitored step.

    Returns
    -------
    dict
        ``conditional`` (table of shape ``(2**T, 2)``), ``n_params``,
        ``final_train_loss``, ``keep_rate_mean`` (weighted by the training
        context distribution), ``keep_rate_min`` and ``input_state``.

    Raises
    ------
    ValueError
        If the counts are not binary-outcome, or a config key is missing.
    """

    for key in (
        "n_photons",
        "reuploads",
        "computation_space",
        "distinguishable",
        "steps",
        "lr",
        "monitor_every",
    ):
        if key not in photonic_cfg:
            raise ValueError(f"Missing config key photonic.{key}")
    n_contexts, outcome_levels = counts.shape
    if outcome_levels != 2:
        raise ValueError(
            "The photonic adaptation predicts one binary step; got "
            f"{outcome_levels} outcomes"
        )
    context_length = int(round(math.log2(n_contexts)))
    if 2**context_length != n_contexts:
        raise ValueError(f"Number of contexts {n_contexts} is not a power of two")

    torch.manual_seed(seed)
    model = PhotonicContextModel(
        context_length,
        int(photonic_cfg["n_photons"]),
        int(photonic_cfg["reuploads"]),
        str(photonic_cfg["computation_space"]),
        bool(photonic_cfg["distinguishable"]),
        dtype,
    )
    bits = context_features(n_contexts, 2).to(dtype)
    weights = torch.as_tensor(counts / counts.sum(), dtype=dtype)
    steps = int(photonic_cfg["steps"])
    monitor_every = int(photonic_cfg["monitor_every"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(photonic_cfg["lr"]))
    # Guards log(0) when a context's readout probability reaches 0 or 1.
    floor = 1e-12

    def weighted_nll() -> torch.Tensor:
        prob_up = model(bits).clamp(floor, 1.0 - floor)
        return -(
            weights[:, 1] * torch.log(prob_up) + weights[:, 0] * torch.log1p(-prob_up)
        ).sum()

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        loss = weighted_nll()
        loss.backward()
        optimizer.step()
        if step % monitor_every == 0 or step == steps:
            loss_value = float(loss.detach())
            history.append(
                {
                    "stage": stage,
                    "epoch": len(history) + 1,
                    "stage_epoch": step,
                    "loss_estimate": loss_value,
                }
            )
            logger.info(
                "TRAIN_EPOCH_COMPLETED | stage=%s | epoch=%d/%d | loss_estimate=%.6f",
                stage,
                step,
                steps,
                loss_value,
            )

    with torch.no_grad():
        prob_up = model(bits).clamp(floor, 1.0 - floor).numpy()
        final_loss = float(weighted_nll())
        keep = model.keep_rate(bits).numpy()
    context_weights = counts.sum(axis=1) / counts.sum()
    return {
        "conditional": np.column_stack([1.0 - prob_up, prob_up]),
        "n_params": model.n_params(),
        "final_train_loss": final_loss,
        "keep_rate_mean": float(keep @ context_weights),
        "keep_rate_min": float(keep.min()),
        "input_state": model.input_state,
    }
