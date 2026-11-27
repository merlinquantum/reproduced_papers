"""Model construction and the paper's training protocol for the fitting experiments.

Protocol (Fig. 4 caption and reference notebook): Adam with learning rate 0.3,
mini-batches of 25 points drawn with replacement from the full grid, a fixed
step budget (200 steps in the paper, no early stopping, LOG.md D2), every
angle initialised from ``U[0, 2 pi)``. The notebook's loss is ``0.5 * MSE``;
plain MSE is used here because Adam is invariant to a constant loss scale (up
to its ``eps``), and both values are recorded.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .baselines import FourierSeriesModel
from .circuits import ParallelPauliModel, SerialPauliModel
from .data import build_dataloader

LOGGER = logging.getLogger(__name__)

MODEL_KINDS = ("quantum", "fourier_classical", "photonic")


def model_frequencies(model_cfg: Mapping[str, Any]) -> tuple[list[int], float]:
    """Return the positive frequencies and scaling ``s`` of a model's spectrum.

    Serial and parallel ``RX`` repetitions give ``{-r..r}`` in units of ``s``
    (paper Section III.B). A photonic model (``layout == "photonic"``) gets
    its spectrum from the photon-number generator (``lib.photonic_merlin``).
    Baselines B1 and B2 use the same set.
    """
    scaling = float(model_cfg["encoding_scaling"])
    if model_cfg["layout"] == "photonic":
        # Imported here so that qubit-only runs do not need MerLin.
        from .photonic_merlin import photonic_positive_frequencies

        return photonic_positive_frequencies(model_cfg), scaling
    repetitions = int(model_cfg["repetitions"])
    return list(range(1, repetitions + 1)), scaling


def build_fit_model(
    kind: str, model_cfg: Mapping[str, Any], dtype: torch.dtype
) -> nn.Module:
    """Build the model for one fitting case.

    Parameters
    ----------
    kind : str
        ``"quantum"`` for the paper's circuit, ``"fourier_classical"`` for
        baseline B2 with the same frequency set, ``"photonic"`` for the
        Phase 4 MerLin adaptation.
    model_cfg : Mapping
        Keys ``layout`` (``"serial"``, ``"parallel"`` or ``"photonic"``) and
        ``encoding_scaling``. Qubit layouts need ``repetitions`` and
        ``readout`` (``"z0"`` or ``"trainable_linear"``); parallel
        models also need ``ansatz`` and ``ansatz_layers``; photonic models need
        the keys of ``lib.photonic_merlin.build_photonic_model``.
    dtype : torch.dtype
        Real parameter dtype.

    Returns
    -------
    torch.nn.Module
        Model mapping inputs ``(batch,)`` to outputs ``(batch,)``.

    Raises
    ------
    ValueError
        If ``kind`` or ``layout`` is unknown.
    KeyError
        If a required model key is missing.
    """
    if kind == "fourier_classical":
        frequencies, scaling = model_frequencies(model_cfg)
        return FourierSeriesModel(
            frequencies, scaling, model_cfg["coefficient_init"], dtype
        )
    if kind == "photonic":
        if model_cfg["layout"] != "photonic":
            raise ValueError("model kind 'photonic' needs layout 'photonic'")
        from .photonic_merlin import build_photonic_model

        return build_photonic_model(model_cfg, dtype)
    if kind != "quantum":
        raise ValueError(f"Unknown model kind {kind!r}; expected one of {MODEL_KINDS}")
    layout = model_cfg["layout"]
    repetitions = int(model_cfg["repetitions"])
    scaling = float(model_cfg["encoding_scaling"])
    readout = model_cfg["readout"]
    if layout == "serial":
        return SerialPauliModel(repetitions, scaling, readout, dtype)
    if layout == "parallel":
        return ParallelPauliModel(
            repetitions,
            scaling,
            model_cfg["ansatz"],
            int(model_cfg["ansatz_layers"]),
            readout,
            dtype,
        )
    raise ValueError(f"Unknown layout {layout!r}; expected 'serial' or 'parallel'")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable real parameters."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


OPTIMIZERS = ("adam", "adam_cosine", "lbfgs")


def _build_optimizer(
    name: str, model: nn.Module, learning_rate: float, steps: int, full_batch: bool
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    """Return the optimiser (and optional schedule) named by ``training.optimizer``.

    - ``"adam"``: the paper's Adam with constant learning rate.
    - ``"adam_cosine"``: Adam with the learning rate cosine-annealed to 0 over
      ``steps`` (Phase 5 extension, protocol P4).
    - ``"lbfgs"``: L-BFGS with strong-Wolfe line search, one iteration per step,
      history 20 (Phase 5 extension, protocol P5). It needs full-batch training,
      because the line search assumes a deterministic objective.

    Raises
    ------
    ValueError
        If the name is unknown, or L-BFGS is requested with mini-batches.
    """
    parameters = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate), None
    if name == "adam_cosine":
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps
        )
    if name == "lbfgs":
        if not full_batch:
            raise ValueError("lbfgs needs training.full_batch = true")
        optimizer = torch.optim.LBFGS(
            parameters,
            lr=learning_rate,
            max_iter=1,
            history_size=20,
            line_search_fn="strong_wolfe",
        )
        return optimizer, None
    raise ValueError(f"Unknown optimizer {name!r}; expected one of {OPTIMIZERS}")


def train_fit_model(
    model: nn.Module,
    case_cfg: Mapping[str, Any],
    seed: int,
    device: torch.device,
    run_label: str,
) -> dict:
    """Train one model on one case with the paper's protocol.

    The same ``seed`` fixes both the parameter initialisation and the batch
    sampler, so a quantum model and baseline B2 trained with one seed see the
    same sequence of mini-batches.

    Parameters
    ----------
    model : torch.nn.Module
        Model with a ``reset_parameters(generator, high)`` method.
    case_cfg : Mapping
        Keys ``grid``, ``target`` and ``training`` (``optimizer``, ``lr``,
        ``steps``, ``init_high``, ``full_batch``, and ``batch_size`` for
        mini-batches).
    seed : int
        Seed of the initialisation and of the batch sampler.
    device : torch.device
        Device for the model and data.
    run_label : str
        Identifier used in log lines.

    Returns
    -------
    dict
        ``full_grid_mse`` per step (index 0 is before training),
        ``final_mse``, ``final_half_mse`` (the reference code's loss) and the
        final ``predictions`` on the grid.
    """
    training_cfg = case_cfg["training"]
    steps = int(training_cfg["steps"])
    learning_rate = float(training_cfg["lr"])
    init_generator = torch.Generator().manual_seed(seed)
    model.reset_parameters(init_generator, float(training_cfg["init_high"]))
    model.to(device)

    eval_inputs, eval_targets = next(iter(build_dataloader(case_cfg, "eval")))
    eval_inputs, eval_targets = eval_inputs.to(device), eval_targets.to(device)
    train_loader = build_dataloader(case_cfg, "train", seed=seed)
    optimizer_name = training_cfg["optimizer"]
    optimizer, scheduler = _build_optimizer(
        optimizer_name, model, learning_rate, steps, bool(training_cfg["full_batch"])
    )

    def full_grid_mse() -> float:
        with torch.no_grad():
            return float(torch.mean((model(eval_inputs) - eval_targets) ** 2))

    history = [full_grid_mse()]
    for step, (batch_inputs, batch_targets) in enumerate(train_loader, start=1):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        # The batch is bound explicitly; L-BFGS may call the closure several times per step.
        def closure(
            inputs: torch.Tensor = batch_inputs, targets: torch.Tensor = batch_targets
        ) -> torch.Tensor:
            optimizer.zero_grad()
            loss = torch.mean((model(inputs) - targets) ** 2)
            loss.backward()
            return loss

        batch_loss = optimizer.step(closure)
        if scheduler is not None:
            scheduler.step()
        history.append(full_grid_mse())
        LOGGER.info(
            "TRAIN_STEP_COMPLETED | %s | step=%d/%d | batch_mse=%.6e | grid_mse=%.6e",
            run_label,
            step,
            steps,
            float(batch_loss),
            history[-1],
        )
    if len(history) != steps + 1:
        raise RuntimeError(f"Expected {steps} optimizer steps, ran {len(history) - 1}")

    with torch.no_grad():
        predictions = model(eval_inputs).cpu()
    return {
        "full_grid_mse": history,
        "final_mse": history[-1],
        "final_half_mse": 0.5 * history[-1],
        "predictions": predictions,
    }
