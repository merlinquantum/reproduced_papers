"""SPSA training loops for the context loader, QSTL and QMTL (paper Sec. III-V).

One epoch is one SPSA iteration (two loss evaluations), as in the paper. The
exact (shot-free) fidelity is recorded for monitoring only, every
``monitor_every`` epochs and at the last epoch; it never feeds back into the
optimizer.

``training.optimizer`` is ``spsa`` (paper) or, for QSTL only, ``lbfgs`` on
the shot-free loss (E14c). ``training.mode`` selects the QSTL objective (E12):
``qbgu`` is the paper's superposed batch loss (Eq. 20); ``per_sample`` draws one
training window per SPSA iteration and uses its one-hot target (Eq. 18-19).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from .circuits import (
    apply_ansatz,
    apply_context_controlled,
    apply_context_entangling,
    apply_context_sequential,
    apply_share_and_specify,
    n_context_controlled_params,
    n_context_entangling_params,
    n_context_sequential_params,
    n_params,
)
from .model import ContextualTask, loader_mse
from .optim import spsa_step
from .statevector import apply_1q, basis_state, probabilities

logger = logging.getLogger(__name__)

History = list[dict[str, Any]]

HADAMARD = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)


def initial_params(
    count: int, rng: np.random.Generator, max_angle: float = 2.0 * np.pi
) -> np.ndarray:
    """Draw initial angles uniformly from ``[0, max_angle]``.

    The paper does not state the initialization; ``2 pi`` is the default used
    for every paper experiment. ``training.init_max_angle`` lowers it for the
    context-preserving circuits (E14): with additive angles that all start
    small and positive, every amplitude starts non-negative, the sign of the
    target's amplitudes. Default value is ``2 pi``.
    """

    return rng.uniform(0.0, max_angle, size=count)


def _shots(training_cfg: dict[str, Any]) -> int | None:
    shots = training_cfg["shots"]
    return None if shots is None else int(shots)


def run_spsa(
    loss_fn: Callable[[np.ndarray], float] | None,
    monitor_fn: Callable[[np.ndarray], dict[str, float]],
    params: np.ndarray,
    epochs: int,
    training_cfg: dict[str, Any],
    rng: np.random.Generator,
    stage: str,
    history: History,
    monitor_every: int = 1,
    draw_loss_fn: Callable[[], Callable[[np.ndarray], float]] | None = None,
) -> np.ndarray:
    """Run ``epochs`` SPSA iterations.

    Parameters
    ----------
    loss_fn : callable or None
        Fixed loss used in every iteration. Exactly one of ``loss_fn`` and
        ``draw_loss_fn`` must be given.
    draw_loss_fn : callable or None
        Called once per iteration to get that iteration's loss, so both SPSA
        evaluations see the same sampled window (per-sample SGD). Default
        value is None.
    monitor_every : int
        Record a history row (and log at INFO) every this many iterations and
        at the last one. ``loss_estimate_mean`` averages the SPSA loss
        estimates since the previous row. Default value is 1.
    """

    if (loss_fn is None) == (draw_loss_fn is None):
        raise ValueError("pass exactly one of loss_fn and draw_loss_fn")
    if monitor_every < 1:
        raise ValueError("monitor_every must be at least 1")
    lr = float(training_cfg["lr"])
    perturbation = float(training_cfg["spsa_perturbation"])
    interval_losses: list[float] = []
    for epoch in range(1, epochs + 1):
        step_loss_fn = loss_fn if draw_loss_fn is None else draw_loss_fn()
        params, loss_estimate = spsa_step(step_loss_fn, params, lr, perturbation, rng)
        interval_losses.append(loss_estimate)
        if epoch % monitor_every and epoch != epochs:
            continue
        monitor = monitor_fn(params)
        history.append(
            {
                "stage": stage,
                "epoch": len(history) + 1,
                "stage_epoch": epoch,
                "loss_estimate": loss_estimate,
                "loss_estimate_mean": float(np.mean(interval_losses)),
                **monitor,
            }
        )
        interval_losses = []
        logger.info(
            "TRAIN_EPOCH_COMPLETED | stage=%s | epoch=%d/%d | loss_estimate=%.6f | %s",
            stage,
            epoch,
            epochs,
            loss_estimate,
            " | ".join(f"{key}={value:.6f}" for key, value in monitor.items()),
        )
    return params


def run_lbfgs(
    loss_fn: Callable[[np.ndarray], float],
    monitor_fn: Callable[[np.ndarray], dict[str, float]],
    params: np.ndarray,
    max_iterations: int,
    stage: str,
    history: History,
    monitor_every: int = 1,
) -> np.ndarray:
    """Minimize a shot-free loss with L-BFGS (E14c, not in the paper).

    Used to find what a circuit can reach when neither SPSA noise nor the
    learning rate limits it. The gradient is a central finite difference of
    the exact loss, because the additive angles of the context-preserving
    circuits enter several gates, where the parameter-shift rule does not
    apply. One history row is one L-BFGS iteration.

    Parameters
    ----------
    loss_fn : callable
        Deterministic loss of the parameters.
    monitor_fn : callable
        Exact monitoring metrics of the parameters.
    params : numpy.ndarray
        Initial parameters.
    max_iterations : int
        Upper bound on L-BFGS iterations (``training.epochs``).
    stage : str
        Stage name written to the history rows.
    history : list
        Receives one row every ``monitor_every`` iterations and at the end.
    monitor_every : int
        Row interval. Default value is 1.

    Returns
    -------
    numpy.ndarray
        Parameters at the last iteration.
    """

    if monitor_every < 1:
        raise ValueError("monitor_every must be at least 1")
    step = 1e-6

    def gradient(point: np.ndarray) -> np.ndarray:
        result = np.zeros_like(point, dtype=float)
        for index in range(len(point)):
            shift = np.zeros_like(point, dtype=float)
            shift[index] = step
            result[index] = (loss_fn(point + shift) - loss_fn(point - shift)) / (
                2.0 * step
            )
        return result

    iteration = 0
    last_recorded = 0

    def record(point: np.ndarray) -> None:
        nonlocal last_recorded
        loss = float(loss_fn(point))
        monitor = monitor_fn(point)
        history.append(
            {
                "stage": stage,
                "epoch": len(history) + 1,
                "stage_epoch": iteration,
                "loss_estimate": loss,
                "loss_estimate_mean": loss,
                **monitor,
            }
        )
        last_recorded = iteration
        logger.info(
            "TRAIN_EPOCH_COMPLETED | stage=%s | epoch=%d/%d | loss_estimate=%.6f | %s",
            stage,
            iteration,
            max_iterations,
            loss,
            " | ".join(f"{key}={value:.6f}" for key, value in monitor.items()),
        )

    def callback(point: np.ndarray) -> None:
        nonlocal iteration
        iteration += 1
        if iteration % monitor_every == 0:
            record(point)

    result = minimize(
        loss_fn,
        params,
        jac=gradient,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": max_iterations, "ftol": 1e-14, "gtol": 1e-9},
    )
    if last_recorded != iteration or not iteration:
        record(result.x)
    logger.info(
        "LBFGS_FINISHED | stage=%s | iterations=%d | loss_evaluations=%d | "
        "converged=%s | message=%s",
        stage,
        iteration,
        result.nfev,
        result.success,
        result.message,
    )
    return result.x


def train_context_loader(
    context_marginal: np.ndarray,
    loader_cfg: dict[str, Any],
    rng: np.random.Generator,
    stage: str,
    history: History,
) -> tuple[np.ndarray, np.ndarray]:
    """Train the HEA context loader with MSE + SPSA (Sec. III, Fig. 4).

    ``loader_cfg["initial_hadamard"]`` true starts the layers from
    ``H^{(x)n} |0...0>`` as drawn in Fig. 4; false starts from ``|0...0>``
    (the implementation used before 2026-11-20).

    The MSE constrains only ``|amplitude|^2``, so the monitor also records the
    fidelity with the phases removed: it separates magnitude errors from the
    phase errors that the QBGU SWAP test sees.

    Returns
    -------
    tuple
        Loaded context state (complex amplitudes) and the loader parameters.
    """

    n_qubits = int(np.log2(len(context_marginal)))
    exact_state = np.sqrt(context_marginal)
    initial_hadamard = loader_cfg["initial_hadamard"]
    if not isinstance(initial_hadamard, bool):
        raise ValueError("model.context_loader.initial_hadamard must be true or false")

    def circuit(params):
        def prepare(state):
            if initial_hadamard:
                for qubit in range(n_qubits):
                    state = apply_1q(state, HADAMARD, qubit, n_qubits)
            return apply_ansatz(state, params, n_qubits)

        return prepare

    def loss_fn(params):
        return loader_mse(circuit(params), context_marginal, _shots(loader_cfg), rng)

    def monitor_fn(params):
        state = circuit(params)(basis_state(0, n_qubits))
        probs = probabilities(state)
        return {
            "mse_exact": float(np.sum((probs - context_marginal) ** 2)),
            "tvd": float(0.5 * np.abs(probs - context_marginal).sum()),
            "fidelity_to_exact_state": float(abs(np.vdot(exact_state, state)) ** 2),
            "fidelity_magnitudes_only": float(np.dot(exact_state, np.abs(state)) ** 2),
        }

    params = initial_params(n_params(n_qubits, int(loader_cfg["layers"])), rng)
    params = run_spsa(
        loss_fn,
        monitor_fn,
        params,
        int(loader_cfg["epochs"]),
        loader_cfg,
        rng,
        stage,
        history,
    )
    return circuit(params)(basis_state(0, n_qubits)), params


def qstl_circuit_factory(
    model_cfg: dict[str, Any], n_qubits: int, levels: int
) -> tuple[Callable[[np.ndarray], Callable[[np.ndarray], np.ndarray]], int]:
    """Return ``params -> circuit`` and the parameter count for ``model.ansatz``.

    ``hea``: the paper's layered RY-RZ-ring-CNOT circuit on all qubits.
    ``context_controlled``: rotations on the prediction qubit only (E11).
    ``context_sequential`` and ``context_entangling``: multi-step
    context-preserving circuits A and B (E14); the prediction register
    holds ``log2(levels)`` binary steps.
    """

    ansatz = model_cfg["ansatz"]
    if ansatz == "hea":
        return (
            lambda params: lambda state: apply_ansatz(state, params, n_qubits),
            n_params(n_qubits, int(model_cfg["layers"])),
        )
    if ansatz == "context_controlled":
        if levels != 2:
            raise ValueError("context_controlled ansatz supports binary levels only")
        context_length = n_qubits - 1
        order = int(model_cfg["interaction_order"])
        return (
            lambda params: (
                lambda state: apply_context_controlled(
                    state, params, context_length, order
                )
            ),
            n_context_controlled_params(context_length, order),
        )
    if ansatz in {"context_sequential", "context_entangling"}:
        horizon = int(np.log2(levels))
        if 2**horizon != levels:
            raise ValueError(f"{ansatz} ansatz needs binary prediction steps")
        context_length = n_qubits - horizon
        if ansatz == "context_sequential":
            return (
                lambda params: (
                    lambda state: apply_context_sequential(
                        state, params, context_length, horizon
                    )
                ),
                n_context_sequential_params(context_length, horizon),
            )
        prediction_layers = int(model_cfg["prediction_layers"])
        return (
            lambda params: (
                lambda state: apply_context_entangling(
                    state, params, context_length, horizon, prediction_layers
                )
            ),
            n_context_entangling_params(context_length, horizon, prediction_layers),
        )
    raise ValueError(f"Unknown model.ansatz '{ansatz}'")


def train_qstl(
    task: ContextualTask,
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    rng: np.random.Generator,
    stage: str,
    history: History,
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """Single-task training (Sec. IV, VI.B) of the configured ansatz.

    ``training.mode = "qbgu"``: every iteration uses the superposed input and
    target (Eq. 20). ``"per_sample"``: every iteration draws one training
    window and uses ``|c>|0>`` against the one-hot ``|c, o>`` (Eq. 18-19);
    with ``training.batch_size = B > 1`` it draws ``B`` windows and averages
    their losses, each window measured with ``shots / B`` shots. Only the
    SWAP-test loss is defined for it. Both modes monitor the two
    objectives, the QBGU fidelity and the expected per-sample fidelity.
    """

    loss_kind = model_cfg["loss"]
    mode = training_cfg["mode"]
    shots = _shots(training_cfg)
    circuit, param_count = qstl_circuit_factory(model_cfg, task.n_qubits, task.levels)

    loss_fn = None
    draw_loss_fn = None
    if mode == "qbgu":

        def loss_fn(params):
            return task.loss(loss_kind, circuit(params), shots, rng)

    elif mode == "per_sample":
        if loss_kind != "swap_test_fidelity":
            raise ValueError("training.mode 'per_sample' needs the SWAP-test loss")

        batch_size = int(training_cfg["batch_size"])
        if batch_size < 1:
            raise ValueError("training.batch_size must be at least 1")
        if batch_size == 1:
            # Kept as its own path so S7 runs (batch size 1) stay reproducible.
            def draw_loss_fn():
                context, outcome = task.sample_window(rng)
                return lambda params: task.per_sample_swap_loss(
                    circuit(params), context, outcome, shots, rng
                )

        else:
            # The shot budget per loss evaluation is split evenly over the
            # batch, so every batch size uses the same total shots (E12b).
            if shots is not None and shots % batch_size:
                raise ValueError(
                    f"training.shots ({shots}) must be a multiple of "
                    f"training.batch_size ({batch_size})"
                )
            shots_per_window = None if shots is None else shots // batch_size
            flat_joint = task.joint.reshape(-1)

            def draw_loss_fn():
                counts = rng.multinomial(batch_size, flat_joint).reshape(
                    task.joint.shape
                )
                return lambda params: task.minibatch_swap_loss(
                    circuit(params), counts, shots_per_window, rng
                )

    else:
        raise ValueError(f"Unknown training.mode '{mode}'")

    def monitor_fn(params):
        fidelity = task.fidelity(circuit(params))
        return {
            "fidelity": fidelity,
            "swap_loss_exact": 0.5 * (1.0 - fidelity),
            "per_sample_fidelity": task.expected_per_sample_fidelity(circuit(params)),
        }

    params = initial_params(param_count, rng, float(training_cfg["init_max_angle"]))
    optimizer = training_cfg["optimizer"]
    if optimizer == "spsa":
        params = run_spsa(
            loss_fn,
            monitor_fn,
            params,
            int(training_cfg["epochs"]),
            training_cfg,
            rng,
            stage,
            history,
            monitor_every=int(training_cfg["monitor_every"]),
            draw_loss_fn=draw_loss_fn,
        )
    elif optimizer == "lbfgs":
        # L-BFGS needs a deterministic loss: the superposed objective without
        # shot sampling.
        if mode != "qbgu" or shots is not None:
            raise ValueError(
                "training.optimizer 'lbfgs' needs training.mode 'qbgu' and "
                "training.shots null"
            )
        params = run_lbfgs(
            loss_fn,
            monitor_fn,
            params,
            int(training_cfg["epochs"]),
            stage,
            history,
            monitor_every=int(training_cfg["monitor_every"]),
        )
    else:
        raise ValueError(f"Unknown training.optimizer '{optimizer}'")
    return circuit(params), params


def train_qmtl(
    tasks: list[ContextualTask],
    names: list[str],
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    rng: np.random.Generator,
    history: History,
) -> tuple[list[Callable[[np.ndarray], np.ndarray]], dict[str, np.ndarray]]:
    """Sequential share-and-specify training (Sec. V.B, VI.C).

    The label is fixed while asset ``k`` trains, so only the shared block and
    asset ``k``'s block move (Eq. 30). Assets are visited in order for
    ``cycles`` rounds of ``epochs_per_task`` epochs (paper: 1 round, 250).
    """

    if training_cfg["mode"] != "qbgu":
        raise ValueError("QMTL supports training.mode 'qbgu' only")
    n_qubits = tasks[0].n_qubits
    layers = int(model_cfg["layers"])
    asset_block = model_cfg["qmtl_asset_block"]
    block_size = n_params(n_qubits, layers)
    shots = _shots(training_cfg)
    shared = initial_params(block_size, rng)
    specific = [initial_params(block_size, rng) for _ in tasks]

    def circuit(shared_params, specific_params):
        return lambda state: apply_share_and_specify(
            state, shared_params, specific_params, n_qubits, asset_block
        )

    def monitor_fn_for(current):
        def monitor_fn(joint_params):
            fidelities = {}
            for index, (task, name) in enumerate(zip(tasks, names)):
                specific_params = (
                    joint_params[block_size:] if index == current else specific[index]
                )
                fidelities[f"fidelity_{name}"] = task.fidelity(
                    circuit(joint_params[:block_size], specific_params)
                )
            return fidelities

        return monitor_fn

    for cycle in range(1, int(model_cfg["cycles"]) + 1):
        for index, (task, name) in enumerate(zip(tasks, names)):

            def loss_fn(joint_params, task=task):
                return task.loss(
                    model_cfg["loss"],
                    circuit(joint_params[:block_size], joint_params[block_size:]),
                    shots,
                    rng,
                )

            joint = run_spsa(
                loss_fn,
                monitor_fn_for(index),
                np.concatenate([shared, specific[index]]),
                int(training_cfg["epochs_per_task"]),
                training_cfg,
                rng,
                f"qmtl_cycle{cycle}_{name}",
                history,
            )
            shared, specific[index] = joint[:block_size], joint[block_size:]

    circuits = [circuit(shared, specific_params) for specific_params in specific]
    params = {"shared": shared, **{f"specific_{n}": p for n, p in zip(names, specific)}}
    return circuits, params
