"""Full-batch training of many independent seeds at once (Fig. 3b, Table 1, Fig. 6).

All seeds of one model are trained together: ``theta`` has shape ``(n_seeds, d)``
and the objective is the sum of per-seed losses, so every seed receives exactly its
own gradient. Adam is elementwise, so this is equivalent to independent runs.

Loss: mean cross-entropy ``-log p(y | x; theta)`` over the training set. For the
qubit models this is the authors' ``CrossEntropy`` on parity probabilities; for
classical networks with a ``softmax`` head it equals ``nn.CrossEntropyLoss`` on
logits, as in the authors' ``classical_loss.py``.

Recorded curve: ``losses[t]`` is the loss at ``theta_t`` for ``t = 0..T``. The
authors' torch loop stored ``L(theta_0..theta_{T-1})`` and their Qiskit ADAM stored
``L(theta_1..theta_T)``; both are slices of this curve.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch

logger = logging.getLogger(__name__)
OPTIMIZERS = ("torch_adam", "qiskit_adam")
GRADIENTS = ("autograd", "finite_difference")


def initial_parameters(
    n_seeds: int, d: int, seeds: list[int], low: float, high: float
) -> np.ndarray:
    """Draw one ``U[low, high]^d`` initial point per seed from ``RandomState(seed)``."""
    if len(seeds) != n_seeds:
        raise ValueError("One seed per trained model is required")
    return np.stack(
        [np.random.RandomState(seed).uniform(low, high, size=d) for seed in seeds]
    )


def _per_seed_labels(y: torch.Tensor, n_seeds: int) -> torch.Tensor:
    """Broadcast shared labels ``(N,)`` or check per-seed labels ``(n_seeds, N)``."""
    if y.dim() == 1:
        return y.reshape(1, -1).expand(n_seeds, -1)
    if y.shape[0] != n_seeds:
        raise ValueError(f"Per-seed labels need {n_seeds} rows; got {tuple(y.shape)}")
    return y


def _probabilities(model, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    n_seeds, n_samples = theta.shape[0], x.shape[0]
    return model.probabilities(
        x.repeat(n_seeds, 1), theta.repeat_interleave(n_samples, dim=0)
    ).reshape(n_seeds, n_samples, -1)


def per_seed_loss(
    model, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """Return the mean cross-entropy of every seed, shape ``(n_seeds,)``.

    ``y`` is ``(N,)`` (shared labels) or ``(n_seeds, N)`` (one label set per seed).
    """
    labels = _per_seed_labels(y, theta.shape[0])
    picked = _probabilities(model, x, theta).gather(2, labels.unsqueeze(2)).squeeze(2)
    return -torch.log(picked).mean(dim=1)


def per_seed_accuracy(
    model, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """Return the accuracy of every seed, shape ``(n_seeds,)``."""
    labels = _per_seed_labels(y, theta.shape[0])
    with torch.no_grad():
        predictions = _probabilities(model, x, theta).argmax(dim=2)
    return (predictions == labels).double().mean(dim=1)


def _gradient(
    model, x, y, theta: torch.Tensor, method: str, step: float
) -> torch.Tensor:
    if method == "autograd":
        theta = theta.detach().requires_grad_(True)
        (gradient,) = torch.autograd.grad(
            per_seed_loss(model, x, y, theta).sum(), theta
        )
        return gradient
    if method == "finite_difference":
        # Forward differences with the authors' step (Qiskit Optimizer.gradient_num_diff, eps=1e-10).
        with torch.no_grad():
            base = per_seed_loss(model, x, y, theta)
            gradient = torch.empty_like(theta)
            for index in range(theta.shape[1]):
                shifted = theta.clone()
                shifted[:, index] += step
                gradient[:, index] = (per_seed_loss(model, x, y, shifted) - base) / step
        return gradient
    raise ValueError(
        f"Unsupported gradient method {method!r}; expected one of {GRADIENTS}"
    )


def train(
    model, x: torch.Tensor, y: torch.Tensor, theta0: torch.Tensor, training_cfg: dict
) -> dict:
    """Train every seed for a fixed number of iterations.

    Parameters
    ----------
    model : object
        Exposes ``probabilities(x, theta)`` and ``d``.
    x, y : torch.Tensor
        Full training batch ``(N, s_in)`` and labels ``(N,)``.
    theta0 : torch.Tensor
        Initial parameters ``(n_seeds, d)``.
    training_cfg : dict
        Keys ``iterations``, ``lr``, ``optimizer`` (``torch_adam``: torch defaults,
        betas 0.9/0.999, eps 1e-8; ``qiskit_adam``: the authors' ADAM, betas
        0.9/0.99, noise 1e-8 added to sqrt(v), bias-corrected step size),
        ``gradient`` (``autograd`` or ``finite_difference``) and
        ``finite_difference_step``, ``log_every`` (INFO progress interval; other
        iterations are logged at DEBUG).

    Returns
    -------
    dict
        ``losses`` ``(T + 1, n_seeds)``, ``accuracies`` ``(T + 1, n_seeds)``, and
        ``theta`` ``(n_seeds, d)`` after training, as NumPy arrays.
    """
    iterations = int(training_cfg["iterations"])
    lr = float(training_cfg["lr"])
    optimizer_name = training_cfg["optimizer"]
    gradient_method = training_cfg["gradient"]
    step = float(training_cfg["finite_difference_step"])
    log_every = int(training_cfg["log_every"])
    if optimizer_name not in OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer {optimizer_name!r}; expected one of {OPTIMIZERS}"
        )

    theta = theta0.detach().clone()
    first_moment = torch.zeros_like(theta)
    second_moment = torch.zeros_like(theta)
    beta1, beta2 = (0.9, 0.999) if optimizer_name == "torch_adam" else (0.9, 0.99)

    losses, accuracies = [], []
    with torch.no_grad():
        losses.append(per_seed_loss(model, x, y, theta))
    accuracies.append(per_seed_accuracy(model, x, y, theta))
    logger.info(
        "MODEL_TRAINING_STARTED | iterations=%d | optimizer=%s | gradient=%s | lr=%g | seeds=%d | initial_loss_mean=%.6f",
        iterations,
        optimizer_name,
        gradient_method,
        lr,
        theta.shape[0],
        losses[0].mean().item(),
    )
    for iteration in range(1, iterations + 1):
        gradient = _gradient(model, x, y, theta, gradient_method, step)
        first_moment = beta1 * first_moment + (1 - beta1) * gradient
        second_moment = beta2 * second_moment + (1 - beta2) * gradient * gradient
        if optimizer_name == "torch_adam":
            corrected_first = first_moment / (1 - beta1**iteration)
            corrected_second = second_moment / (1 - beta2**iteration)
            theta = theta - lr * corrected_first / (corrected_second.sqrt() + 1e-8)
        else:
            effective_lr = lr * math.sqrt(1 - beta2**iteration) / (1 - beta1**iteration)
            theta = theta - effective_lr * first_moment / (second_moment.sqrt() + 1e-8)
        with torch.no_grad():
            loss = per_seed_loss(model, x, y, theta)
        losses.append(loss)
        accuracies.append(per_seed_accuracy(model, x, y, theta))
        level = (
            logging.INFO
            if iteration % log_every == 0 or iteration == iterations
            else logging.DEBUG
        )
        logger.log(
            level,
            "TRAIN_ITERATION_COMPLETED | iteration=%d/%d | loss_mean=%.6f | loss_std=%.6f | accuracy_mean=%.4f",
            iteration,
            iterations,
            loss.mean().item(),
            loss.std().item() if loss.numel() > 1 else 0.0,
            accuracies[-1].mean().item(),
        )
    return {
        "losses": torch.stack(losses).cpu().numpy(),
        "accuracies": torch.stack(accuracies).cpu().numpy(),
        "theta": theta.cpu().numpy(),
    }
