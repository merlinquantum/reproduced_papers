"""Finite-shot distribution losses and the reference SPSA optimizer."""

import numpy as np


class SPSA:
    """Optimize phases using two finite-shot objective calls per update.

    Parameters
    ----------
    parameters : numpy.ndarray
        Initial phase vector.
    objective : callable
        Finite-shot scalar objective accepting a phase vector.
    settings : dict
        alpha, gamma, c, stability_fraction, initial_step and gain_calibration.
    total_steps : int
        Update count defining the stability constant A.
    seed : int
        Independent Bernoulli perturbation RNG seed.
    """

    def __init__(self, parameters, objective, settings, total_steps, seed):
        self.parameters = np.array(parameters, dtype=float, copy=True)
        self.objective = objective
        self.settings = dict(settings)
        self.rng = np.random.RandomState(seed)
        self.step_number = 0
        self.stability = settings["stability_fraction"] * total_steps
        initial_gradient = self._gradient(settings["c"])
        if settings["gain_calibration"] != "author_absolute_mean":
            raise ValueError(
                "Only the pinned author's absolute-mean gain is implemented"
            )
        magnitude = float(abs(initial_gradient.mean()))
        if not np.isfinite(magnitude) or magnitude <= 1e-12:
            raise ValueError(
                "Degenerate author SPSA gain calibration; no epsilon or retry applied"
            )
        self.calibration_magnitude = magnitude
        self.gain = (
            settings["initial_step"]
            * (self.stability + 1) ** settings["alpha"]
            / magnitude
        )

    def _gradient(self, radius):
        delta = self.rng.binomial(1, 0.5, len(self.parameters)) * 2 - 1
        positive = self.objective(self.parameters + radius * delta)
        negative = self.objective(self.parameters - radius * delta)
        gradient = (positive - negative) / (2 * radius * delta)
        if not np.isfinite(gradient).all():
            raise RuntimeError("SPSA produced a nonfinite gradient")
        return gradient

    def step(self) -> dict:
        """Apply one update and return its size and schedule diagnostics.

        Returns
        -------
        dict
            Step number, perturbation radius, gain and Euclidean update norm.
        """
        self.step_number += 1
        k = self.step_number
        rate = self.gain / (k + self.stability) ** self.settings["alpha"]
        radius = self.settings["c"] / k ** self.settings["gamma"]
        gradient = self._gradient(radius)
        update = rate * gradient
        self.parameters -= update
        if not np.isfinite(self.parameters).all():
            raise RuntimeError("SPSA parameters became nonfinite")
        return {
            "step": k,
            "gain": float(rate),
            "radius": float(radius),
            "update_norm": float(np.linalg.norm(update)),
        }

    def checkpoint(self) -> dict:
        """Return numeric state sufficient to restore the optimizer exactly.

        Returns
        -------
        dict
            NPZ-compatible parameters, schedule and perturbation RNG state.
        """
        state = self.rng.get_state()
        return {
            "parameters": self.parameters.copy(),
            "step": self.step_number,
            "gain": self.gain,
            "stability": self.stability,
            "calibration_magnitude": self.calibration_magnitude,
            "rng_keys": state[1].copy(),
            "rng_position": state[2],
            "rng_has_gaussian": state[3],
            "rng_cached_gaussian": state[4],
        }

    @classmethod
    def restore(cls, checkpoint, objective, settings):
        """Restore a checkpoint without evaluating or recalibrating the objective.

        Parameters
        ----------
        checkpoint : mapping
            Numeric state returned by :meth:`checkpoint`.
        objective : callable
            Restored objective with its own sampling ledger.
        settings : dict
            Unchanged SPSA schedule settings.

        Returns
        -------
        SPSA
            Optimizer positioned immediately after the saved update.

        Raises
        ------
        ValueError
            If saved parameters, schedule scalars or RNG state are invalid.
        """
        optimizer = cls.__new__(cls)
        optimizer.parameters = np.array(
            checkpoint["parameters"], dtype=float, copy=True
        )
        optimizer.objective = objective
        optimizer.settings = dict(settings)
        optimizer.step_number = int(checkpoint["step"])
        optimizer.stability = float(checkpoint["stability"])
        optimizer.gain = float(checkpoint["gain"])
        optimizer.calibration_magnitude = float(checkpoint["calibration_magnitude"])
        if (
            optimizer.parameters.ndim != 1
            or not np.isfinite(optimizer.parameters).all()
            or optimizer.step_number < 0
            or not np.isfinite(
                [optimizer.stability, optimizer.gain, optimizer.calibration_magnitude]
            ).all()
            or optimizer.stability < 0
            or optimizer.gain <= 0
            or optimizer.calibration_magnitude <= 1e-12
        ):
            raise ValueError("Checkpoint SPSA state is invalid")
        optimizer.rng = np.random.RandomState()
        optimizer.rng.set_state(
            (
                "MT19937",
                np.array(checkpoint["rng_keys"], dtype=np.uint32),
                int(checkpoint["rng_position"]),
                int(checkpoint["rng_has_gaussian"]),
                float(checkpoint["rng_cached_gaussian"]),
            )
        )
        return optimizer


def distribution_loss(model: np.ndarray, target: np.ndarray, name: str) -> float:
    """Compute TVD or base-two KL(model || target) without pseudocounts.

    Parameters
    ----------
    model : numpy.ndarray
        Normalized model probability vector.
    target : numpy.ndarray
        Normalized target probability vector in the same fixed bin order.
    name : str
        tvd or kl.

    Returns
    -------
    float
        Divergence; KL is infinite if positive model mass has zero target mass.
    """
    model, target = np.asarray(model), np.asarray(target)
    if model.shape != target.shape or model.ndim != 1:
        raise ValueError("Model and target must be equal-length probability vectors")
    for values in (model, target):
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or abs(values.sum() - 1.0) > 1e-10
        ):
            raise ValueError(
                "Loss requires finite normalized nonnegative probabilities"
            )
    if name == "tvd":
        return float(0.5 * np.abs(model - target).sum())
    if name != "kl":
        raise ValueError("Loss must be tvd or kl")
    positive = model > 0
    if (target[positive] == 0).any():
        return float("inf")
    return float(np.sum(model[positive] * np.log2(model[positive] / target[positive])))
