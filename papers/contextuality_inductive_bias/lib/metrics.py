"""Training objective and evaluation metrics of Sec. 8.4.

All functions take model expectation values ``e`` of shape ``(N, 3)``, read as
E[Y_k | x], so that the model's win probability is P_k(+1 | x) = (1 + e_k) / 2.
"""

from __future__ import annotations

import torch

# The author code floors model probabilities at 1e-8 inside the KL divergence
# (never in the training loss).
DEFAULT_KL_FLOOR = 1e-8


def win_probabilities(expectations: torch.Tensor) -> torch.Tensor:
    """P_k(+1 | x) = (1 + E[Y_k | x]) / 2."""
    return (1.0 + expectations) / 2.0


def negative_log_likelihood(
    expectations: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Mean negative log-likelihood of the three marginals (Eq. (101)).

    The paper writes a sum over data; the author code (and the NLL axis of
    Fig. 8, about 0.68 per label) uses the mean over the 3N labels, which is
    what is returned here (discrepancy D9).

    Parameters
    ----------
    expectations : torch.Tensor
        Model E[Y_k | x], shape ``(N, 3)``.
    labels : torch.Tensor
        0 for a positive payoff, 1 for a negative one, shape ``(N, 3)``.

    Returns
    -------
    torch.Tensor
        Scalar loss. It is ``inf`` if the model gives probability 0 to an
        observed label (possible for the HardTanh surrogate); callers must
        treat a non-finite loss as a failed run.
    """
    signs = 1.0 - 2.0 * labels
    likelihood = (1.0 + signs * expectations) / 2.0
    return -torch.log(likelihood).mean()


def per_sample_kl(
    true_win: torch.Tensor,
    expectations: torch.Tensor,
    floor: float = DEFAULT_KL_FLOOR,
) -> torch.Tensor:
    """Mean over the three players of KL(D_k(. | x) || P_k(. | x)), per sample.

    Parameters
    ----------
    true_win : torch.Tensor
        Exact P(Y_k = +1 | x), shape ``(N, 3)``.
    expectations : torch.Tensor
        Model E[Y_k | x], shape ``(N, 3)``.
    floor : float
        Lower cut-off applied to the model probabilities. Default value is 1e-8.

    Returns
    -------
    torch.Tensor
        Shape ``(N,)``.
    """
    model_win = torch.clamp(win_probabilities(expectations), min=floor)
    model_lose = torch.clamp(1.0 - win_probabilities(expectations), min=floor)
    true_lose = 1.0 - true_win
    kl = (
        torch.xlogy(true_win, true_win)
        - true_win * torch.log(model_win)
        + torch.xlogy(true_lose, true_lose)
        - true_lose * torch.log(model_lose)
    )
    return kl.mean(dim=1)


def average_kl(
    true_win: torch.Tensor, expectations: torch.Tensor, floor: float = DEFAULT_KL_FLOOR
) -> float:
    """Average test KL divergence of Eq. (102)-(103)."""
    return float(per_sample_kl(true_win, expectations, floor).mean())


def zero_sum_violation(expectations: torch.Tensor) -> float:
    """max_x |sum_k E[Y_k | x]|; zero for a model encoding the bias exactly."""
    return float(expectations.sum(dim=1).abs().max())
