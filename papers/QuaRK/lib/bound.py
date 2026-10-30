"""Terms of the generalization bound of Theorem 2 (paper Eq. 17), for a fitted readout.

Inputs we must choose, because the paper gives no numbers:

* ``Lambda`` is the RKHS norm of the fitted readout, ``sqrt(alpha^T K alpha)``, so the
  bound is evaluated for the smallest ball that contains the fitted function;
* ``Upsilon_Y`` is the largest ``|y|`` over all windows of the dataset;
* the beta-mixing coefficient of the VARMA at gap g is not quantified in the paper;
  we set ``delta' = delta``, which drops the dependence correction and can only make
  the bound smaller (more favourable to the paper).
"""

from __future__ import annotations

import math


def generalization_bound_terms(
    *,
    rkhs_norm: float,
    label_bound: float,
    num_train: int,
    delta: float,
    nu: float,
    xi: float,
    num_features: int,
    max_contraction: float,
    window: int,
) -> dict:
    """Return the three terms of Eq. (17) and their sum.

    The window-truncation term needs ``nu > 1`` (Matérn twice differentiable at 0);
    for ``nu <= 1`` it and the total are ``None``.
    """
    budget = rkhs_norm + label_bound
    rademacher = 4.0 * math.sqrt(2.0) * rkhs_norm * budget / math.sqrt(num_train)
    mixing = 3.0 * budget**2 * math.sqrt(math.log(4.0 / delta) / num_train)
    if nu > 1.0:
        truncation = (
            4.0
            * rkhs_norm
            * budget
            / xi
            * math.sqrt(nu * num_features / (nu - 1.0))
            * max_contraction**window
        )
        total = rademacher + mixing + truncation
    else:
        truncation = None
        total = None
    return {
        "rademacher": rademacher,
        "mixing": mixing,
        "truncation": truncation,
        "total": total,
        "rkhs_norm": rkhs_norm,
        "label_bound": label_bound,
        "delta": delta,
        "nu_valid": nu > 1.0,
    }
