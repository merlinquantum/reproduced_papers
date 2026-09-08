"""Batched solver: the same algorithm as :mod:`lib.bbs`, many instances at once.

One update step needs samples from the base circuit and from ``2K`` shifted
circuits, for each instance. Run per-instance on CPU that is thousands of
separate sampler calls; here the ``I * (1 + 2K)`` distinct unitaries go into one
call, which is what the GPU Clifford & Clifford sampler wants. At m=30 that is
155 circuits per instance.

Nothing about the algorithm changes: same input state, same threshold detection,
same trainable bit-flip layer, same parameter-shift estimator, same plain SGD,
and the same Appendix B accounting -- every string generated while estimating a
gradient counts as a candidate, and the best is tracked over all of them.
``evaluations`` is asserted against the bound in tests/test_bbs_gpu.py, exactly
as on the CPU path.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from lib.sampler_torch import draw_clicks
from lib.tbi import alternating_input, beamsplitter_layout, candidate_budget, shifted_angles, unitary


def solve_batch(
    cost_batch,
    m,
    seeds,
    delays=(1, 3, 9),
    updates=200,
    samples=50,
    lr_theta=0.01,
    lr_alpha=0.05,
    shift=math.pi / 2,
    shift_scale=1.0,
    source="boson",
    rate_scale=1.0,
    topology="chain",
    device=None,
    sampler_backend="triton",
    sampler_algo="auto",
    freeze_theta=False,
):
    """Solve ``len(seeds)`` instances in lockstep.

    Parameters
    ----------
    cost_batch : callable
        Maps ``(I, Q, m)`` bit tensors to ``(I, Q)`` costs.
    m : int
        Number of binary variables.
    seeds : sequence of int
        One instance seed per lockstep slot; also seeds the angle initialisation,
        so a slot reproduces the CPU path's instance exactly.
    freeze_theta : bool
        Ablation: hold the interferometer fixed and train only the bit-flip
        layer. Written as an explicit flag rather than ``lr_theta = 0`` because a
        falsy learning-rate check is an easy way to run the ablation by accident
        and not notice. Default value is False.

    Returns
    -------
    dict
        ``best_cost`` per instance, ``evaluations`` per instance, ``budget``,
        ``mean_clicks`` per instance, and the mean ``history`` of E[C(X)].
    """
    device = device or torch.device("cpu")
    n_angles = len(beamsplitter_layout(m, delays, topology))
    input_state = alternating_input(m, delays, topology)
    input_modes = [i for i, n in enumerate(input_state) if n]
    instances = len(seeds)
    circuits = 1 + 2 * n_angles

    theta = torch.tensor(
        np.stack([np.random.default_rng(seed).random(n_angles) * 2 * np.pi for seed in seeds]),
        dtype=torch.float64, device=device,
    )
    alpha = torch.zeros(instances, m, dtype=torch.float64, device=device)
    generator = torch.Generator(device=device).manual_seed(int(seeds[0]) + 1000)

    best = torch.full((instances,), float("inf"), dtype=torch.float64, device=device)
    evaluations = 0
    click_sum = torch.zeros(instances, dtype=torch.float64, device=device)
    history = []

    def apply_flips(clicks, probability):
        """XOR ``clicks`` (I, Q, m) with independent Bernoulli(probability)."""
        draw = torch.rand(clicks.shape, generator=generator, device=device, dtype=torch.float64)
        return torch.bitwise_xor(clicks, (draw < probability[:, None, :]).to(clicks.dtype))

    for _ in range(updates):
        batched_theta = shifted_angles(theta, shift).reshape(instances * circuits, n_angles)
        unitaries = unitary(batched_theta, m, delays, topology)
        clicks = draw_clicks(
            unitaries, input_modes, samples, generator, source, m,
            rate_scale=rate_scale, sampler_backend=sampler_backend, sampler_algo=sampler_algo,
        ).reshape(instances, circuits, samples, m)

        click_sum += clicks[:, 0].to(torch.float64).sum(-1).mean(-1)
        probability = torch.sigmoid(alpha)

        flat = clicks.reshape(instances, circuits * samples, m)
        draw = torch.rand(flat.shape, generator=generator, device=device, dtype=torch.float64)
        flips = (draw < probability[:, None, :]).to(flat.dtype)
        candidates = torch.bitwise_xor(flat, flips)
        costs = cost_batch(candidates).reshape(instances, circuits, samples)
        evaluations += instances * circuits * samples
        best = torch.minimum(best, costs.reshape(instances, -1).min(dim=1).values)
        history.append(float(costs[:, 0].mean()))

        plus, minus = costs[:, 1::2].mean(-1), costs[:, 2::2].mean(-1)          # (I, K)
        theta_gradient = shift_scale * (plus - minus) / math.sin(shift)

        # Bit-flip gradient, paired with the base circuit's own draws: forcing
        # bit i to flipped and to unflipped while every other draw is held fixed
        # is the low-variance estimator of E[C | B_i = 1] - E[C | B_i = 0], and it
        # is what lib/bbs.py does, so the two paths stay comparable.
        base_clicks = clicks[:, 0]
        base_flips = flips[:, :samples]
        alpha_gradient = torch.empty_like(alpha)
        for index in range(m):
            forced_costs = []
            for value in (1, 0):
                forced = base_flips.clone()
                forced[:, :, index] = value
                forced_candidates = torch.bitwise_xor(base_clicks, forced)
                scored = cost_batch(forced_candidates)
                evaluations += instances * samples
                best = torch.minimum(best, scored.min(dim=1).values)
                forced_costs.append(scored.mean(-1))
            alpha_gradient[:, index] = (
                (forced_costs[0] - forced_costs[1]) * probability[:, index] * (1 - probability[:, index])
            )

        if not freeze_theta:
            theta = theta - lr_theta * theta_gradient
        alpha = alpha - lr_alpha * alpha_gradient

    return {
        "best_cost": best.cpu().numpy(),
        "evaluations": evaluations // instances,
        "budget": candidate_budget(m, delays, updates, samples),
        "mean_clicks": (click_sum / updates).cpu().numpy(),
        "history": history,
    }
