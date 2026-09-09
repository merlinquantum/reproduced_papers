"""Time-bin interferometer, unrolled into an ordinary multimode unitary.

A delay line of length l holds a pulse for l time bins, so on the bin train it
applies the same beamsplitter to every pair (t, t+l) in ascending t. Unrolling
time into modes turns K loops of lengths {l_1, ..., l_K} into a static
interferometer with

    sum_i (m - l_i)

beamsplitters -- the count that Appendix B of the paper uses in its bound. No
time-loop primitive is needed anywhere in this reproduction.

Two circuit models
------------------
``topology="chain"`` puts a beamsplitter on each pair of bins (t, t+l), giving
``m - l`` beamsplitters per loop. This is the model the paper's own arithmetic
describes: Appendix B counts ``sum_i (m - l_i)`` trainable beamsplitters, which
is 77 at m=30 for delays (1, 3, 9), and 77 angles is what its stated bound of
2.15e6 cost evaluations requires.

``topology="loop"`` models the fibre loop as a rail with memory: one extra mode
per loop, a beamsplitter between the rail and every bin in turn, so photons may
circulate for several round trips. This is what ORCA's released simulator
actually implements (``quantumqubo/tbi/tbi_sampler.py``), and
tests/test_tbi.py checks the two agree to machine precision for a single loop.
It needs ``m`` beamsplitters per loop rather than ``m - l``, so it cannot be the
model behind the paper's count.

The two are different circuits, not two descriptions of one circuit. The paper
is explicit and internally consistent about its count, so ``chain`` is the
default here and the reproduction runs on it; ``loop`` is kept as the validated
reference. See README.md, "Disagreements with the released reference code".

Beamsplitter convention
-----------------------
Reflectivity R = cos^2(theta / 2), balanced at theta = pi/2, matching Perceval
and MerLin. ORCA's released toolkit (github.com/orcacomputing/quantumqubo) uses
R = cos^2(theta) instead, balanced at pi/4. Both are self-consistent; the
consequence for the parameter-shift rule is worked out in ``lib/bbs.py`` and in
README.md. Getting this wrong silently rescales every gradient, so the convention
is asserted in tests/test_tbi.py.
"""

from __future__ import annotations

import numpy as np
import torch


def beamsplitter_layout(m, delays, topology="chain"):
    """List the ``(mode_a, mode_b)`` beamsplitter pairs, in application order.

    Parameters
    ----------
    m : int
        Number of time bins, which become modes after unrolling.
    delays : sequence of int
        Delay-line lengths, applied in the given order (the paper uses 1, 3, 9).
    topology : str
        ``"chain"`` (the paper's count) or ``"loop"`` (ORCA's simulator). Default
        value is ``"chain"``.

    Returns
    -------
    list of tuple of int
        One pair per trainable beamsplitter.
    """
    if topology == "chain":
        pairs = []
        for delay in delays:
            if delay >= m:
                continue
            pairs.extend((t, t + delay) for t in range(m - delay))
        return pairs
    if topology == "loop":
        pairs = []
        for index, _delay in enumerate(delays):
            rail = m + index
            pairs.extend((rail, t) for t in range(m))
        return pairs
    raise ValueError(f"unknown topology {topology!r}")


def mode_count(m, delays, topology="chain"):
    """Total modes the circuit needs; the ``loop`` model adds one rail per loop."""
    return m if topology == "chain" else m + len(delays)


def unitary(theta, m, delays, topology="chain"):
    """Interferometer unitary for one or many angle vectors.

    Parameters
    ----------
    theta : torch.Tensor
        Shape ``(K,)`` or ``(B, K)`` where K is the number of beamsplitters.
    m : int
        Number of time bins.
    delays : sequence of int
        Delay-line lengths.
    topology : str
        See :func:`beamsplitter_layout`. Default value is ``"chain"``.

    Returns
    -------
    torch.Tensor
        Shape ``(n, n)`` or ``(B, n, n)`` with ``n = mode_count(...)``, real
        orthogonal, differentiable in ``theta``.
    """
    pairs = beamsplitter_layout(m, delays, topology)
    n = mode_count(m, delays, topology)
    if theta.shape[-1] != len(pairs):
        raise ValueError(
            f"expected {len(pairs)} angles for m={m}, delays={tuple(delays)}, topology={topology}; "
            f"got {theta.shape[-1]}"
        )
    batched = theta.dim() == 2
    angles = theta if batched else theta.unsqueeze(0)
    batch = angles.shape[0]

    current = (
        torch.eye(n, dtype=angles.dtype, device=angles.device)
        .expand(batch, n, n)
        .clone()
    )
    half = angles / 2
    cos, sin = torch.cos(half), torch.sin(half)
    for k, (a, b) in enumerate(pairs):
        row_a = current[:, a, :].clone()
        row_b = current[:, b, :].clone()
        if topology == "chain":
            current[:, a, :] = cos[:, k, None] * row_a - sin[:, k, None] * row_b
            current[:, b, :] = sin[:, k, None] * row_a + cos[:, k, None] * row_b
        else:
            # rail-bin block chosen so that a photon in the bin enters the rail
            # with amplitude cos(theta/2): at theta = 0 the pulse is stored
            # completely, which is how ORCA describes a "fully reflective"
            # beamsplitter. Validated against their sampler in tests/test_tbi.py.
            current[:, a, :] = sin[:, k, None] * row_a - cos[:, k, None] * row_b
            current[:, b, :] = cos[:, k, None] * row_a + sin[:, k, None] * row_b
    return current if batched else current[0]


def alternating_input(m, delays=(), topology="chain"):
    """The paper's input state |1,0,1,0,...>: m/2 photons in alternating bins.

    The ``loop`` topology appends one empty rail mode per delay line.
    """
    bins = [1 if i % 2 == 0 else 0 for i in range(m)]
    if topology == "loop":
        bins += [0] * len(delays)
    return tuple(bins)


def candidate_budget(m, delays, updates, samples):
    """Appendix B's bound on cost-function evaluations: N S (2 sum(m-l_i) + 2m + 1).

    Every string generated while estimating gradients counts, not only the
    forward-pass samples. The terms are the 2 sum(m-l_i) shifted circuit
    evaluations for the parameter-shift rule, the 2m evaluations for the
    bit-flip probabilities, and 1 for the unshifted expectation. ORCA's released
    solver tracks its best candidate the same way: ``QuboOneConfiguration.readout``
    updates ``E_min_encountered`` on every call, including the shifted ones.
    """
    n_bs = sum(m - delay for delay in delays if delay < m)
    return updates * samples * (2 * n_bs + 2 * m + 1)


def perceval_unitary(theta, m, delays, topology="chain"):
    """Same unitary as :func:`unitary`, as a plain numpy array for Perceval."""
    with torch.no_grad():
        return unitary(
            torch.as_tensor(theta, dtype=torch.float64), m, delays, topology
        ).numpy()


def orca_single_loop_distribution(input_state, thetas):
    """Exact output distribution of ORCA's sequential single-loop model.

    Ported from ``quantumqubo/tbi/tbi_sampler.py`` (ORCA Computing) with the
    convention changed from R = cos^2(theta) to cos^2(theta/2) to match the rest
    of this package, and with the sampling replaced by exact enumeration so the
    test is deterministic. Their sampler is exact because the output mode of each
    step is never touched again and the loop is left in a Fock state, so
    measuring as you go cannot destroy any later interference.

    Parameters
    ----------
    input_state : sequence of int
        Photons per input bin; at most one per bin, as in ORCA's code.
    thetas : sequence of float
        One angle per bin.

    Returns
    -------
    dict
        Maps an output tuple of length ``len(input_state) + 1`` (the bins, then
        the photons left in the loop) to its probability.
    """
    from math import factorial, sqrt

    def step_probabilities(n_loop, incoming, theta):
        if incoming > 1:
            raise ValueError(
                "ORCA's single-loop sampler allows at most one photon per input bin"
            )
        half = theta / 2
        amplitudes = {}
        for k in range(n_loop + 1):
            for p in range(incoming + 1):
                prefactor = (
                    sqrt(factorial(k + p))
                    * sqrt(factorial(n_loop + incoming - p - k))
                    / (sqrt(factorial(n_loop)) * sqrt(factorial(incoming)))
                    * factorial(n_loop)
                    * factorial(incoming)
                    / (
                        factorial(k)
                        * factorial(n_loop - k)
                        * factorial(p)
                        * factorial(incoming - p)
                    )
                )
                amplitudes[k + p] = amplitudes.get(k + p, 0.0) + (
                    prefactor
                    * ((-1) ** p)
                    * (np.cos(half) ** (incoming - p + k))
                    * (np.sin(half) ** (n_loop - k + p))
                )
        outputs = sorted(amplitudes)
        probabilities = np.array([amplitudes[o] ** 2 for o in outputs])
        return outputs, probabilities / probabilities.sum()

    states = {((), 0): 1.0}
    for index, theta in enumerate(thetas):
        nxt = {}
        for (prefix, loop), probability in states.items():
            outputs, probabilities = step_probabilities(loop, input_state[index], theta)
            for leaving, weight in zip(outputs, probabilities):
                key = (prefix + (leaving,), loop + input_state[index] - leaving)
                nxt[key] = nxt.get(key, 0.0) + probability * weight
        states = nxt
    distribution = {}
    for (prefix, loop), probability in states.items():
        distribution[prefix + (loop,)] = (
            distribution.get(prefix + (loop,), 0.0) + probability
        )
    return distribution


def shifted_angles(theta, shift):
    """Stack the base angles with every parameter-shifted variant.

    Parameters
    ----------
    theta : torch.Tensor
        Shape ``(I, K)``: one angle vector per instance.
    shift : float
        Parameter-shift magnitude phi.

    Returns
    -------
    torch.Tensor
        Shape ``(I, 1 + 2K, K)``. Row 0 is the unshifted circuit, then the
        ``+phi`` and ``-phi`` variants of angle k interleaved at rows
        ``1 + 2k`` and ``2 + 2k``. One update step of the paper's algorithm needs
        exactly these ``1 + 2K`` circuits, so batching them into a single sampler
        call is what makes the GPU path worth having.
    """
    instances, n_angles = theta.shape
    out = theta[:, None, :].repeat(1, 1 + 2 * n_angles, 1)
    eye = torch.eye(n_angles, dtype=theta.dtype, device=theta.device)
    out[:, 1::2, :] += shift * eye
    out[:, 2::2, :] -= shift * eye
    return out
