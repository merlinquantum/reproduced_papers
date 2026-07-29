"""Photonic CVaR-VQE baseline (MerLin/torch).

Mirrors the ObliQ methodology in :mod:`models.obliq`: the MerLin
``QuantumLayer`` emits a probability distribution over Fock states, a constant
energy table maps each outcome to its QUBO objective, and the *differentiable*
CVaR of that distribution is minimized. The same two optimizers are supported:

* ``adam`` / ``sgd`` -- autograd on the local differentiable simulator
  (``backend`` must be ``None``; gradients cannot flow through a sampled backend).
* ``cobyla``         -- gradient-free (SciPy), forward evaluations only, so it can
  train through a remote ``backend`` such as ``"sim:ascella"`` or a real
  ``"qpu:ascella"`` QPU.

It also shares the harness's QUBO (:func:`utils.qubo.build_qubo`) and energy table
(:class:`utils.readout.EnergyTable`); what makes it a *different* solver is the
ansatz, the CVaR objective, and searching over both readout polarities.

**Reproducibility.** Unlike ObliQ, this solver has two stochastic steps: the
interferometer's trainable parameters are *not* overwritten before training (the
random initialization is the starting point), and the final bitstring is drawn by
``torch.multinomial``. Both read torch's global generator, so ``seed`` is applied
via :func:`lib.seeding.set_global_seed` and fixes the whole run.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import perceval as pcvl
import torch
from lib.seeding import set_global_seed
from merlin import MeasurementStrategy, QuantumLayer
from merlin.core import ComputationSpace
from models.circuits import (
    SimulationConfig,
    get_coeff_parameter,
    run_distribution,
    set_coeff_parameter,
)
from networkx import Graph
from perceval.components import GenericInterferometer
from perceval.components.unitary_components import BS
from scipy.optimize import minimize
from utils.qubo import build_qubo
from utils.readout import EnergyTable, number_mapping


# ---------------------------------------------------------------------------
# Photonic circuit / model
# ---------------------------------------------------------------------------
def _device_setup(nb_modes: int, nb_inputs: int):
    """Configure the input states and the trainable interferometer."""
    inputs: list[pcvl.BasicState] = []
    for k in range(1, nb_inputs + 1):
        interval = 2**k
        pattern = [int((i + interval / 2 + 1) % interval == 0) for i in range(nb_modes)]
        inputs.append(pcvl.BasicState(pattern))

    circuit = GenericInterferometer(
        nb_modes, lambda idx: BS(theta=pcvl.P(f"coeffs{idx}"))
    )
    return circuit, inputs


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def _cvar(
    probabilities: torch.Tensor, energies: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Differentiable CVaR of a distribution over outcomes with fixed energies.

    Identical semantics to the greedy scalar version (accumulate probability mass
    up to ``alpha``, clipping the straddling outcome), but expressed on tensors so
    gradients flow back into ``probabilities``. Energies are constant, so the sort
    order is fixed and only the probability weights carry gradient. Works on both
    an exact (local) and a sampled (remote) distribution tensor.
    """
    order = torch.argsort(energies)
    probs = probabilities[order]
    vals = energies[order]
    cumulative = torch.cumsum(probs, dim=0)
    # weight_i = min(cumsum_i, alpha) - cumsum_{i-1}, clamped at 0 past the quantile
    weights = torch.clamp(
        torch.clamp(cumulative, max=alpha) - (cumulative - probs), min=0.0
    )
    total = weights.sum().clamp_min(torch.finfo(probabilities.dtype).eps)
    return (weights * vals).sum() / total


# ---------------------------------------------------------------------------
# Per-configuration training + decoding
# ---------------------------------------------------------------------------
def _cvar_over_support(
    distribution: torch.Tensor, table: EnergyTable, alpha: float
) -> float:
    """CVaR restricted to the outcomes that carry probability mass.

    Zero-probability outcomes contribute zero CVaR weight, so dropping them is
    exact -- and it lets ``table`` materialize energies only for the observed
    support instead of the full Fock space (the finite-shot path).
    """
    support = torch.nonzero(distribution, as_tuple=True)[0]
    probs = distribution[support]
    energies = table.for_indices(support)
    return float(_cvar(probs, energies, alpha))


def _train_autograd(model, table, cvar_alpha, max_iter, learning_rate, optimizer):
    """Local differentiable path: backprop the exact-probability CVaR (Adam/SGD)."""
    optim = (
        torch.optim.Adam(model.parameters(), lr=learning_rate)
        if optimizer == "adam"
        else torch.optim.SGD(model.parameters(), lr=learning_rate)
    )
    energies = table.full()
    model.train()
    for _ in range(max_iter):
        optim.zero_grad()
        distribution = model().reshape(-1)
        loss = _cvar(distribution, energies, cvar_alpha)
        loss.backward()
        optim.step()


def _train_cobyla(model, table, cvar_alpha, max_iter, config):
    """Gradient-free path: forward-only CVaR, so it trains through a remote backend.

    Each forward pass yields a finite-shot distribution, so the CVaR is taken over
    only the sampled outcomes -- energies are computed for exactly those outcomes
    rather than the entire output space.
    """
    model.eval()
    x0 = get_coeff_parameter(model).detach().reshape(-1).cpu().numpy().astype(float)

    def objective(vec):
        set_coeff_parameter(model, vec)
        with torch.no_grad():
            distribution = run_distribution(model, config).reshape(-1)
        return _cvar_over_support(distribution, table, cvar_alpha)

    # SciPy's COBYLA "maxiter" option is really MAXFUN (function-evaluation
    # budget). To make max_iter comparable to an Adam/SGD step count, scale by
    # num_vars: one gradient step uses full-gradient information, whose
    # derivative-free equivalent costs ~num_vars evaluations. Floor at
    # num_vars+2 (COBYLA's minimum) so the budget is always valid.
    num_vars = x0.size
    maxfun = max(max_iter * num_vars, num_vars + 2)
    best = minimize(objective, x0, method="COBYLA", options={"maxiter": maxfun})
    set_coeff_parameter(model, best.x)


def _run_configuration(
    circuit: pcvl.Circuit,
    invert: bool,
    input_state: pcvl.BasicState,
    H: np.ndarray,
    cvar_alpha: float,
    max_iter: int,
    learning_rate: float,
    optimizer: str,
    config: SimulationConfig,
):
    """Minimize the CVaR objective for one (readout, input) configuration.

    The reference implementation calls the two readout polarities "parities"; see
    :func:`utils.readout.number_mapping` for why that name is a misnomer.
    """
    model = QuantumLayer(
        input_size=0,
        circuit=circuit,
        input_state=list(input_state),
        trainable_parameters=["coeffs"],
        dtype=torch.float32,
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=ComputationSpace.FOCK
        ),
    )
    table = EnergyTable(
        H, model.output_keys, mapping=partial(number_mapping, invert=invert)
    )

    if optimizer == "cobyla":
        _train_cobyla(model, table, cvar_alpha, max_iter, config)
    else:
        _train_autograd(model, table, cvar_alpha, max_iter, learning_rate, optimizer)

    model.eval()
    with torch.no_grad():
        distribution = run_distribution(model, config).reshape(-1)
    return distribution, table, model.output_keys


def _best_bitstring(
    distribution, table, output_keys, invert: bool, size: int, nb_samples: int
) -> np.ndarray:
    """Lowest-energy outcome among ``nb_samples`` draws from the final distribution.

    Sampling uses torch's global generator, seeded in :func:`qubo_solver`.
    """
    draws = torch.multinomial(distribution, nb_samples, replacement=True)
    observed = torch.unique(draws)
    best = observed[torch.argmin(table.for_indices(observed))].item()
    return number_mapping(output_keys[best], size, invert=invert).astype(int)


def run_photonic_cvarvqe(
    graph: Graph,
    problem_type: str,
    nb_samples: int = 2048,
    nb_inputs: int = 1,
    max_iter: int = 50,
    learning_rate: float = 0.05,
    cvar_alpha: float = 1.0,
    optimizer: str = "adam",
    backend: str | None = None,
    token: str | None = None,
    seed: int | None = None,
) -> list[int]:
    """Solve one instance with photonic CVaR-VQE; returns the best bitstring.

    Searches over both readout polarities and every input state, training each
    configuration independently, and keeps the assignment with the lowest expected
    energy.

    ``optimizer`` selects the training method:

    * ``"adam"`` / ``"sgd"`` -- autograd on the local differentiable simulator
      (requires ``backend=None``).
    * ``"cobyla"``           -- gradient-free forward evaluations, so it can train
      through a remote ``backend`` (e.g. ``"sim:ascella"``, ``"qpu:ascella"``).

    ``seed`` seeds the global torch/NumPy generators, which makes both the layer's
    parameter initialization and the final multinomial draw reproducible.
    """
    optimizer = optimizer.lower()
    if optimizer not in {"adam", "sgd", "cobyla"}:
        raise ValueError("optimizer must be 'adam', 'sgd', or 'cobyla'.")
    if optimizer in {"adam", "sgd"} and backend is not None:
        raise ValueError(
            "Adam/SGD training is autograd-based and runs on the local simulator; "
            "use optimizer='cobyla' to train through a remote backend."
        )

    set_global_seed(seed)

    # The same QUBO the rest of the harness uses (:func:`utils.qubo.build_qubo`),
    # rather than a private re-encoding: the baseline must optimize exactly the
    # objective its bitstring is later scored against.
    H = build_qubo(problem_type, graph)
    circuit, inputs = _device_setup(len(H), nb_inputs)
    config = SimulationConfig(nsamples=nb_samples, backend=backend, token=token)

    best_state = None
    best_avg_energy = float("inf")
    # Both readout polarities, both input states: the reference solver's search.
    for invert in (True, False):
        for input_state in inputs:
            distribution, table, output_keys = _run_configuration(
                circuit,
                invert,
                input_state,
                H,
                cvar_alpha,
                max_iter,
                learning_rate,
                optimizer,
                config,
            )
            support = torch.nonzero(distribution, as_tuple=True)[0]
            avg_energy = float(
                (distribution[support] * table.for_indices(support)).sum()
            )
            if avg_energy < best_avg_energy:
                best_avg_energy = avg_energy
                best_state = _best_bitstring(
                    distribution, table, output_keys, invert, len(H), nb_samples
                )

    if best_state is None:
        best_state = np.zeros(len(H), dtype=int)

    return np.asarray(best_state, dtype=int).tolist()
