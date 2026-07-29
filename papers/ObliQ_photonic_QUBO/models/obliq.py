"""The ObliQ photonic QUBO solvers: static, VQC, and hybrid.

The whole method, in the order it runs:

1. **Encode** -- :func:`augment_qubo` homogenizes a non-constant diagonal onto one
   ancilla mode, :func:`utils.qubo.normalize_qubo` scales the off-diagonal terms to
   [0, 1], and :func:`qubo_norm_to_theta` turns each entry into a beam-splitter angle.
2. **Run** -- circuits and their execution come from :mod:`models.circuits`.
3. **Decode** -- :func:`distribution_to_result` picks the best bitstring the output
   distribution implies, using the shared readout in :mod:`utils.readout`.
4. **Train** -- :func:`train_obliq_vqc_coeffs` optimizes the VQC coefficients.

:func:`run_obliq_solver` is the entry point: it takes a config's ``solver_options``
as keyword arguments, trains when asked, and returns the decoded result.

**Reproducibility.** Inference is deterministic on the local simulator: the
coefficients are always written explicitly and the simulator returns exact
probabilities, so nothing samples. The one stochastic step is the *initial*
coefficient draw when training without supplied coefficients, which is why
:func:`train_obliq_vqc_coeffs` takes a ``seed``. Given that seed, Adam/SGD and
COBYLA are both deterministic, so the whole training run replays exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from lib.seeding import set_global_seed
from models.circuits import (
    SimulationConfig,
    expected_coeff_count,
    get_coeff_parameter,
    obliq_model,
    prepare_coeff_vector,
    run_distribution,
    set_coeff_parameter,
    static_model,
    vqc_model,
)
from utils.qubo import normalize_qubo, qubo_objective
from utils.readout import EnergyTable, number_mapping


@dataclass
class ObliqResult:
    """One decoded solver output."""

    objective: float
    bitstring: list[int]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# QUBO -> circuit parameters
# ---------------------------------------------------------------------------
def augment_qubo(Q: np.ndarray, tol: float = 1e-9) -> tuple[np.ndarray, bool]:
    """Homogenize a *non-constant* diagonal onto a single ancilla mode.

    ObliQ derives its beam-splitter angles from off-diagonal entries only, so a
    QUBO with varying linear terms (Max-Cut, for instance) cannot be encoded
    directly. Moving the diagonal into couplings with one extra always-occupied
    mode makes the matrix purely off-diagonal at the cost of one mode: an
    ``n``-variable problem becomes an ``(n+1)``-mode circuit.

    A constant diagonal (Max-Clique) shifts every assignment equally, so it is
    left untouched.

    Returns:
        ``(Q_out, augmented)``. Decoding and scoring always use the original Q.
    """
    diag = np.diag(Q)
    if float(np.ptp(diag)) <= tol:
        return Q, False
    n = Q.shape[0]
    Q_aug = np.zeros((n + 1, n + 1), dtype=float)
    Q_aug[:n, :n] = Q
    np.fill_diagonal(Q_aug, 0.0)  # drop the original (and ancilla) diagonal
    Q_aug[:n, n] = diag / 2.0  # linear terms -> ancilla couplings
    Q_aug[n, :n] = diag / 2.0
    return Q_aug, True


def qubo_norm_to_theta(Q_norm: torch.Tensor, num_rep: int) -> torch.Tensor:
    """Convert a normalized QUBO into the anchor layer's ``theta`` values.

    Implements the paper's anchor-point relation: with
    ``theta_ij = 0.5 * arccos(sqrt(1 - Q_ij))`` the joint photon occupancy of
    modes ``i`` and ``j`` satisfies ``<n_i n_j> = 1 - Q_ij``, so a strong QUBO
    coupling suppresses co-occupancy. The weight is divided by ``num_rep ** 2``
    because the anchor is applied ``num_rep`` times with ``theta / num_rep``.

    Emits one value per strict-upper-triangular ``(i, j)`` pair in row-major
    order (``torch.triu_indices``); this ordering MUST match the pair order in
    ``models.circuits._add_anchor_layers``.
    """
    weight = Q_norm / (num_rep**2)
    weight = torch.clamp(weight, 0, 1 - 1e-9)
    theta = torch.acos(torch.sqrt(1 - weight)) * 0.5

    n = theta.shape[-1]
    iu = torch.triu_indices(n, n, offset=1, device=theta.device)
    return theta[..., iu[0], iu[1]]


# ---------------------------------------------------------------------------
# Distribution -> solution
# ---------------------------------------------------------------------------
def solution_guesses(avg_num: np.ndarray, graph_mode: int = 0) -> list[list[int]]:
    """Candidate variable subsets, ordered by average photon occupancy.

    ``graph_mode`` mirrors the original ObliQ code:

    * ``0``    -- generic QUBO: every prefix of the occupancy ranking.
    * ``1``    -- grouped variables: at most one index per group of three.
    * else     -- matrix problems needing unique row/column picks (assignment).
    """
    if graph_mode == 0:
        ordering = np.argsort(avg_num)[::-1].tolist()
        return [ordering[: i + 1] for i in range(len(ordering))]

    if graph_mode == 1:
        ordering = np.argsort(avg_num)[::-1].tolist()
        solution: list[int] = []
        seen_groups = set()
        for idx in ordering:
            group = int(idx / 3)
            if group not in seen_groups:
                solution.append(idx)
                seen_groups.add(group)
        return [solution[: i + 1] for i in range(len(solution))]

    ordering = np.argsort(avg_num)[::-1].tolist()
    size = int(np.sqrt(len(avg_num)))
    solution = []
    selected_rows = set()
    selected_cols = set()
    for idx in ordering:
        row = idx // size
        col = idx % size
        if row not in selected_rows and col not in selected_cols:
            solution.append(idx)
            selected_rows.add(row)
            selected_cols.add(col)
    return [solution[: i + 1] for i in range(len(solution))]


def distribution_to_result(
    distribution: Sequence[float],
    Q: np.ndarray,
    graph_mode: int,
    output_keys: Sequence,
) -> ObliqResult:
    """Decode an output distribution into the best bitstring it implies.

    Each Fock outcome is mapped to a bitstring by the number mapping (a mode with
    at least one photon is a selected variable). Two families of candidates are
    then scored against ``Q`` and the best is returned:

    * prefixes of the average-occupancy ranking (:func:`solution_guesses`), and
    * the single most probable outcome.

    Ties are broken toward the larger support, which favours bigger cliques /
    cuts among equal-energy assignments. Selection is a deterministic function of
    the distribution -- no sampling here.
    """
    size = Q.shape[0]
    fock_table = output_keys

    if len(distribution) != len(fock_table):
        raise ValueError(
            f"Distribution has {len(distribution)} states but the model's output "
            f"basis has {len(fock_table)} keys."
        )

    avg = np.zeros(size)
    best_prob = -1.0
    best_vec = np.zeros(size)

    for occupation, prob in zip(fock_table, distribution, strict=True):
        vector = number_mapping(occupation, size)
        if vector.sum() < 1:
            continue
        avg += vector * prob
        if prob > best_prob:
            best_prob = prob
            best_vec = vector

    candidates = solution_guesses(avg, graph_mode)
    best_value = float("inf")
    best_state = best_vec.copy()

    def _evaluate(indices: list[int]) -> None:
        nonlocal best_value, best_state
        state = np.zeros(size)
        state[indices] = 1
        value = qubo_objective(Q, state)
        if value < best_value or (
            np.isclose(value, best_value) and state.sum() > best_state.sum()
        ):
            best_value = value
            best_state = state

    for indices in candidates:
        _evaluate(indices)

    candidate_idx = np.where(best_vec >= 1)[0].tolist()
    if candidate_idx:
        _evaluate(candidate_idx)

    return ObliqResult(
        objective=-best_value,
        bitstring=best_state.astype(int).tolist(),
        metadata={
            "graph_mode": graph_mode,
            "candidate_count": len(candidates),
            "energy": best_value,
        },
    )


# ---------------------------------------------------------------------------
# Problem evaluator
# ---------------------------------------------------------------------------
def _anchor_theta(Q_elem: np.ndarray, num_rep: int) -> tuple[torch.Tensor, int, bool]:
    """Build the anchor ``theta`` for one QUBO, augmenting a non-constant diagonal.

    Returns ``(theta_values, circuit_size, augmented)``. ``circuit_size`` is
    ``n+1`` when augmented, else ``n``; decoding and scoring still use the
    original Q.
    """
    Q_aug, augmented = augment_qubo(Q_elem)
    Q_t = torch.from_numpy(Q_aug).float().unsqueeze(0)
    theta = qubo_norm_to_theta(normalize_qubo(Q_t), num_rep)
    return theta[0], Q_aug.shape[0], augmented


def estimate_static_result(
    Q: np.ndarray, config: SimulationConfig, num_rep: int, graph_mode: int
) -> ObliqResult:
    """Run the anchor-only circuit -- ObliQ's zero-shot solution.

    Decoding always uses the original ``Q``: when the circuit was augmented its
    output keys carry an extra ancilla mode, which the number mapping slices off.
    """
    theta_values, size, augmented = _anchor_theta(Q, num_rep)
    model = static_model(size, num_rep, theta_values)
    distribution = run_distribution(model, config)
    result = distribution_to_result(
        distribution[0].tolist(), Q, graph_mode, model.output_keys
    )
    result.metadata["augmented"] = augmented
    return result


def estimate_vqc_result(
    Q: np.ndarray,
    coeffs: Sequence[float] | None,
    config: SimulationConfig,
    graph_mode: int,
) -> ObliqResult:
    """Run the anchor-less trainable mesh at the given coefficients.

    The mesh is sized to match the anchor variants (``n+1`` modes when the diagonal
    would have forced an ancilla) so all three run on the same photonic footprint.
    The extra mode carries no problem information here -- there is no anchor -- it
    only equalizes modes and photons.
    """
    size = augment_qubo(np.asarray(Q, dtype=float))[0].shape[0]
    model = vqc_model(size)
    set_coeff_parameter(model, prepare_coeff_vector(coeffs, size))
    distribution = run_distribution(model, config)
    result = distribution_to_result(
        distribution[0].tolist(), Q, graph_mode, model.output_keys
    )
    result.metadata["augmented"] = size > Q.shape[0]
    return result


def estimate_obliq_result(
    Q: np.ndarray,
    input_state: Sequence[int],
    coeffs: Sequence[float] | None,
    config: SimulationConfig,
    num_rep: int,
    graph_mode: int,
) -> ObliqResult:
    """Run the hybrid circuit seeded with ``input_state``."""
    theta_values, size, augmented = _anchor_theta(Q, num_rep)
    # When augmented, the ancilla mode is pinned to 1, so extend the seed state.
    state = [*input_state, 1] if augmented else list(input_state)
    model = obliq_model(size, num_rep, state, theta_values)
    set_coeff_parameter(model, prepare_coeff_vector(coeffs, size))
    distribution = run_distribution(model, config)
    result = distribution_to_result(
        distribution[0].tolist(), Q, graph_mode, model.output_keys
    )
    result.metadata["augmented"] = augmented
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _expected_energy(dist: torch.Tensor, energies: torch.Tensor) -> torch.Tensor:
    """Dense expected QUBO energy -- the autograd loss, over the full basis."""
    energies = energies.to(dtype=dist.dtype, device=dist.device)
    if dist.shape[-1] != energies.shape[0]:
        raise ValueError(
            f"Distribution has {dist.shape[-1]} states but the energy table "
            f"has {energies.shape[0]} (photon-count mismatch)."
        )
    return (dist * energies).sum(dim=-1).mean()


def _expected_energy_over_support(dist: torch.Tensor, table: EnergyTable) -> float:
    """Expected energy over only the outcomes that carry probability mass.

    Zero-probability outcomes contribute nothing to the expectation, so this is
    exact -- and it lets ``table`` materialize energies for just the observed
    outcomes instead of the whole Fock basis (the finite-shot path).
    """
    flat = dist.reshape(-1)
    support = torch.nonzero(flat, as_tuple=True)[0]
    return float((flat[support] * table.for_indices(support)).sum())


def _train_autograd(
    model,
    table: EnergyTable,
    max_iter: int,
    learning_rate: float,
    optimizer: str,
    betas: tuple[float, float],
    epsilon: float,
    verbose: bool,
) -> list[float]:
    """Local differentiable path: backprop the exact expected energy (Adam/SGD)."""
    optim = (
        torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=epsilon)
        if optimizer == "adam"
        else torch.optim.SGD(model.parameters(), lr=learning_rate)
    )
    energies = table.full()
    model.train()

    history: list[float] = []
    for iteration in range(1, max_iter + 1):
        optim.zero_grad()
        loss = _expected_energy(model(), energies)
        loss.backward()
        optim.step()
        history.append(float(loss.detach()))
        if verbose:
            print(f"[ObliQ train] iteration {iteration}: energy={history[-1]:.4f}")
    return history


def _train_cobyla(
    model,
    table: EnergyTable,
    max_iter: int,
    config: SimulationConfig,
    rhobeg: float,
    verbose: bool,
) -> tuple[list[float], dict]:
    """Gradient-free path: forward-only energy, so it trains through a remote backend.

    Each forward pass yields a finite-shot distribution, so the energy is taken over
    only the sampled outcomes. COBYLA is not monotone, so the best point seen is kept
    rather than the last one.
    """
    from scipy.optimize import minimize

    model.eval()
    x0 = get_coeff_parameter(model).detach().reshape(-1).cpu().numpy().astype(float)

    history: list[float] = []
    best = {"loss": float("inf"), "x": x0}

    def objective(vec):
        set_coeff_parameter(model, vec)
        with torch.no_grad():
            loss = _expected_energy_over_support(run_distribution(model, config), table)
        history.append(loss)
        if loss < best["loss"]:
            best["loss"], best["x"] = loss, np.array(vec, dtype=float)
        if verbose:
            print(f"[ObliQ train] cobyla eval {len(history)}: energy={loss:.4f}")
        return loss

    # SciPy's COBYLA "maxiter" option is really MAXFUN (function-evaluation budget).
    # To make max_iter comparable to an Adam/SGD step count, scale by num_vars: one
    # gradient step uses full-gradient information, whose derivative-free equivalent
    # costs ~num_vars evaluations. Floor at num_vars+2 (COBYLA's minimum) so the
    # budget is always valid.
    num_vars = x0.size
    maxfun = max(max_iter * num_vars, num_vars + 2)
    result = minimize(
        objective, x0, method="COBYLA", options={"maxiter": maxfun, "rhobeg": rhobeg}
    )
    set_coeff_parameter(model, best["x"])
    return history, {
        "best_energy": best["loss"],
        "success": bool(result.success),
        "message": str(result.message),
    }


def _build_trainable_model(
    Q: np.ndarray,
    variant: str,
    num_rep: int,
    graph_mode: int,
    config: SimulationConfig,
):
    """Build the model to train; returns it with its circuit size.

    The hybrid bakes in the anchor theta and augments (one ancilla mode) when the
    diagonal is non-constant -- which also widens the VQC layers -- so the circuit
    size has to be resolved here, before the coefficients are sized.
    """
    if variant == "obliq-hybrid":
        seed_bits = estimate_static_result(Q, config, num_rep, graph_mode).bitstring
        theta_values, circuit_size, augmented = _anchor_theta(
            np.asarray(Q, dtype=float), num_rep
        )
        state = [*seed_bits, 1] if augmented else list(seed_bits)
        return obliq_model(circuit_size, num_rep, state, theta_values), circuit_size

    # obliq-vqc, size-matched to the anchor variants for fairness.
    circuit_size = augment_qubo(np.asarray(Q, dtype=float))[0].shape[0]
    return vqc_model(circuit_size), circuit_size


def train_obliq_vqc_coeffs(
    Q: np.ndarray,
    variant: str = "obliq-vqc",
    initial_coeffs: Sequence[float] | None = None,
    nsamples: int = 5_000,
    num_rep: int = 10,
    graph_mode: int = 0,
    backend: str | None = None,
    token: str | None = None,
    max_iter: int = 50,
    learning_rate: float = 0.05,
    finite_diff_step: float = np.pi / 4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    verbose: bool = False,
    optimizer: str = "adam",
    init_scale: float = 0.1,
    seed: int | None = None,
) -> tuple[list[float], dict]:
    """Optimize the VQC coefficients, returning ``(coeffs, history)``.

    ``optimizer`` selects the path:

    * ``"adam"`` / ``"sgd"`` -- :func:`_train_autograd`, backprop through the local
      differentiable simulator (``backend`` must be ``None``).
    * ``"cobyla"``           -- :func:`_train_cobyla`, forward evaluations only, so
      it can train through a remote ``backend`` such as a noisy ``"sim:ascella"`` or
      a real ``"qpu:ascella"``.

    ``seed`` fixes the one random step -- the initial coefficient draw -- and is also
    applied to the global torch/NumPy generators, so a run is reproducible end to
    end. Pass ``initial_coeffs`` to remove the randomness entirely. Both optimizers
    are deterministic from their starting point, so the returned coefficients and
    ``history["energies"]`` replay exactly.

    Raises:
        ValueError: for a non-trainable variant, an unknown optimizer, or asking for
            autograd through a remote backend.
    """
    if variant not in {"obliq-vqc", "obliq-hybrid"}:
        raise ValueError(
            "Coefficient training is only supported for VQC-enabled variants."
        )

    optimizer = optimizer.lower()
    if optimizer not in {"adam", "sgd", "cobyla"}:
        raise ValueError("Optimizer must be 'adam', 'sgd', or 'cobyla'.")
    if optimizer in {"adam", "sgd"} and backend is not None:
        raise ValueError(
            "Adam/SGD training is autograd-based and runs on the local simulator; "
            "use optimizer='cobyla' to train through a remote backend."
        )

    # Seed before any model is built: MerLin initializes its trainable tensor from
    # the global torch generator, and although every coefficient is overwritten
    # below, seeding keeps the whole call graph replayable.
    set_global_seed(seed)

    config = SimulationConfig(nsamples=nsamples, backend=backend, token=token)
    model, circuit_size = _build_trainable_model(
        Q, variant, num_rep, graph_mode, config
    )

    if initial_coeffs is None:
        generator = np.random.default_rng(seed)
        coeffs = init_scale * generator.standard_normal(
            expected_coeff_count(circuit_size)
        )
    else:
        coeffs = prepare_coeff_vector(initial_coeffs, circuit_size).astype(float)
    set_coeff_parameter(model, coeffs)

    table = EnergyTable(Q, model.output_keys)
    extra: dict = {}
    if optimizer == "cobyla":
        history, extra = _train_cobyla(
            model, table, max_iter, config, finite_diff_step, verbose
        )
    else:
        history = _train_autograd(
            model,
            table,
            max_iter,
            learning_rate,
            optimizer,
            (beta1, beta2),
            epsilon,
            verbose,
        )

    trained_coeffs = get_coeff_parameter(model).detach().reshape(-1).tolist()
    return trained_coeffs, {
        "energies": history,
        "optimizer": optimizer,
        "seed": seed,
        **extra,
    }


def run_obliq_solver(
    Q: np.ndarray,
    variant: str = "obliq-static",
    nsamples: int = 1_024,
    num_rep: int = 10,
    graph_mode: int = 0,
    coeffs: Sequence[float] | None = None,
    train: object = False,
    backend: str | None = None,
    token: str | None = None,
    seed: int | None = None,
) -> ObliqResult:
    """Run one ObliQ variant on a QUBO matrix and return the decoded result.

    This is the whole solver behind one call, and it takes a config's
    ``solver_options`` directly as keyword arguments -- an unrecognised option is
    a ``TypeError`` from the signature itself, so no separate validation exists.

    Runs on the local simulator by default. Set ``backend`` to a Quandela platform
    (``"sim:slos"``, ``"sim:ascella"``, ``"qpu:ascella"``, ...) to offload
    inference to Quandela Cloud; the token is resolved by
    :func:`models.backend.read_quandela_api_key`.

    The QUBO is rescaled to unit maximum magnitude before encoding (the anchor
    angles are scale-free) and the objective is scaled back afterwards.

    Args:
        Q: QUBO as a symmetric matrix.
        variant: ``"obliq-static"``, ``"obliq-vqc"`` or ``"obliq-hybrid"``.
        nsamples: shots per forward pass (remote backends only).
        num_rep: anchor repetitions.
        graph_mode: candidate-generation mode for decoding.
        coeffs: VQC coefficients; ``None`` means all-zero (the identity mesh),
            or the trained result when ``train`` is set.
        train: ``False`` to skip training, or a dict of
            :func:`train_obliq_vqc_coeffs` keyword arguments (``True`` for its
            defaults). Unavailable for ``obliq-static``.
        backend: Quandela platform, or ``None`` for the local simulator.
        token: explicit Quandela API token.
        seed: seeds the coefficient draw when training without ``coeffs``.

    Raises:
        ValueError: for an unknown variant, or training a static circuit.
    """
    if train:
        if variant == "obliq-static":
            raise ValueError(
                "Coefficient training is only available for VQC or hybrid variants."
            )
        train_kwargs = {} if train is True else dict(train)
        train_kwargs.setdefault("seed", seed)
        coeffs, _history = train_obliq_vqc_coeffs(
            np.asarray(Q, dtype=float),
            variant=variant,
            initial_coeffs=coeffs,
            nsamples=nsamples,
            num_rep=num_rep,
            graph_mode=graph_mode,
            backend=backend,
            token=token,
            **train_kwargs,
        )

    Q = np.array(Q, dtype=float)
    scale = np.max(np.abs(Q)) or 1.0
    Q = Q / scale
    size = Q.shape[0]
    config = SimulationConfig(nsamples=nsamples, backend=backend, token=token)
    if variant == "obliq-static":
        result = estimate_static_result(Q, config, num_rep, graph_mode)
    elif variant == "obliq-vqc":
        result = estimate_vqc_result(Q, coeffs, config, graph_mode)
    elif variant == "obliq-hybrid":
        input_result = estimate_static_result(Q, config, num_rep, graph_mode)
        result = estimate_obliq_result(
            Q, input_result.bitstring, coeffs, config, num_rep, graph_mode
        )
    else:
        raise ValueError(
            "Unknown ObliQ variant. Use 'obliq-static', 'obliq-vqc', or 'obliq-hybrid'."
        )

    result.objective *= scale
    if "energy" in result.metadata:
        result.metadata["normalized_energy"] = result.metadata["energy"]
        result.metadata["energy"] = result.metadata["energy"] * scale
    # ``augmented`` is set by the estimate routines: True when the QUBO's
    # diagonal forced a homogenization ancilla, i.e. the circuit ran on n+1 modes.
    result.metadata.update(
        {
            "variant": variant,
            "nsamples": nsamples,
            "num_rep": num_rep,
            "coeffs_provided": coeffs is not None,
            "circuit_modes": size + 1 if result.metadata.get("augmented") else size,
            "trained": bool(train),
        }
    )
    return result
