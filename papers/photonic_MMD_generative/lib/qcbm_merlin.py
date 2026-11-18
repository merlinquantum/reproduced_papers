"""MerLin adaptation (Phase 04): the paper's QCBM on MerLin, kept separate from the Phase 03 code.

Three pieces, all labeled as an adaptation of the original JAX reproduction:

M1 — training backend. The paper's Gurvits/MMD estimator (Fig. 1, Eq. (15)) in
PyTorch on a MerLin-compiled interferometer:
- ``merlin_mesh``: ``merlin.CircuitBuilder.add_entangling_layer()`` (rectangular
  MZI mesh, m(m-1) phases ``el_*_li*`` / ``el_*_lo*``), compiled to a
  differentiable unitary by ``merlin.pcvl_pytorch.CircuitConverter`` (public
  MerLin API). The mesh is the identity at internal phases
  li = pi (probed 2026-10-14);
- ``haar``: QR parametrization in torch, same layout and initial point as the
  JAX ``circuits.haar_unitary``.
The Walsh operators K and Gurvits signs Z are drawn with the *same* JAX samplers
and key chain as the original backend, so both backends see identical K, Z at
every step; only the framework (torch vs XLA) differs.

M2 — deployment: exact boson samples of the trained U from the engine's Clifford &
Clifford sampler (``merlin.photonic_engines.clifford.fock_sample``, CUDA/Triton),
seeded with a ``torch.Generator``. ``photonic_engines`` exists only in the MerLin
benchmarking fork (0.5.0.dev0); with public MerLin use ``--sampler perceval``.

M3 — hardware-aware evaluation: exact output distributions of a trained U from
``merlin.QuantumLayer`` (SLOS, FOCK space) under ``perceval.NoiseModel``
(indistinguishability V, transmittance T), read with photon-number-resolving
(PNR) or threshold detectors, postselected on n detected photons.
"""

from __future__ import annotations

import logging
import math
import time

import jax
import numpy as np
import torch

from . import initialization, training

logger = logging.getLogger(__name__)
MERLIN_ANSATZE = ("haar", "merlin_mesh")


def haar_unitary_torch(params: torch.Tensor, m: int) -> torch.Tensor:
    """QR-based Haar-compatible unitary (torch twin of ``circuits.haar_unitary``)."""
    z = torch.complex(
        params[: m * m].reshape(m, m), params[m * m : 2 * m * m].reshape(m, m)
    )
    q, r = torch.linalg.qr(z)
    d = torch.diagonal(r)
    d = torch.where(d.abs() > 0, d / d.abs(), torch.ones_like(d))
    return q * torch.conj(d)[None, :]


class MerlinInterferometer:
    """Trainable m-mode interferometer: MerLin-compiled mesh or torch Haar/QR.

    Parameters
    ----------
    m : int
        Number of modes.
    ansatz : str
        ``"merlin_mesh"`` or ``"haar"``.
    device : str
        Torch device.
    """

    def __init__(self, m: int, ansatz: str, device: str):
        if ansatz not in MERLIN_ANSATZE:
            raise ValueError(
                f"MerLin backend ansatz must be one of {MERLIN_ANSATZE}, got {ansatz!r}."
            )
        self.m, self.ansatz, self.device = m, ansatz, device
        if ansatz == "merlin_mesh":
            import merlin as ml
            from merlin.pcvl_pytorch import CircuitConverter

            builder = ml.CircuitBuilder(n_modes=m)
            builder.add_entangling_layer()
            circuit = builder.to_pcvl_circuit()
            self.parameter_names = [p.name for p in circuit.get_parameters()]
            self.converter = CircuitConverter(
                circuit, input_specs=["el"], dtype=torch.complex128, device=device
            )
            self.num_parameters = len(self.parameter_names)
        else:
            self.parameter_names = None
            self.num_parameters = 2 * m * m

    def unitary(self, params: torch.Tensor) -> torch.Tensor:
        """Differentiable (m, m) complex128 unitary."""
        if self.ansatz == "merlin_mesh":
            return self.converter.to_tensor(params)
        return haar_unitary_torch(params, self.m)

    def initial_parameters(
        self, init: dict, jax_key, unitary_fn_jax=None, input_modes=None, X_train=None
    ):
        """Initial parameters as a float64 torch tensor.

        ``haar`` reuses ``initialization.initial_parameters`` with the same key as the JAX
        backend (identical initial point). ``merlin_mesh`` supports ``identity``
        (li = pi + U[0, eps), lo = U[0, eps)) and ``random`` (U[0, 2 pi)).
        """
        if self.ansatz == "haar":
            params, _ = initialization.initial_parameters(
                init, jax_key, self.m, "haar", unitary_fn_jax, input_modes, X_train
            )
            return torch.as_tensor(
                np.asarray(params), dtype=torch.float64, device=self.device
            )
        rng = np.random.default_rng(
            np.asarray(jax.random.key_data(jax_key)).astype(np.uint64)
        )
        strategy = init["strategy"]
        if strategy == "identity":
            eps = init["perturbation"]
            values = [
                (math.pi if "_li" in name else 0.0) + rng.uniform(0.0, eps)
                for name in self.parameter_names
            ]
        elif strategy == "random":
            values = rng.uniform(0.0, 2 * math.pi, size=self.num_parameters)
        else:
            raise ValueError(
                f"merlin_mesh supports init 'identity' or 'random', got {strategy!r}."
            )
        return torch.as_tensor(
            np.asarray(values), dtype=torch.float64, device=self.device
        )


def walsh_expectations_torch(K: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """E_x[(-1)^{k.x}] per operator (torch twin of ``estimator.walsh_expectations``)."""
    return (1.0 - 2.0 * torch.remainder(X @ K.T, 2.0)).mean(dim=0)


def mmd_terms_torch(U_inputs, K, Z, X, photon_model: str) -> torch.Tensor:
    """Per-operator unbiased MMD^2 terms (torch twin of ``estimator.qcbm_mmd_terms`` / ``distinguishable_mmd_terms``)."""
    T = walsh_expectations_torch(K, X)
    count = X.shape[0]
    data_square = (count * T * T - 1.0) / (count - 1)
    signs = (1.0 - 2.0 * K).to(U_inputs.dtype)  # (ops, m)
    if photon_model == "distinguishable":
        model = torch.prod(signs.real @ U_inputs.abs() ** 2, dim=1)
        return model**2 - 2.0 * model * T + data_square
    # A_k = U_S^dagger diag(signs_k) U_S for every k at once: (ops, n, n)
    A = torch.einsum("im,km,mj->kij", U_inputs.conj().T, signs, U_inputs)
    Zc = Z.to(U_inputs.dtype)
    G = torch.prod(Zc, dim=1)[None, :] * torch.prod(
        torch.einsum("zj,kij->kzi", Zc, A), dim=2
    )  # (ops, n_z)
    n_z = G.shape[1]
    square = (G.sum(dim=1) ** 2 - (G**2).sum(dim=1)) / (n_z * (n_z - 1))
    return square - 2.0 * G.mean(dim=1) * T + data_square


def make_loss(
    interferometer: MerlinInterferometer,
    input_modes,
    operator_sampler,
    n_ops: int,
    n_z: int,
    photon_model: str,
    chunk: int = 250,
):
    """``loss(params, jax_key, X_torch) -> torch scalar``; K, Z from the JAX key exactly as the JAX backend."""
    modes = torch.as_tensor(list(input_modes), device=interferometer.device)
    n = len(input_modes)
    device = interferometer.device

    def loss(params, key, X):
        key_k, key_z = jax.random.split(key)
        K = torch.as_tensor(np.asarray(operator_sampler(key_k, n_ops)), device=device)
        Z = torch.as_tensor(
            np.asarray(
                2 * jax.random.randint(key_z, (n_z, n), 0, 2) - 1, dtype=np.float64
            ),
            device=device,
        )
        U_inputs = interferometer.unitary(params)[:, modes]
        total = 0.0
        for start in range(
            0, n_ops, chunk
        ):  # chunks bound the (ops, n_z, n) intermediate
            total = (
                total
                + mmd_terms_torch(
                    U_inputs, K[start : start + chunk], Z, X, photon_model
                ).real.sum()
            )
        return total / n_ops

    return loss


def train_adam_torch(
    loss_fn,
    params,
    X_train,
    steps: int,
    learning_rate: float,
    key,
    convergence_interval: int | None,
    log_every: int = 1,
) -> dict:
    """Adam with the original semantics (pre-update loss recorded, fresh key per step, window stopping rule)."""
    params = params.clone().requires_grad_(True)
    optimizer = torch.optim.Adam(
        [params], lr=learning_rate, betas=(0.9, 0.999), eps=1e-8
    )
    losses: list[float] = []
    stopped_early = False
    start = time.perf_counter()
    logger.info(
        "TRAINING_STARTED | steps=%d | optimizer=adam(torch) | lr=%g | convergence_interval=%s | backend=merlin",
        steps,
        learning_rate,
        convergence_interval,
    )
    for step in range(1, steps + 1):
        key, subkey = jax.random.split(key)
        step_start = time.perf_counter()
        optimizer.zero_grad()
        value = loss_fn(params, subkey, X_train)
        value.backward()
        optimizer.step()
        loss_value = float(value.detach())
        losses.append(loss_value)
        if step % log_every == 0 or step == 1:
            logger.info(
                "TRAIN_STEP_COMPLETED | step=%d/%d | loss=%.8g | seconds=%.3f",
                step,
                steps,
                loss_value,
                time.perf_counter() - step_start,
            )
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite loss {loss_value} at step {step}.")
        if convergence_interval is not None and training.converged(
            losses, convergence_interval
        ):
            stopped_early = True
            logger.info(
                "TRAINING_CONVERGED | step=%d | rule=window_%d",
                step,
                convergence_interval,
            )
            break
    seconds = time.perf_counter() - start
    logger.info(
        "TRAINING_COMPLETED | steps_run=%d | converged=%s | seconds=%.2f",
        len(losses),
        stopped_early,
        seconds,
    )
    return {
        "params": params.detach(),
        "losses": losses,
        "steps_run": len(losses),
        "converged": stopped_early,
        "seconds": seconds,
    }


def merlin_clifford_samples(
    U: np.ndarray, modes, num_samples: int, seed: int, device: str
) -> np.ndarray:
    """M2: exact boson samples from the engine's GPU Clifford & Clifford sampler, seeded."""
    from merlin.photonic_engines.clifford import fock_sample

    generator = torch.Generator(device=device).manual_seed(int(seed))
    U_t = torch.as_tensor(np.asarray(U), dtype=torch.complex128, device=device)
    occupations = fock_sample(
        U_t, [int(s) for s in modes], int(num_samples), generator=generator
    )
    return occupations.cpu().numpy().astype(np.int64)


def noisy_distribution(
    U: np.ndarray, modes, indistinguishability: float, transmittance: float
) -> dict:
    """M3: exact Fock output distribution of U under a Perceval noise model (MerLin SLOS, FOCK, float64)."""
    import merlin as ml
    import perceval as pcvl

    m = U.shape[0]
    occupied = {int(s) for s in modes}
    layer = ml.QuantumLayer(
        circuit=pcvl.Circuit(m).add(
            0, pcvl.components.Unitary(pcvl.Matrix(np.asarray(U)))
        ),
        input_state=[1 if i in occupied else 0 for i in range(m)],
        measurement_strategy=ml.MeasurementStrategy.probs(ml.ComputationSpace.FOCK),
        noise=pcvl.NoiseModel(
            indistinguishability=indistinguishability, transmittance=transmittance
        ),
        dtype=torch.float64,
    )
    with torch.no_grad():
        probabilities = layer().reshape(-1).cpu().numpy()
    return {
        tuple(int(v) for v in k): float(p)
        for k, p in zip(layer.output_keys, probabilities, strict=True)
    }


def detect(distribution: dict, n: int, detector: str):
    """Apply a detector model and postselect on n detections.

    ``pnr``: photon-number-resolving, keep outcomes with n photons.
    ``threshold``: clicks = min(x, 1), keep patterns with n clicks (collision-free).
    Returns (postselected, renormalized distribution, postselection probability).
    """
    kept: dict = {}
    for outcome, prob in distribution.items():
        if detector == "pnr":
            if sum(outcome) == n:
                kept[outcome] = kept.get(outcome, 0.0) + prob
        elif detector == "threshold":
            clicks = tuple(min(v, 1) for v in outcome)
            if sum(clicks) == n:
                kept[clicks] = kept.get(clicks, 0.0) + prob
        else:
            raise ValueError("detector must be 'pnr' or 'threshold'.")
    rate = sum(kept.values())
    return {k: v / rate for k, v in kept.items()}, rate


def exact_vs_samples_mmd(
    distribution: dict, samples: np.ndarray, sigma: float, chunk: int = 2048
) -> float:
    """Unbiased MMD^2 between an exact distribution p and a sample set: p'Kp - 2 p'K q + U-stat(q)."""
    support = np.array(list(distribution.keys()), dtype=np.float64)
    p = np.array(list(distribution.values()))
    X = np.asarray(samples, dtype=np.float64)

    def kernel_dot(A, wa, B, wb):
        total = 0.0
        for start in range(0, A.shape[0], chunk):
            block = A[start : start + chunk]
            sq = (block**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * block @ B.T
            total += (
                wa[start : start + chunk]
                @ np.exp(-np.maximum(sq, 0.0) / (2 * sigma**2))
                @ wb
            )
        return total

    N = X.shape[0]
    ones = np.ones(N)
    pp = kernel_dot(support, p, support, p)
    pq = kernel_dot(support, p, X, ones / N)
    qq = (kernel_dot(X, ones, X, ones) - N) / (N * (N - 1))
    return float(pp - 2 * pq + qq)


__all__ = [
    "MERLIN_ANSATZE",
    "MerlinInterferometer",
    "make_loss",
    "train_adam_torch",
    "merlin_clifford_samples",
    "noisy_distribution",
    "detect",
    "exact_vs_samples_mmd",
]
