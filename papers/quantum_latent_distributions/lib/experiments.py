"""The four studies this reproduction can run.

Each function has the same contract as the shared runtime entry point: it takes
the resolved config and a run directory, writes structured artifacts into that
directory, and returns a summary dictionary.

=============================  ===============================================
``sampler_validation``         boson sampler vs MerLin's exact simulation
``mixture_of_gaussians``       paper Fig. 2 -- mode coverage / interpolation
``synthetic_datasets``         paper Table I -- L1 distance to nearest integer
``qm9``                        paper Table II -- MolGAN-style generation
=============================  ===============================================
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lib.circuits import haar_unitary
from lib.datasets import (
    DATA_MODES,
    bernoulli_dataset,
    mixture_centers,
    mixture_samples,
    quantum_dataset,
)
from lib.gan import Critic, Generator, WGANGPConfig, train_wgan_gp
from lib.latents import (
    build_latent,
    exact_distribution,
    sample_boson,
    sample_distinguishable,
)
from lib.metrics import (
    interpolation_rate,
    l1_to_nearest_integer,
    mmd_rbf,
    mode_coverage,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EXPERIMENTS",
    "run_mixture_of_gaussians",
    "run_qm9",
    "run_sampler_validation",
    "run_synthetic_datasets",
]

_TARGET_BUILDERS = {"quantum": quantum_dataset, "bernoulli": bernoulli_dataset}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_list(value: Any) -> list[str]:
    """Accept either a JSON list or a comma-separated CLI string."""
    if isinstance(value, str):
        return [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    return list(value)


def _mean_sem(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    sem = float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
    return {"mean": float(array.mean()), "sem": sem, "n": int(len(array))}


def _fano(bank: np.ndarray) -> float:
    """Fano factor (variance / mean) of the per-mode photon counts.

    Greater than 1 is super-Poissonian (bunching), less than 1 is
    sub-Poissonian (anti-bunching). This is the second-moment statistic that
    separates a boson sampler from the paper's distinguishable-photon control,
    even though their mean occupancies agree.

    Parameters
    ----------
    bank : numpy.ndarray
        Sample bank of shape ``(n_samples, n_modes)``.

    Returns
    -------
    float
        Mean per-mode variance divided by mean per-mode mean.
    """
    return float(bank.var(axis=0).mean() / bank.mean(axis=0).mean())


def _write_json(run_dir: Path, name: str, payload: Any) -> None:
    (run_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_latent(cfg: dict[str, Any], kind: str, seed: int):
    latent_cfg = cfg["latent"]
    if latent_cfg.get("source", "simulation") == "hardware" and kind == "boson":
        # Real-QPU banks are drawn once and reused; see lib/hardware.py.
        from lib.hardware import hardware_latent

        hw = latent_cfg["hardware"]
        return hardware_latent(
            hw["platform"],
            n_modes=latent_cfg["dim"],
            n_photons=latent_cfg.get("n_photons") or latent_cfg["dim"] // 2,
            n_samples=latent_cfg["bank_size"],
            architecture=latent_cfg["architecture"],
            seed=seed,
        )
    return build_latent(
        kind,
        latent_cfg["dim"],
        seed,
        n_photons=latent_cfg.get("n_photons"),
        architecture=latent_cfg["architecture"],
        bank_size=latent_cfg["bank_size"],
        normalize=latent_cfg["normalize"],
    )


def _build_gan(cfg: dict[str, Any], out_dim: int) -> tuple[Generator, Critic]:
    model_cfg = cfg["model"]
    activation = model_cfg.get("activation", "leaky_relu")
    generator = Generator(
        cfg["latent"]["dim"],
        out_dim,
        hidden=tuple(model_cfg["generator_hidden"]),
        reinject=model_cfg["latent_reinjection"],
        activation=activation,
    )
    critic = Critic(
        out_dim, hidden=tuple(model_cfg["critic_hidden"]), activation=activation
    )
    return generator, critic


def _wgan_config(cfg: dict[str, Any]) -> WGANGPConfig:
    training = cfg["training"]
    return WGANGPConfig(
        iterations=training["iterations"],
        batch_size=cfg["dataset"]["batch_size"],
        lr=training["lr"],
        optimizer=training.get("optimizer", "adam"),
        betas=tuple(training["betas"]),
        n_critic=training["n_critic"],
        gp_weight=training["gp_weight"],
        device=cfg.get("device", "cpu"),
        log_every=max(training["iterations"] // 5, 1),
    )


# --------------------------------------------------------------------------- #
# 1. sampler validation
# --------------------------------------------------------------------------- #
def run_sampler_validation(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Check the boson sampler against MerLin's exact Fock-space distribution.

    Four checks are written to ``sampler_validation.json``: convergence of the
    empirical histogram to the exact distribution, equality of the
    distinguishable control's *mean occupancy*, the Fano factor of both (which
    is where the control stops matching), and the size of the probability mass
    that ``ComputationSpace.UNBUNCHED`` would discard.

    Parameters
    ----------
    cfg : dict
        Resolved configuration; reads ``sampler_validation``.
    run_dir : pathlib.Path
        Timestamped run directory created by the shared runtime.

    Returns
    -------
    dict
        Summary payload, also written to disk.
    """
    params = cfg["sampler_validation"]
    modes, photons = params["modes"], params["photons"]
    unitary = haar_unitary(modes, np.random.default_rng(cfg["seed"]))
    keys, exact = exact_distribution(unitary, photons)

    def empirical(samples: np.ndarray) -> np.ndarray:
        counts = Counter(map(tuple, samples.tolist()))
        hist = [counts.get(tuple(int(x) for x in key), 0) for key in keys]
        return np.asarray(hist, dtype=float) / len(samples)

    convergence = []
    for n_samples in params["sample_counts"]:
        samples = sample_boson(unitary, photons, n_samples, seed=cfg["seed"])
        tvd = 0.5 * float(np.abs(empirical(samples) - exact).sum())
        convergence.append(
            {"n_samples": n_samples, "tvd": tvd, "inv_sqrt_n": 1.0 / np.sqrt(n_samples)}
        )
        logger.info("sampler TVD at N=%d: %.4f", n_samples, tvd)

    control_n = max(params["sample_counts"])
    quantum = sample_boson(unitary, photons, control_n, seed=cfg["seed"])
    classical = sample_distinguishable(unitary, photons, control_n, seed=cfg["seed"])
    summary = {
        "n_fock_states": len(keys),
        "exact_mass": float(exact.sum()),
        "convergence": convergence,
        "control": {
            # First moment: the control cannot be told apart from the boson
            # sampler by mean occupancy. This is what the paper's framing rests
            # on -- and it is *all* that matches.
            "max_mean_occupancy_difference": float(
                np.abs(quantum.mean(0) - classical.mean(0)).max()
            ),
            # Second moment: it does NOT match. Boson sampling always bunches
            # relative to the control (fano_quantum > fano_distinguishable, by
            # a factor of about 1.4 at half filling). Whether that puts it above
            # 1 in absolute terms depends on the mode count: at the Table I size
            # of 16 modes / 8 photons it is 1.28 against 0.89, i.e. genuinely
            # super- against sub-Poissonian, while at the small 6/3 size used
            # for this validation the fixed photon total suppresses both below 1
            # (0.93 against 0.67) without changing the ratio.
            #
            # Either way the two latents differ in their single-mode *marginals*
            # and not only in their correlations, so the control does not isolate
            # interference. See the classical challenger study in the README.
            "fano_quantum": _fano(quantum),
            "fano_distinguishable": _fano(classical),
            # The joint distributions are far apart, which is what the control
            # was meant to demonstrate.
            "tvd_distinguishable_vs_quantum": 0.5
            * float(np.abs(empirical(classical) - exact).sum()),
        },
        "bunching_fraction": float((quantum.max(axis=1) > 1).mean()),
    }
    _write_json(run_dir, "sampler_validation.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# 2. 2D mixture of Gaussians (paper Fig. 2)
# --------------------------------------------------------------------------- #
def run_mixture_of_gaussians(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Reproduce the 2D mixture-of-Gaussians comparison (paper Fig. 2).

    Trains one WGAN-GP per (latent, repeat) pair and measures how much each
    model interpolates between the mixture modes. Generated samples are saved as
    ``samples_<latent>_seed<k>.npy`` so ``utils/plot_mixture.py`` can render the
    Fig. 2 style panel.

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    run_dir : pathlib.Path
        Timestamped run directory created by the shared runtime.

    Returns
    -------
    dict
        Per-run records plus a mean/sem aggregate, also written to disk.
    """
    mixture = cfg["mixture"]
    centers = mixture_centers(mixture["n_components"], mixture["radius"])
    # A blob's extent is radial_std across and radius * tangential_std along the
    # circle; the capture disc uses the larger of the two.
    blob_scale = max(
        mixture["radial_std"], mixture["radius"] * mixture["tangential_std"]
    )
    capture_radius = mixture["capture_sigmas"] * blob_scale
    kinds = _as_list(cfg["latent"]["kinds"])
    records: list[dict[str, Any]] = []

    for repeat in range(cfg["evaluation"]["repeats"]):
        seed = cfg["seed"] + repeat
        data = torch.from_numpy(
            mixture_samples(
                cfg["dataset"]["size"],
                mixture["n_components"],
                mixture["radius"],
                mixture["radial_std"],
                mixture["tangential_std"],
                seed=seed,
            )
        )
        for kind in kinds:
            latent = _make_latent(cfg, kind, seed)
            torch.manual_seed(seed)
            generator, critic = _build_gan(cfg, out_dim=2)
            train_wgan_gp(data, latent, generator, critic, _wgan_config(cfg))

            with torch.no_grad():
                fake = generator(latent.sample(cfg["evaluation"]["n_samples"])).numpy()
            coverage = mode_coverage(fake, centers, capture_radius)
            record = {
                "latent": kind,
                "seed": seed,
                "interpolation_rate": interpolation_rate(fake, centers, capture_radius),
                "n_modes_covered": coverage["n_modes_covered"],
                "captured_fraction": coverage["captured_fraction"],
                "mmd": mmd_rbf(fake[:2000], data[:2000].numpy()),
            }
            records.append(record)
            np.save(run_dir / f"samples_{kind}_seed{seed}.npy", fake[:4000])
            logger.info(
                "seed %d | %-15s interpolation=%.3f modes=%d/%d",
                seed,
                kind,
                record["interpolation_rate"],
                record["n_modes_covered"],
                mixture["n_components"],
            )

    aggregate = {
        kind: {
            metric: _mean_sem([r[metric] for r in records if r["latent"] == kind])
            for metric in ("interpolation_rate", "n_modes_covered", "mmd")
        }
        for kind in kinds
    }
    summary = {"records": records, "aggregate": aggregate, "centers": centers.tolist()}
    _write_json(run_dir, "metrics.json", records)
    _write_json(run_dir, "summary.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# 3. synthetic discrete datasets (paper Table I)
# --------------------------------------------------------------------------- #
def run_synthetic_datasets(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Reproduce Table I: L1 distance between generated values and integers.

    Parameters
    ----------
    cfg : dict
        Resolved configuration; reads ``dataset.targets``.
    run_dir : pathlib.Path
        Timestamped run directory created by the shared runtime.

    Returns
    -------
    dict
        Per-run records plus the Table I style aggregate, also written to disk.
    """
    kinds = _as_list(cfg["latent"]["kinds"])
    targets = _as_list(cfg["dataset"]["targets"])
    records: list[dict[str, Any]] = []

    for target in targets:
        build_target = _TARGET_BUILDERS[target]
        for repeat in range(cfg["evaluation"]["repeats"]):
            seed = cfg["seed"] + repeat
            data = torch.from_numpy(build_target(cfg["dataset"]["size"], seed))
            for kind in kinds:
                latent = _make_latent(cfg, kind, seed)
                torch.manual_seed(seed)
                generator, critic = _build_gan(cfg, out_dim=DATA_MODES)
                train_wgan_gp(data, latent, generator, critic, _wgan_config(cfg))

                with torch.no_grad():
                    fake = generator(latent.sample(cfg["evaluation"]["n_samples"]))
                record = {
                    "target": target,
                    "latent": kind,
                    "seed": seed,
                    "l1_nearest_int": l1_to_nearest_integer(fake),
                    # Total photon number is a cheap sanity check: the quantum
                    # target always carries exactly 8 photons.
                    "mean_total": float(fake.sum(1).mean()),
                }
                records.append(record)
                logger.info(
                    "[%s] seed %d | %-15s L1=%.4f",
                    target,
                    seed,
                    kind,
                    record["l1_nearest_int"],
                )

    aggregate = {
        target: {
            kind: _mean_sem(
                [
                    r["l1_nearest_int"]
                    for r in records
                    if r["latent"] == kind and r["target"] == target
                ]
            )
            for kind in kinds
        }
        for target in targets
    }
    summary = {"records": records, "aggregate": aggregate}
    _write_json(run_dir, "metrics.json", records)
    _write_json(run_dir, "summary.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# 4. QM9 (paper Table II)
# --------------------------------------------------------------------------- #
def run_qm9(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Reproduce Table II: MolGAN-style molecular generation on QM9.

    Requires the optional extras in ``requirements.txt`` (rdkit, fcd-torch,
    torch-geometric) and a GPU for anything beyond a smoke run.

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    run_dir : pathlib.Path
        Timestamped run directory created by the shared runtime.

    Returns
    -------
    dict
        Per-run records plus aggregate FCD / valid-and-unique / novel counts.
    """
    from lib.molgan import evaluate_molecules, load_qm9_dense, train_molgan

    kinds = _as_list(cfg["latent"]["kinds"])
    edges, nodes, train_smiles = load_qm9_dense(cfg)
    records: list[dict[str, Any]] = []

    for repeat in range(cfg["evaluation"]["repeats"]):
        seed = cfg["seed"] + repeat
        for kind in kinds:
            latent = _make_latent(cfg, kind, seed)
            torch.manual_seed(seed)
            generator = train_molgan(cfg, latent, edges, nodes, seed)
            metrics = evaluate_molecules(cfg, generator, latent, train_smiles)
            records.append({"latent": kind, "seed": seed, **metrics})
            logger.info("seed %d | %-15s %s", seed, kind, metrics)

    aggregate = {
        kind: {
            metric: _mean_sem([r[metric] for r in records if r["latent"] == kind])
            for metric in ("fcd", "valid_and_unique", "novel")
        }
        for kind in kinds
    }
    summary = {"records": records, "aggregate": aggregate}
    _write_json(run_dir, "metrics.json", records)
    _write_json(run_dir, "summary.json", summary)
    return summary


EXPERIMENTS = {
    "sampler_validation": run_sampler_validation,
    "mixture_of_gaussians": run_mixture_of_gaussians,
    "synthetic_datasets": run_synthetic_datasets,
    "qm9": run_qm9,
}
