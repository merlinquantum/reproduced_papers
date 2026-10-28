"""Runtime entry point: train and evaluate one photonic QCBM (or its B5 twin).

One run = one (dataset, ansatz, kernel, init, seed) configuration:

1. build or load the dataset (``lib.data``);
2. initialize parameters (``lib.initialization``);
3. train with the paper's MMD estimator (``lib.estimator``, ``lib.training``);
4. evaluate on the test split:
   - ``test_mmd_estimator``: the paper's protocol, Fig. 1 estimator with fresh
     K, Z (indistinguishable photons) or the exact B5 expectation;
   - sample-based unbiased MMD^2 for the trained model and each enabled
     baseline, all with the same kernel operators (``walsh``), plus the exact
     Gaussian kernel on raw occupations and on binarized samples when the
     kernel is Gaussian (D5 collision check);
5. write artifacts.

Artifacts in ``run_dir`` (public contract of this reproduction):
``run.log`` and ``config_snapshot.json`` (shared runtime), ``run_status.json``,
``metrics.json``, ``losses.csv`` (step, train MMD estimate), ``params.npy``,
``unitary.npy``, ``samples.npz`` (model and baseline samples).
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import baselines, circuits, data, estimator, initialization, kernels, training
from .experiment_logging import git_state, sha256, utc_now, write_json

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINES = ("random", "marginal", "rbm", "test_to_test")
# A trained matrix further than this from unitary is not a physical interferometer
# (original butterfly mesh at m not a power of two, D3) and cannot be sampled.
UNITARITY_TOLERANCE = 1e-8


def resolve_input_modes(model_cfg: dict, m: int, n: int, manifest: dict) -> list[int]:
    """Input modes S from ``model.input_state``.

    Named states follow the original code; ``pairs`` puts one photon on the
    first mode of each pair (pairwise datasets), ``blocks`` on the first mode
    of each categorical block (``manifest["block_sizes"]``).
    """
    state = model_cfg["input_state"]
    if state == "pairs":
        return list(range(0, 2 * n, 2))
    if state == "blocks":
        starts = np.cumsum([0] + manifest["block_sizes"][:-1])
        return [int(s) for s in starts]
    return data.input_modes(m, n, state)


def _write_losses(path: Path, losses: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "train_mmd_estimate"])
        writer.writerows((i + 1, value) for i, value in enumerate(losses))


def _sample_metrics(
    samples: dict, X_test: np.ndarray, K: jnp.ndarray, kernel_cfg: dict
) -> dict:
    """Unbiased sample MMD^2 of every sample set against the test split."""
    results = {}
    X_test_j = jnp.asarray(X_test, dtype=jnp.float64)
    for name, X in samples.items():
        X_j = jnp.asarray(X, dtype=jnp.float64)
        entry = {
            "num_samples": int(len(X)),
            "collision_fraction": float((np.asarray(X).max(axis=1) > 1).mean()),
            "walsh_mmd": estimator.walsh_mmd_unbiased(X_j, X_test_j, K),
        }
        if kernel_cfg["type"] == "gaussian":
            sigma = kernel_cfg["sigma"]
            entry["gaussian_mmd_raw"] = estimator.gaussian_mmd_unbiased(
                X, X_test, sigma
            )
            entry["gaussian_mmd_binarized"] = estimator.gaussian_mmd_unbiased(
                np.minimum(X, 1), np.minimum(X_test, 1), sigma
            )
        results[name] = entry
        logger.info(
            "EVALUATION_COMPLETED | split=test | model=%s | walsh_mmd=%.6g%s",
            name,
            entry["walsh_mmd"],
            f" | gaussian_mmd_raw={entry['gaussian_mmd_raw']:.6g}"
            if "gaussian_mmd_raw" in entry
            else "",
        )
    return results


def _train_jax(cfg, m, modes, sampler, X_train, X_test, keys) -> dict:
    """Phase 03 path: JAX estimator, ansätze and Adam (the original reproduction)."""
    model_cfg, train_cfg, eval_cfg = cfg["model"], cfg["training"], cfg["evaluation"]
    key_init, key_train, key_eval_init, key_eval, _ = keys
    unitary_fn = circuits.make_unitary_fn(
        m, model_cfg["ansatz"], model_cfg["butterfly_variant"]
    )
    photon_model = model_cfg["photon_model"]
    train_loss = estimator.make_mmd_loss(
        unitary_fn,
        modes,
        sampler,
        train_cfg["n_operators"],
        train_cfg["n_gurvits"],
        photon_model,
    )
    eval_loss = estimator.make_mmd_loss(
        unitary_fn,
        modes,
        sampler,
        eval_cfg["n_operators"],
        eval_cfg["n_gurvits"],
        photon_model,
    )
    X_train_j = jnp.asarray(X_train, dtype=jnp.float64)
    X_test_j = jnp.asarray(X_test, dtype=jnp.float64)
    params, init_info = initialization.initial_parameters(
        cfg["init"], key_init, m, model_cfg["ansatz"], unitary_fn, modes, X_train_j
    )
    init_test = float(eval_loss(params, key_eval_init, X_test_j))
    logger.info(
        "EVALUATION_COMPLETED | split=test | stage=init | test_mmd_estimator=%.6g | unitarity_error=%.2e",
        init_test,
        circuits.unitarity_error(unitary_fn(params)),
    )
    result = training.train_adam(
        train_loss,
        params,
        X_train_j,
        train_cfg["steps"],
        train_cfg["learning_rate"],
        key_train,
        train_cfg["convergence_interval"],
        train_cfg["log_every"],
    )
    logger.info("EVALUATION_STARTED | split=test | stage=final")
    test_estimate = float(eval_loss(result["params"], key_eval, X_test_j))
    logger.info(
        "EVALUATION_COMPLETED | split=test | stage=final | test_mmd_estimator=%.6g",
        test_estimate,
    )
    return {
        "result": result,
        "U": np.asarray(unitary_fn(result["params"])),
        "params": np.asarray(result["params"]),
        "init_info": init_info,
        "init_test": init_test,
        "test_estimate": test_estimate,
    }


def _train_merlin(cfg, m, modes, sampler, X_train, X_test, keys) -> dict:
    """Phase 04 MerLin adaptation: same estimator, K/Z key chain and Adam in PyTorch on MerLin circuits."""
    import torch

    from . import qcbm_merlin

    model_cfg, train_cfg, eval_cfg = cfg["model"], cfg["training"], cfg["evaluation"]
    key_init, key_train, key_eval_init, key_eval, _ = keys
    device = cfg["device"]
    interferometer = qcbm_merlin.MerlinInterferometer(m, model_cfg["ansatz"], device)
    photon_model = model_cfg["photon_model"]
    train_loss = qcbm_merlin.make_loss(
        interferometer,
        modes,
        sampler,
        train_cfg["n_operators"],
        train_cfg["n_gurvits"],
        photon_model,
    )
    eval_loss = qcbm_merlin.make_loss(
        interferometer,
        modes,
        sampler,
        eval_cfg["n_operators"],
        eval_cfg["n_gurvits"],
        photon_model,
    )
    X_train_t = torch.as_tensor(X_train, dtype=torch.float64, device=device)
    X_test_t = torch.as_tensor(X_test, dtype=torch.float64, device=device)
    unitary_fn_jax = (
        circuits.make_unitary_fn(m, "haar") if model_cfg["ansatz"] == "haar" else None
    )
    params = interferometer.initial_parameters(
        cfg["init"],
        key_init,
        unitary_fn_jax,
        modes,
        jnp.asarray(X_train, dtype=jnp.float64),
    )
    with torch.no_grad():
        init_test = float(eval_loss(params, key_eval_init, X_test_t))
        init_unitarity = circuits.unitarity_error(
            jnp.asarray(interferometer.unitary(params).cpu().numpy())
        )
    logger.info(
        "EVALUATION_COMPLETED | split=test | stage=init | test_mmd_estimator=%.6g | unitarity_error=%.2e | backend=merlin",
        init_test,
        init_unitarity,
    )
    result = qcbm_merlin.train_adam_torch(
        train_loss,
        params,
        X_train_t,
        train_cfg["steps"],
        train_cfg["learning_rate"],
        key_train,
        train_cfg["convergence_interval"],
        train_cfg["log_every"],
    )
    logger.info("EVALUATION_STARTED | split=test | stage=final")
    with torch.no_grad():
        test_estimate = float(eval_loss(result["params"], key_eval, X_test_t))
        U = interferometer.unitary(result["params"]).cpu().numpy()
    logger.info(
        "EVALUATION_COMPLETED | split=test | stage=final | test_mmd_estimator=%.6g",
        test_estimate,
    )
    return {
        "result": result,
        "U": U,
        "params": result["params"].cpu().numpy(),
        "init_info": {},
        "init_test": init_test,
        "test_estimate": test_estimate,
    }


def _noise_study(
    study: dict, U: np.ndarray, modes, n: int, X_test: np.ndarray
) -> list[dict]:
    """Phase 04 M3: exact MMD^2 of the trained U under source/loss noise and detector models (MerLin SLOS)."""
    from . import qcbm_merlin

    rows = []
    for visibility in study["indistinguishability"]:
        for transmittance in study["transmittance"]:
            start = time.perf_counter()
            distribution = qcbm_merlin.noisy_distribution(
                U, modes, visibility, transmittance
            )
            for detector in study["detectors"]:
                kept, rate = qcbm_merlin.detect(distribution, n, detector)
                test = X_test if detector == "pnr" else np.minimum(X_test, 1)
                value = qcbm_merlin.exact_vs_samples_mmd(kept, test, study["sigma"])
                rows.append(
                    {
                        "indistinguishability": visibility,
                        "transmittance": transmittance,
                        "detector": detector,
                        "postselection_rate": rate,
                        "exact_mmd": value,
                    }
                )
                logger.info(
                    "EVALUATION_COMPLETED | split=test | noise V=%.2f T=%.2f detector=%s | exact_mmd=%.6g | postselection=%.4f",
                    visibility,
                    transmittance,
                    detector,
                    value,
                    rate,
                )
            logger.debug(
                "noise point V=%s T=%s took %.1fs",
                visibility,
                transmittance,
                time.perf_counter() - start,
            )
    return rows


def _run_rbm_tuned(cfg, dataset, X_train, X_test, seed, run_dir) -> dict:
    """B6 (Phase 5): tune, refit and evaluate the RBM baseline for every sigma.

    For each sigma the test metrics use the same kernel operators K as the
    QCBM run with this seed (``keys[4]`` of ``PRNGKey(seed)``), so ``walsh_mmd``
    is paired with Fig. 3a.
    """
    from . import rbm_classical

    m, n = cfg["dataset"]["m"], cfg["dataset"]["n"]
    tuning_cfg, eval_cfg = cfg["tuning"], cfg["evaluation"]
    sigmas = [float(s) for s in tuning_cfg["sigmas"]]
    logger.info(
        "TRAINING_STARTED | model=rbm_tuned | grid=hidden%s x lr%s | gibbs=%s | sigmas=%s",
        tuning_cfg["hidden_units"],
        tuning_cfg["learning_rates"],
        tuning_cfg["gibbs_steps"],
        sigmas,
    )
    record, samples = rbm_classical.tune_and_refit(X_train, tuning_cfg, sigmas, seed)
    key_ops = jax.random.split(jax.random.PRNGKey(seed), 5)[4]
    by_sigma = {}
    for sigma in sigmas:
        kernel_cfg = {**cfg["kernel"], "type": "gaussian", "sigma": sigma}
        sampler = kernels.make_operator_sampler(
            kernel_cfg, m, n, mode_marginals=X_train.mean(axis=0)
        )
        K_eval = sampler(key_ops, eval_cfg["n_operators"])
        by_sigma[str(sigma)] = _sample_metrics(
            {"rbm_tuned": samples[f"sigma{sigma}"]}, X_test, K_eval, kernel_cfg
        )
    np.savez_compressed(run_dir / "samples.npz", **samples)
    logger.info("ARTIFACT_WRITTEN | samples.npz")
    return {
        "experiment": cfg.get("experiment", {}),
        "seed": seed,
        "backend": "rbm_tuned",
        "dataset": {
            "id": dataset["manifest"]["dataset_id"],
            "collision_free_train": dataset["manifest"]["collision_free_train"],
            "collision_fraction": dataset["manifest"].get("collision_fraction"),
        },
        "tuning": record,
        "sample_metrics_by_sigma": by_sigma,
    }


def _run_qcbm(cfg, run_dir, dataset, X_train, X_test, seed) -> tuple[dict, float]:
    """Train and evaluate the QCBM or its B5 twin; return (metrics, test estimator)."""
    m, n = cfg["dataset"]["m"], cfg["dataset"]["n"]
    model_cfg, kernel_cfg, eval_cfg = cfg["model"], cfg["kernel"], cfg["evaluation"]
    modes = resolve_input_modes(model_cfg, m, n, dataset["manifest"])
    marginals = X_train.mean(axis=0)
    sampler = kernels.make_operator_sampler(kernel_cfg, m, n, mode_marginals=marginals)
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    backend = model_cfg["backend"]
    if backend == "jax":
        trained = _train_jax(cfg, m, modes, sampler, X_train, X_test, keys)
    elif backend == "merlin":
        trained = _train_merlin(cfg, m, modes, sampler, X_train, X_test, keys)
    else:
        raise ValueError(f"model.backend must be 'jax' or 'merlin', got {backend!r}.")
    result, U, init_info = trained["result"], trained["U"], trained["init_info"]
    init_test, test_estimate = trained["init_test"], trained["test_estimate"]
    key_ops = keys[4]
    photon_model = model_cfg["photon_model"]
    _write_losses(run_dir / "losses.csv", result["losses"])
    np.save(run_dir / "params.npy", trained["params"])
    np.save(run_dir / "unitary.npy", U)
    logger.info("ARTIFACT_WRITTEN | losses.csv, params.npy, unitary.npy")

    metrics = {
        "experiment": cfg.get("experiment", {}),
        "seed": seed,
        "input_modes": modes,
        "num_parameters": int(trained["params"].size),
        "backend": backend,
        "dataset": {
            "id": dataset["manifest"]["dataset_id"],
            "collision_free_train": dataset["manifest"]["collision_free_train"],
            "collision_fraction": dataset["manifest"].get("collision_fraction"),
        },
        "init": {
            "strategy": cfg["init"]["strategy"],
            **init_info,
            "test_mmd_estimator": init_test,
        },
        "training": {
            "steps_run": result["steps_run"],
            "converged": result["converged"],
            "seconds": result["seconds"],
            "seconds_per_step": result["seconds"] / max(result["steps_run"], 1),
            "final_train_mmd_estimate": result["losses"][-1],
        },
        "test_mmd_estimator": test_estimate,
        "unitarity_error": circuits.unitarity_error(jnp.asarray(U)),
    }
    metrics["deployable"] = metrics["unitarity_error"] <= UNITARITY_TOLERANCE

    if eval_cfg["sample_metrics"]:
        num = eval_cfg["num_model_samples"]
        sample_seed = seed + 7919
        samples = {}
        start = time.perf_counter()
        if not metrics["deployable"]:
            # No sample-based model metrics: the trained U is not unitary.
            logger.info(
                "MODEL_NOT_DEPLOYABLE | unitarity_error=%.3g | tolerance=%.0e | model sampling skipped",
                metrics["unitarity_error"],
                UNITARITY_TOLERANCE,
            )
        elif photon_model == "distinguishable":
            samples["model"] = baselines.distinguishable_samples(
                U, modes, num, sample_seed
            )
        elif eval_cfg["sampler"] == "merlin_clifford":
            from . import qcbm_merlin

            samples["model"] = qcbm_merlin.merlin_clifford_samples(
                U, modes, num, sample_seed, cfg["device"]
            )
        elif eval_cfg["sampler"] == "perceval":
            samples["model"] = baselines.boson_samples(U, modes, num, sample_seed)
        else:
            raise ValueError(
                f"evaluation.sampler must be 'perceval' or 'merlin_clifford', got {eval_cfg['sampler']!r}."
            )
        if "model" in samples:
            logger.info(
                "MODEL_SAMPLED | samples=%d | seconds=%.1f | sampler=%s",
                num,
                time.perf_counter() - start,
                eval_cfg["sampler"]
                if photon_model == "indistinguishable"
                else "distinguishable",
            )
        for name in eval_cfg["baselines"]:
            if name not in BASELINES:
                raise ValueError(
                    f"Unknown baseline {name!r}; expected one of {BASELINES}."
                )
            if name == "random":
                samples["random"] = baselines.uniform_fixed_weight(
                    m, n, num, sample_seed
                )
            elif name == "marginal":
                samples["marginal"] = baselines.independent_marginal_fixed_weight(
                    X_train, num, sample_seed
                )
            elif name == "rbm":
                start = time.perf_counter()
                rbm = baselines.train_rbm(X_train, eval_cfg["rbm"], seed)
                samples["rbm"] = baselines.rbm_samples(
                    rbm, num, eval_cfg["rbm"]["gibbs_steps"]
                )
                logger.info(
                    "BASELINE_TRAINED | model=rbm | seconds=%.1f",
                    time.perf_counter() - start,
                )
        K_eval = sampler(key_ops, eval_cfg["n_operators"])
        metrics["sample_metrics"] = _sample_metrics(samples, X_test, K_eval, kernel_cfg)
        if "test_to_test" in eval_cfg["baselines"]:
            half_a, half_b = baselines.test_halves(X_test, sample_seed)
            entry = {
                "walsh_mmd": estimator.walsh_mmd_unbiased(
                    jnp.asarray(half_a, dtype=jnp.float64),
                    jnp.asarray(half_b, dtype=jnp.float64),
                    K_eval,
                )
            }
            if kernel_cfg["type"] == "gaussian":
                entry["gaussian_mmd_raw"] = estimator.gaussian_mmd_unbiased(
                    half_a, half_b, kernel_cfg["sigma"]
                )
            metrics["sample_metrics"]["test_to_test"] = entry
            logger.info(
                "EVALUATION_COMPLETED | split=test | model=test_to_test | walsh_mmd=%.6g",
                entry["walsh_mmd"],
            )
        np.savez_compressed(run_dir / "samples.npz", **samples)
        logger.info("ARTIFACT_WRITTEN | samples.npz")

    if eval_cfg["noise_study"] is not None:
        metrics["noise_study"] = _noise_study(
            eval_cfg["noise_study"], U, modes, n, X_test
        )
    return metrics, test_estimate


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Run one configured experiment and write its artifacts to ``run_dir``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration after defaults, overlays, and CLI flags. Required
        sections: ``dataset``, ``model``, ``kernel``, ``init``, ``training``,
        ``evaluation``; plus ``seed``, ``data_root``, ``device``.
    run_dir : pathlib.Path
        Timestamped output directory created by the shared runtime.

    Raises
    ------
    KeyError
        If a required config key is missing.
    """
    run_dir = Path(run_dir)
    run_id = run_dir.name
    seed = int(cfg["seed"])
    status = {
        "schema_version": 1,
        "run_id": run_id,
        "sweep_id": cfg.get("sweep_id"),
        "candidate": cfg.get("candidate", {}),
        "seed": seed,
        "repetition": cfg.get("repetition", 0),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "dataset": {
            "name": cfg["dataset"]["name"],
            "splits": ["train", "test"],
            "preprocessing": data.recipe(cfg["dataset"]),
            "subset": None,
        },
        "selection": {
            "name": "test_mmd_estimator (reported, no selection within a single run)",
            "direction": "minimize",
            "split": "test",
            "checkpoint": "final",
        },
        "log_path": "run.log",
        "config_path": "config_snapshot.json",
        "config_sha256": sha256(run_dir / "config_snapshot.json")
        if (run_dir / "config_snapshot.json").exists()
        else None,
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "RUN_STARTED | run_id=%s | seed=%d | experiment=%s",
        run_id,
        seed,
        cfg.get("experiment", {}).get("name"),
    )
    try:
        dataset = data.load_dataset(cfg)
        X_train, X_test = dataset["train"], dataset["test"]
        status["dataset"]["subset"] = {
            "path": dataset["path"],
            "train_shape": list(X_train.shape),
            "test_shape": list(X_test.shape),
            "train_sha256": dataset["manifest"]["train_sha256"],
            "test_sha256": dataset["manifest"]["test_sha256"],
        }
        write_json(run_dir / "run_status.json", status)
        logger.info(
            "DATASET_READY | name=%s | train=%s | test=%s | collision_free_train=%s",
            cfg["dataset"]["name"],
            X_train.shape,
            X_test.shape,
            dataset["manifest"]["collision_free_train"],
        )

        if cfg["model"]["backend"] == "rbm_tuned":
            metrics = _run_rbm_tuned(cfg, dataset, X_train, X_test, seed, run_dir)
            headline = "gaussian_mmd_binarized by sigma: " + ", ".join(
                f"{s}={e['rbm_tuned']['gaussian_mmd_binarized']:.6g}"
                for s, e in metrics["sample_metrics_by_sigma"].items()
            )
        else:
            metrics, test_estimate = _run_qcbm(
                cfg, run_dir, dataset, X_train, X_test, seed
            )
            headline = f"test_mmd_estimator={test_estimate:.6g}"
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, default=float), encoding="utf-8"
        )
        status["metrics_sha256"] = sha256(metrics_path)
        logger.info("METRICS_WRITTEN | path=metrics.json | %s", headline)
        status.update(status="COMPLETED", completed_at=utc_now())
        write_json(run_dir / "run_status.json", status)
        logger.info("RUN_COMPLETED | run_id=%s", run_id)
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(run_dir / "run_status.json", status)
        logger.exception("RUN_FAILED | run_id=%s", run_id)
        raise
