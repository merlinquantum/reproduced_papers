"""Shared-runtime entry point and offline environment checks."""

import importlib.metadata
import json
import logging
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import merlin as ml
import perceval as pcvl
import torch

from runtime_lib.dtypes import dtype_torch


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Run validation or an explicitly configured finite-shot campaign.

    Parameters
    ----------
    cfg : dict
        Configuration resolved by the shared runtime.
    run_dir : pathlib.Path
        Runtime-created output directory.

    Raises
    ------
    ValueError
        If the task is unknown or required training settings are absent.
    """
    task = cfg["task"]
    if task == "plot_results":
        from .results import run_result_plotting

        run_result_plotting(cfg, run_dir)
        return
    if task == "prepare_data":
        from .data import prepare_data

        prepare_data(cfg, run_dir)
        return
    if task == "sampler_validation":
        from .sampler_validation import run_sampler_validation

        run_sampler_validation(cfg, run_dir)
        return
    if task == "native_sampling_check":
        from .validation import check_native_sampling

        check_native_sampling(cfg, run_dir)
        return
    if task == "train":
        if cfg.get("training", {}).get("resumable", False):
            from .resumable import run_resumable_training

            run_resumable_training(cfg, run_dir)
            return
        from .training import run_training

        run_training(cfg, run_dir)
        return
    if task == "phase1":
        from .validation import run_phase_one

        run_phase_one(cfg, run_dir)
        return
    if task != "check_environment":
        raise ValueError(f"Unknown task: {task}")
    report = check_environment(cfg)
    (run_dir / "environment.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logging.getLogger(__name__).info(
        "Phase-zero checks passed; no paper training performed"
    )


def check_environment(cfg: dict) -> dict:
    """Check imports, dependencies, probability agreement and differentiation.

    Parameters
    ----------
    cfg : dict
        Runtime-resolved dtype, device and check tolerance.

    Returns
    -------
    dict
        JSON-serializable environment and numerical evidence.

    Raises
    ------
    RuntimeError
        If a numerical check or dependency check fails.
    ValueError
        If precision or tolerance is unsupported.
    """
    dtype = dtype_torch(cfg["dtype"])
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Environment check requires float32 or float64")
    tolerance = cfg["checks"]["atol"]
    if not 0 < tolerance < 0.01:
        raise ValueError("checks.atol must be in (0, 0.01)")
    dependency_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    )
    if dependency_check.returncode:
        raise RuntimeError(dependency_check.stdout + dependency_check.stderr)

    theta = 0.7
    circuit = pcvl.Circuit(2).add(0, pcvl.BS(theta=pcvl.P("theta")))
    layer = ml.QuantumLayer(
        input_size=0,
        circuit=circuit,
        input_state=[1, 0],
        trainable_parameters=["theta"],
        measurement_strategy=ml.MeasurementStrategy.probs(),
        dtype=dtype,
        device=torch.device(cfg["device"]),
    )
    parameters = list(layer.parameters())
    if sum(p.numel() for p in parameters) != 1:
        raise RuntimeError("Expected exactly one trainable beam-splitter angle")
    with torch.no_grad():
        parameters[0].fill_(theta)
    probabilities = layer().reshape(-1)
    keys = [tuple(int(v) for v in key) for key in layer.output_keys]
    reference = pcvl.Processor("SLOS", pcvl.Circuit(2).add(0, pcvl.BS(theta=theta)))
    reference.with_input(pcvl.BasicState([1, 0]))
    reference_distribution = pcvl.algorithm.Sampler(reference).probs()["results"]
    reference_map = {
        tuple(int(v) for v in key): float(value)
        for key, value in reference_distribution.items()
    }
    if set(keys) != set(reference_map):
        raise RuntimeError("Backend output states differ")
    expected = torch.tensor(
        [reference_map[key] for key in keys], dtype=dtype, device=probabilities.device
    )
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise RuntimeError("Invalid probabilities")
    if not torch.allclose(
        probabilities.sum(), probabilities.new_tensor(1), atol=tolerance, rtol=0
    ):
        raise RuntimeError("Probabilities are not normalized")
    error = float((probabilities.detach() - expected).abs().max())
    if error > tolerance:
        raise RuntimeError(f"MerLin/Perceval mismatch: {error}")

    # A single outcome has a nonzero derivative; the probability sum does not.
    outcome = keys.index((1, 0))
    probabilities[outcome].backward()
    gradient = parameters[0].grad
    if (
        gradient is None
        or not torch.isfinite(gradient).all()
        or gradient.abs().max() < 1e-6
    ):
        raise RuntimeError("Missing, invalid or zero gradient")
    analytic_gradient = -0.5 * torch.sin(torch.tensor(theta, dtype=dtype)).item()
    gradient_error = abs(gradient.item() - analytic_gradient)
    if gradient_error > tolerance:
        raise RuntimeError(f"Gradient mismatch: {gradient_error}")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        requested_sampling = layer(shots=100).detach().reshape(-1)
    sampling_messages = [str(item.message) for item in captured]
    if not any(
        "Sampling was requested but is disabled" in msg for msg in sampling_messages
    ):
        raise RuntimeError("Expected sampling-disabled warning during autograd")
    if not torch.allclose(
        requested_sampling, probabilities.detach(), atol=tolerance, rtol=0
    ):
        raise RuntimeError("Autograd sampling behavior changed")

    packages = [
        "merlinquantum",
        "perceval-quandela",
        "torch",
        "numpy",
        "scipy",
        "pytest",
    ]
    return {
        "status": "passed",
        "scope": "phase_zero_only",
        "training_implemented": True,
        "training_exercised": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "module_versions": {"merlin": ml.__version__, "perceval": pcvl.__version__},
        "pip_check": dependency_check.stdout.strip(),
        "device_tested": cfg["device"],
        "dtype": str(dtype),
        "seed": cfg["seed"],
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "xpu_available": torch.xpu.is_available(),
        "cuda_build": torch.version.cuda,
        "circuit": {"modes": 2, "input": [1, 0], "theta": theta},
        "states": keys,
        "probabilities": probabilities.detach().cpu().tolist(),
        "perceval_probabilities": expected.cpu().tolist(),
        "normalization": probabilities.detach().sum().item(),
        "max_probability_error": error,
        "gradient": gradient.item(),
        "analytic_gradient": analytic_gradient,
        "gradient_error": gradient_error,
        "sampling_disabled_with_grad": True,
        "sampling_warnings": sampling_messages,
        "hub_queried": False,
        "qpu_jobs_submitted": 0,
    }
