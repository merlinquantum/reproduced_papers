"""Runtime entrypoints for the Photonic QCNN project."""

from __future__ import annotations

import copy
import json
import logging
import os
import runpy
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "defaults.json"
DATASET_CHOICES = ("BAS", "Custom BAS", "MNIST")
PAPER_SCRIPT_MAP = {
    "BAS": "run_BAS_paper.py",
    "Custom BAS": "run_custom_BAS_paper.py",
    "MNIST": "run_MNIST_paper.py",
}

LOGGER = logging.getLogger(__name__)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: Path | None) -> dict[str, Any]:
    config = read_json(DEFAULT_CONFIG_PATH)
    if config_path:
        if not config_path.exists():  # pragma: no cover - sanity check
            raise FileNotFoundError(f"Config file '{config_path}' does not exist.")
        user_cfg = read_json(config_path)
        config = deep_update(config, user_cfg)
    return config


def _resolve_outdir(config: dict[str, Any]) -> str:
    outdir = config.get("outdir", "results")
    if not os.path.isabs(outdir):
        return str(PROJECT_ROOT / outdir)
    return outdir


def _prepare_dataset_config(
    base_config: dict[str, Any], dataset: str, overrides: dict[str, Any]
) -> dict[str, Any]:
    cfg = copy.deepcopy(overrides)
    cfg.setdefault("outdir", _resolve_outdir(base_config))
    cfg.setdefault("seed", base_config.get("seed", 42))
    cfg.setdefault("device", base_config.get("device", "cpu"))
    return cfg


def _get_merlin_runners() -> dict[str, Any]:
    from photonic_QCNN.lib.run_BAS import run_bas_experiments
    from photonic_QCNN.lib.run_custom_BAS import run_custom_bas_experiments
    from photonic_QCNN.lib.run_MNIST import run_mnist_experiments

    return {
        "BAS": run_bas_experiments,
        "Custom BAS": run_custom_bas_experiments,
        "MNIST": run_mnist_experiments,
    }


def run_merlin_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    runners = _get_merlin_runners()
    datasets = config.get("datasets", list(DATASET_CHOICES))
    results: dict[str, Any] = {}
    training_params = config.get("training") or None
    runs_cfg = config.get("runs", {})

    for dataset in datasets:
        if dataset not in runners:
            raise ValueError(f"Unknown dataset '{dataset}' for MerLin pipeline")
        runner = runners[dataset]
        dataset_overrides = runs_cfg.get(dataset, {})
        cfg = _prepare_dataset_config(config, dataset, dataset_overrides)
        print("=" * 60)
        print(f"Running {dataset} with MerLin implementation")
        print("=" * 60)
        try:
            results[dataset] = runner(cfg, training_params)
        except Exception as exc:  # pragma: no cover - surface failure info
            results[dataset] = {"status": "failed", "error": str(exc)}
            print(f"MerLin run for {dataset} failed: {exc}")
            LOGGER.exception("MerLin run for %s failed", dataset)
    return results


def run_paper_pipeline(
    datasets: list[str], *, results_dir: Path, models_dir: Path
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    lib_dir = PROJECT_ROOT / "lib"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PHOTONIC_QCNN_RESULTS_DIR"] = str(results_dir)
    os.environ["PHOTONIC_QCNN_MODELS_DIR"] = str(models_dir)

    original_cwd = Path.cwd()
    try:
        os.chdir(models_dir)
        for dataset in datasets:
            script_name = PAPER_SCRIPT_MAP[dataset]
            script_path = lib_dir / script_name
            print("=" * 60)
            print(f"Running {dataset} with the paper implementation")
            print("=" * 60)
            try:
                runpy.run_path(str(script_path), run_name="__main__")
                results[dataset] = {"status": "success", "script": str(script_path)}
            except Exception as exc:  # pragma: no cover - bubble up info
                results[dataset] = {"status": "failed", "error": str(exc)}
                print(f"Paper run for {dataset} failed: {exc}")
                LOGGER.exception("Paper run for %s failed", dataset)
    finally:
        os.chdir(original_cwd)
    return results


def _prepare_figure4_dir(outdir: str | None) -> Path:
    base = Path(outdir) if outdir else PROJECT_ROOT / "results" / "figure4"
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target = base / timestamp
    target.mkdir(parents=True, exist_ok=True)
    return target


def _prepare_figure12_dir(outdir: str | None) -> Path:
    base = Path(outdir) if outdir else PROJECT_ROOT / "results" / "figure12"
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target = base / timestamp
    target.mkdir(parents=True, exist_ok=True)
    return target


def run_readout_pipeline(
    outdir: str | None = None,
    *,
    max_iter: int | None = None,
    num_runs: int | None = None,
) -> dict[str, Any]:
    from photonic_QCNN.lib.run_modes_pair_readout import run_modes_pair_readout
    from photonic_QCNN.lib.run_two_fold_readout import run_two_fold_readout
    from photonic_QCNN.utils import readout_visu

    base_dir = _prepare_figure4_dir(outdir)
    print("=" * 60)
    print("Running Figure 4 readout analysis")
    if max_iter is not None:
        print(
            f"Two-fold readout will be truncated after {max_iter} iterations "
            "(results may be incomplete)."
        )
    print("=" * 60)
    time.sleep(2)

    two_fold_dir = base_dir / "two_fold"
    two_fold_results: dict[int, dict[str, Any]] = {}
    for k in (7, 8):
        print(f"[Figure4] Launching two-fold readout strategy for k={k}")
        time.sleep(2)
        result = run_two_fold_readout(k, two_fold_dir / f"k_{k}", max_iter=max_iter)
        if result.get("incomplete"):
            print(
                f"[Figure4] WARNING: Two-fold readout for k={k} completed only "
                f"{result.get('experiments_run', 'N/A')} / {result.get('total_label_sets', 'N/A')} experiments."
            )
        two_fold_results[k] = result

    modes_dir = base_dir / "modes_pair"
    print("\n[Figure4] Launching modes-pair readout strategy")
    time.sleep(2)
    modes_result = run_modes_pair_readout(
        modes_dir, num_runs=num_runs if num_runs is not None else 10
    )

    figures_dir = base_dir / "figures"
    first_fig_paths = readout_visu.readout_visu_first(
        two_fold_results[7]["results_json"],
        two_fold_results[8]["results_json"],
        figures_dir,
    )
    second_fig_path = readout_visu.readout_visu_second(
        modes_result["results_json"],
        figures_dir,
    )

    summary = {
        "base_dir": base_dir,
        "k7": two_fold_results[7],
        "k8": two_fold_results[8],
        "modes_pair": modes_result,
        "figures": {
            "first": [str(path) for path in first_fig_paths],
            "second": str(second_fig_path),
        },
    }
    print("Figure 4 analysis complete. Outputs stored in:", base_dir)
    return summary


def run_simulation_pipeline(
    outdir: str | None = None, *, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    from photonic_QCNN.utils import simulation_visu

    base_dir = _prepare_figure12_dir(outdir)
    runs_dir = base_dir / "runs"
    figures_dir = base_dir / "figures"
    runs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = load_config(None)
    else:
        config = copy.deepcopy(config)
    config["implementation"] = "merlin"
    config["outdir"] = str(runs_dir)
    datasets = config.get("datasets", list(DATASET_CHOICES))
    config["datasets"] = datasets

    print("=" * 60)
    print("Running Figure 12 simulation pipeline")
    print(f"Artifacts root: {base_dir}")
    print("=" * 60)

    merlin_results = run_merlin_pipeline(config)

    figure_paths: dict[str, str] = {}
    for dataset in datasets:
        info = merlin_results.get(dataset)
        output_dir = Path(info.get("output_dir", "")) if info else None
        if not output_dir or not output_dir.exists():
            print(f"[Figure12] WARNING: Missing output directory for {dataset}.")
            continue
        detailed_path = output_dir / "detailed_results.json"
        if not detailed_path.exists():
            print(
                f"[Figure12] WARNING: detailed_results.json not found for {dataset} at {detailed_path}."
            )
            continue
        slug = dataset.lower().replace(" ", "_")
        target_path = figures_dir / f"{slug}_simulation_results.png"
        try:
            figure_path = simulation_visu.generate_simulation_plot(
                detailed_path, target_path
            )
            figure_paths[dataset] = str(figure_path)
        except Exception as exc:  # pragma: no cover - surface failure info
            print(f"[Figure12] Failed to plot {dataset}: {exc}")
            LOGGER.exception("Figure12 plot for %s failed", dataset)

    print("Figure 12 simulation pipeline complete. Outputs stored in:", base_dir)
    return {
        "base_dir": base_dir,
        "runs_dir": runs_dir,
        "figures": figure_paths,
        "results": merlin_results,
    }


def format_merlin_summary(dataset: str, info: dict[str, Any]) -> str:
    if info.get("status") == "failed":
        return f"X {dataset}: FAILED ({info.get('error', 'see logs')})"

    summary = info.get("summary", {})
    test_mean = summary.get("test_acc_mean")
    test_std = summary.get("test_acc_std")
    if test_mean is None:
        return f"✓ {dataset}: SUCCESS"
    return (
        f"✓ {dataset}: SUCCESS - Test Accuracy {test_mean:.4f}"
        + (f" ± {test_std:.4f}" if test_std is not None else "")
        + f" (outputs in {info.get('output_dir')})"
    )


def format_paper_summary(dataset: str, info: dict[str, Any]) -> str:
    status = info.get("status")
    if status == "success":
        return f"✓ {dataset}: SUCCESS (script {info.get('script')})"
    return f"X {dataset}: FAILED ({info.get('error', 'see logs')})"


def print_summary(results: dict[str, Any], implementation: str) -> None:
    print("\n" + "=" * 60)
    print(f"EXPERIMENT SUMMARY - {implementation.upper()} IMPLEMENTATION")
    print("=" * 60)

    for dataset in results:
        info = results[dataset]
        if info is None:
            print(f"X {dataset}: FAILED")
        elif implementation == "merlin":
            print(format_merlin_summary(dataset, info))
        else:
            print(format_paper_summary(dataset, info))

    success_count = sum(
        1
        for info in results.values()
        if info and info.get("status", "success") == "success"
    )
    total_count = len(results)
    print(
        f"\nOverall: {success_count}/{total_count} experiments completed successfully"
    )


def _stringify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _stringify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    if cfg.get("figure4") or cfg.get("figure12"):
        raise ValueError(
            "Figure 4/12 pipelines are now launched via utils/figure4.py or "
            "utils/figure12.py (remove figure4/figure12 from configs)."
        )

    config = copy.deepcopy(cfg)
    config.setdefault("datasets", list(DATASET_CHOICES))
    config.setdefault("outdir", str(run_dir))
    config.setdefault("device", "cpu")
    config.setdefault("seed", 42)
    config["outdir"] = str(run_dir)

    summary_path = run_dir / "summary.json"
    implementation = config.get("implementation", "merlin")
    if implementation not in {"merlin", "paper"}:
        raise ValueError(
            f"Unsupported implementation '{implementation}' (use 'merlin' or 'paper')"
        )

    if implementation == "merlin":
        results = run_merlin_pipeline(config)
    else:
        datasets = config.get("datasets", list(DATASET_CHOICES))
        models_dir = PROJECT_ROOT / "models" / run_dir.name
        results = run_paper_pipeline(
            datasets, results_dir=run_dir, models_dir=models_dir
        )

    print_summary(results, implementation)
    summary_path.write_text(json.dumps(_stringify(results), indent=2))


__all__ = [
    "DATASET_CHOICES",
    "train_and_evaluate",
    "run_merlin_pipeline",
    "run_paper_pipeline",
    "run_readout_pipeline",
    "run_simulation_pipeline",
    "print_summary",
]
