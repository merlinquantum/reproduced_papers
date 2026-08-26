"""Five-stage RF-RQKS ablation protocol."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .ablation_data import DatasetSplits
from .qks import build_sampler

READOUT_NAMES = (
    "linear_svm",
    "linear_logistic",
    "nystroem_logistic",
    "random_fourier_linear_svm",
)


@dataclass(frozen=True)
class ModelConfiguration:
    """One QRKS circuit configuration.

    Parameters
    ----------
    photon_count : int
        Number of photons.
    mode_count : int
        Number of optical modes.
    depth : int
        Circuit depth.
    episode_count : int
        Number of random episodes.
    encoding_strategy : str
        Encoding layer strategy.
    entangling_strategy : str | None
        Entangling strategy. If omitted, no entangler is used.
    same_haar : bool
        Whether V1 layers reuse the same Haar unitary.
    """

    photon_count: int
    mode_count: int
    depth: int
    episode_count: int
    encoding_strategy: str
    entangling_strategy: str | None
    same_haar: bool

    @property
    def nominal_output_feature_count(self) -> int:
        """Return the ungrouped episode-output dimension.

        Returns
        -------
        int
            ``E * comb(m, n)``.
        """
        return self.episode_count * math.comb(self.mode_count, self.photon_count)


@dataclass(frozen=True)
class RuntimeConfiguration:
    """Runtime values shared by all ablation stages.

    Parameters
    ----------
    sampler : str
        Sampler backend name.
    device : str
        Torch feature-extraction device.
    seed : int
        Base random seed.
    batch_size : int
        Feature-extraction batch size.
    readout_c : float
        Linear readout regularization parameter.
    readout_max_iter : int
        Maximum linear readout iterations.
    scale_sampled_features : bool
        Whether to standardize sampled features before the readout.
    kernel_components : int
        Approximate-kernel feature count used during Stage 5.
    kernel_gamma : float
        RBF gamma used during Stage 5.
    maximum_output_features : int
        Safety limit for a single sampler configuration.
    """

    sampler: str
    device: str
    seed: int
    batch_size: int
    readout_c: float
    readout_max_iter: int
    scale_sampled_features: bool
    kernel_components: int
    kernel_gamma: float
    maximum_output_features: int


def _configuration_seed(base_seed: int, configuration: ModelConfiguration) -> int:
    payload = json.dumps(asdict(configuration), sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**31)


def _extract_features(
    model: torch.nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    sampled_batches = []
    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            sampled_batches.append(model(batch).detach().cpu().numpy())
    return np.concatenate(sampled_batches, axis=0)


def _build_model(
    configuration: ModelConfiguration,
    runtime: RuntimeConfiguration,
    input_feature_count: int,
) -> torch.nn.Module:
    if configuration.nominal_output_feature_count > runtime.maximum_output_features:
        raise ValueError(
            "Configuration exceeds maximum_output_features: "
            f"E * comb(m, n) = {configuration.nominal_output_feature_count} > "
            f"{runtime.maximum_output_features}"
        )
    seed = _configuration_seed(runtime.seed, configuration)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_sampler(
        sampler_name=runtime.sampler,
        photon_count=configuration.photon_count,
        mode_count=configuration.mode_count,
        depth=configuration.depth,
        episode_count=configuration.episode_count,
        input_feature_count=input_feature_count,
        encoding_strategy=configuration.encoding_strategy,
        entangling_strategy=configuration.entangling_strategy,
        same_haar=configuration.same_haar,
    )
    return model.to(runtime.device)


def _selection_metrics(
    configuration: ModelConfiguration,
    dataset: DatasetSplits,
    runtime: RuntimeConfiguration,
) -> dict[str, Any]:
    model = _build_model(configuration, runtime, dataset.input_feature_count)
    sampled_train = _extract_features(
        model, dataset.train_features, runtime.batch_size, runtime.device
    )
    sampled_validation = _extract_features(
        model, dataset.validation_features, runtime.batch_size, runtime.device
    )
    steps: list[tuple[str, Any]] = []
    if runtime.scale_sampled_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "classifier",
            LinearSVC(
                C=runtime.readout_c,
                max_iter=runtime.readout_max_iter,
                random_state=_configuration_seed(runtime.seed, configuration),
                dual="auto",
            ),
        )
    )
    readout = Pipeline(steps)
    readout.fit(sampled_train, dataset.train_labels)
    train_scores = readout.decision_function(sampled_train)
    validation_scores = readout.decision_function(sampled_validation)
    return {
        "seed": _configuration_seed(runtime.seed, configuration),
        "nominal_output_feature_count": configuration.nominal_output_feature_count,
        "actual_output_feature_count": int(sampled_train.shape[1]),
        "metrics": {
            "train_auroc": float(roc_auc_score(dataset.train_labels, train_scores)),
            "train_f1": float(f1_score(dataset.train_labels, readout.predict(sampled_train))),
            "validation_auroc": float(
                roc_auc_score(dataset.validation_labels, validation_scores)
            ),
            "validation_f1": float(
                f1_score(dataset.validation_labels, readout.predict(sampled_validation))
            ),
        },
    }


def _best(results: list[dict[str, Any]], count: int = 1) -> list[dict[str, Any]]:
    return sorted(
        results,
        key=lambda result: (
            result["metrics"]["validation_auroc"],
            result["metrics"]["validation_f1"],
        ),
        reverse=True,
    )[:count]


def _write_state(state: dict[str, Any], run_dir: Path) -> None:
    output_path = run_dir / "results.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)


def _run_configuration(
    stage_name: str,
    configuration: ModelConfiguration,
    dataset: DatasetSplits,
    runtime: RuntimeConfiguration,
    state: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    print(f"  [{stage_name}] Testing config: m={configuration.mode_count}, "
          f"n={configuration.photon_count}, D={configuration.depth}, "
          f"E={configuration.episode_count}, ent={configuration.entangling_strategy}")
    result = {
        "configuration": asdict(configuration),
        **_selection_metrics(configuration, dataset, runtime),
    }
    print(f"  [{stage_name}] Result - validation_auroc: {result['metrics']['validation_auroc']:.4f}, "
          f"validation_f1: {result['metrics']['validation_f1']:.4f}")
    state["stages"][stage_name]["results"].append(result)
    _write_state(state, run_dir)
    return result


def _configuration_from_result(result: dict[str, Any]) -> ModelConfiguration:
    return ModelConfiguration(**result["configuration"])


def _plot_stage_1_heatmap(
    results: list[dict[str, Any]],
    entangling_enabled: bool,
    mode_counts: list[int],
    episode_counts: list[int],
    output_path: Path,
) -> None:
    score_grid = np.full((len(mode_counts), len(episode_counts)), np.nan)
    mode_indices = {value: index for index, value in enumerate(mode_counts)}
    episode_indices = {value: index for index, value in enumerate(episode_counts)}
    for result in results:
        configuration = result["configuration"]
        if (configuration["entangling_strategy"] is not None) != entangling_enabled:
            continue
        mode_index = mode_indices.get(configuration["mode_count"])
        episode_index = episode_indices.get(configuration["episode_count"])
        if mode_index is not None and episode_index is not None:
            score_grid[mode_index, episode_index] = result["metrics"][
                "validation_auroc"
            ]

    figure, axis = plt.subplots(figsize=(8, 5), dpi=150)
    color_map = plt.get_cmap("viridis").copy()
    color_map.set_bad(color="white")
    image = axis.imshow(score_grid, aspect="auto", cmap=color_map, vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(episode_counts)), episode_counts)
    axis.set_yticks(range(len(mode_counts)), mode_counts)
    axis.set_xlabel("Episodes (E)")
    axis.set_ylabel("Modes / qubit count (m)")
    label = "with entangling" if entangling_enabled else "without entangling"
    axis.set_title(f"Stage 1 validation AUROC {label}")
    for mode_index in range(len(mode_counts)):
        for episode_index in range(len(episode_counts)):
            score = score_grid[mode_index, episode_index]
            axis.text(
                episode_index,
                mode_index,
                "None" if np.isnan(score) else f"{score:.3f}",
                ha="center",
                va="center",
                color="black" if np.isnan(score) or score >= 0.65 else "white",
                fontsize=8,
            )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Validation AUROC")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _plot_stage_2_curves(results: list[dict[str, Any]], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), dpi=150)
    groups: dict[tuple[int, int, str | None], list[dict[str, Any]]] = {}
    for result in results:
        configuration = result["configuration"]
        group_key = (
            configuration["mode_count"],
            configuration["episode_count"],
            configuration["entangling_strategy"],
        )
        groups.setdefault(group_key, []).append(result)
    for (mode_count, episode_count, entangling_strategy), group in groups.items():
        group.sort(key=lambda result: result["configuration"]["depth"])
        entangling_label = "on" if entangling_strategy is not None else "off"
        axis.plot(
            [result["configuration"]["depth"] for result in group],
            [result["metrics"]["validation_auroc"] for result in group],
            marker="o",
            linewidth=1.8,
            label=f"m={mode_count}, E={episode_count}, ent={entangling_label}",
        )
    axis.set_xlabel("Depth (D)")
    axis.set_ylabel("Validation AUROC")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Stage 2 validation AUROC by depth")
    axis.grid(alpha=0.25)
    if groups:
        axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _plot_stage_3_bars(results: list[dict[str, Any]], output_path: Path) -> None:
    ordered_results = sorted(
        results,
        key=lambda result: (
            result["configuration"]["depth"],
            result["configuration"]["episode_count"],
        ),
    )
    labels = [
        f"({result['configuration']['depth']}, {result['configuration']['episode_count']})"
        for result in ordered_results
    ]
    positions = np.arange(len(ordered_results))
    bar_width = 0.36
    figure, axis = plt.subplots(figsize=(8, 5), dpi=150)
    axis.bar(
        positions - bar_width / 2,
        [result["metrics"]["validation_auroc"] for result in ordered_results],
        width=bar_width,
        color="#1f77b4",
        label="AUROC",
    )
    axis.bar(
        positions + bar_width / 2,
        [result["metrics"]["validation_f1"] for result in ordered_results],
        width=bar_width,
        color="#ff7f0e",
        label="F1",
    )
    axis.set_xticks(positions, labels)
    axis.set_xlabel("(D, E)")
    axis.set_ylabel("Validation score")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Stage 3 validation score by matched (D, E)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _plot_stage_4_lines(results: list[dict[str, Any]], output_path: Path) -> None:
    ordered_results = sorted(
        results, key=lambda result: result["configuration"]["photon_count"]
    )
    photon_counts = [
        result["configuration"]["photon_count"] for result in ordered_results
    ]
    figure, axis = plt.subplots(figsize=(8, 5), dpi=150)
    axis.plot(
        photon_counts,
        [result["metrics"]["validation_auroc"] for result in ordered_results],
        marker="o",
        linewidth=1.8,
        color="#1f77b4",
        label="AUROC",
    )
    axis.plot(
        photon_counts,
        [result["metrics"]["validation_f1"] for result in ordered_results],
        marker="o",
        linewidth=1.8,
        color="#ff7f0e",
        label="F1",
    )
    axis.set_xlabel("Photon count (n)")
    axis.set_ylabel("Validation score")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Stage 4 validation score by photon count")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _plot_stage_5_readouts(result: dict[str, Any], output_path: Path) -> None:
    readout_names = list(READOUT_NAMES)
    direct_values = [result["direct_readouts"][name]["test_auroc"] for name in readout_names]
    qks_values = [result["qks_readouts"][name]["test_auroc"] for name in readout_names]
    positions = np.arange(len(readout_names))
    figure, axis = plt.subplots(figsize=(10, 4.8), dpi=150)
    bar_height = 0.35
    axis.barh(positions - bar_height / 2, direct_values, height=bar_height, color="#7f7f7f", label="Direct")
    axis.barh(positions + bar_height / 2, qks_values, height=bar_height, color="#0e89e6", label="Photonic")
    axis.set_yticks(positions, readout_names)
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("TEST AUROC")
    axis.set_title("Stage 5 TEST AUROC")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _readout_pipeline(name: str, runtime: RuntimeConfiguration) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if name == "linear_svm":
        estimator = LinearSVC(
            C=runtime.readout_c,
            max_iter=runtime.readout_max_iter,
            random_state=runtime.seed,
            dual="auto",
        )
    elif name == "linear_logistic":
        estimator = LogisticRegression(
            C=runtime.readout_c,
            max_iter=runtime.readout_max_iter,
            random_state=runtime.seed,
        )
    elif name == "nystroem_logistic":
        steps.append(
            (
                "kernel",
                Nystroem(
                    gamma=runtime.kernel_gamma,
                    n_components=runtime.kernel_components,
                    random_state=runtime.seed,
                ),
            )
        )
        estimator = LogisticRegression(
            C=runtime.readout_c,
            max_iter=runtime.readout_max_iter,
            random_state=runtime.seed,
        )
    elif name == "random_fourier_linear_svm":
        steps.append(
            (
                "kernel",
                RBFSampler(
                    gamma=runtime.kernel_gamma,
                    n_components=runtime.kernel_components,
                    random_state=runtime.seed,
                ),
            )
        )
        estimator = LinearSVC(
            C=runtime.readout_c,
            max_iter=runtime.readout_max_iter,
            random_state=runtime.seed,
            dual="auto",
        )
    else:
        raise ValueError(f"Unsupported readout: {name}")
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def _test_readouts(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    runtime: RuntimeConfiguration,
) -> dict[str, dict[str, float]]:
    results = {}
    for name in READOUT_NAMES:
        pipeline = _readout_pipeline(name, runtime)
        pipeline.fit(train_features, train_labels)
        if hasattr(pipeline, "decision_function"):
            scores = pipeline.decision_function(test_features)
        else:
            scores = pipeline.predict_proba(test_features)[:, 1]
        results[name] = {
            "test_auroc": float(roc_auc_score(test_labels, scores)),
            "test_f1": float(f1_score(test_labels, pipeline.predict(test_features))),
        }
    return results


def _run_stage_5(
    best_result: dict[str, Any],
    dataset: DatasetSplits,
    runtime: RuntimeConfiguration,
) -> dict[str, Any]:
    configuration = _configuration_from_result(best_result)
    model = _build_model(configuration, runtime, dataset.input_feature_count)
    sampled_development = _extract_features(
        model, dataset.development_features, runtime.batch_size, runtime.device
    )
    sampled_test = _extract_features(
        model, dataset.test_features, runtime.batch_size, runtime.device
    )
    direct = _test_readouts(
        dataset.development_features,
        dataset.development_labels,
        dataset.test_features,
        dataset.test_labels,
        runtime,
    )
    sampled = _test_readouts(
        sampled_development,
        dataset.development_labels,
        sampled_test,
        dataset.test_labels,
        runtime,
    )
    return {
        "configuration": asdict(configuration),
        "direct_readouts": direct,
        "qks_readouts": sampled,
        "best_qks_readout": max(
            sampled,
            key=lambda name: (sampled[name]["test_auroc"], sampled[name]["test_f1"]),
        ),
    }


def run_ablation(
    config: dict[str, Any], dataset: DatasetSplits, run_dir: Path
) -> dict[str, Any]:
    """Execute the five-stage QRKS ablation and save its artifacts.

    Parameters
    ----------
    config : dict[str, Any]
        Validated ablation configuration.
    dataset : DatasetSplits
        Leakage-safe DCT dataset splits.
    run_dir : pathlib.Path
        Timestamped shared-runtime output directory.

    Returns
    -------
    dict[str, Any]
        Complete ablation state.
    """
    print("\n" + "="*80)
    print("Starting RF-RQKS Five-Stage Ablation")
    print("="*80)
    print(f"Dataset: train={dataset.train_labels.size}, "
          f"validation={dataset.validation_labels.size}, "
          f"development={dataset.development_labels.size}, "
          f"test={dataset.test_labels.size}")
    print(f"Input features: {dataset.input_feature_count}")
    print(f"Sampler: {config['sampler']}, Device: {config['device']}")
    print()
    runtime = RuntimeConfiguration(
        sampler=str(config["sampler"]),
        device=str(config["device"]),
        seed=int(config["seed"]),
        batch_size=int(config["batch_size"]),
        readout_c=float(config["readout_c"]),
        readout_max_iter=int(config["readout_max_iter"]),
        scale_sampled_features=bool(config["scale_sampled_features"]),
        kernel_components=int(config["kernel_components"]),
        kernel_gamma=float(config["kernel_gamma"]),
        maximum_output_features=int(config["maximum_output_features"]),
    )
    state: dict[str, Any] = {
        "dataset": {
            "train_samples": int(dataset.train_labels.size),
            "validation_samples": int(dataset.validation_labels.size),
            "development_samples": int(dataset.development_labels.size),
            "test_samples": int(dataset.test_labels.size),
            "input_features": dataset.input_feature_count,
        },
        "sampler": runtime.sampler,
        "stages": {f"stage_{number}": {"results": []} for number in range(1, 6)},
    }
    encoding = str(config["encoding_strategy"])
    entangler = str(config["entangling_strategy"])
    same_haar = bool(config["same_haar"])

    # Stage 1: Sweep mode and episode counts
    total_stage_1 = len(config["stage_1"]["mode_counts"]) * len(config["stage_1"]["episode_counts"]) * 2
    print(f"[Stage 1] Starting - will test {total_stage_1} configurations")
    stage_1_count = 0
    for mode_count in config["stage_1"]["mode_counts"]:
        for episode_count in config["stage_1"]["episode_counts"]:
            for entangling_strategy in (None, entangler):
                _run_configuration(
                    "stage_1",
                    ModelConfiguration(
                        photon_count=int(mode_count) // 2,
                        mode_count=int(mode_count),
                        depth=int(config["initial_depth"]),
                        episode_count=int(episode_count),
                        encoding_strategy=encoding,
                        entangling_strategy=entangling_strategy,
                        same_haar=same_haar,
                    ),
                    dataset,
                    runtime,
                    state,
                    run_dir,
                )
                stage_1_count += 1
                print(f"[Stage 1] Completed {stage_1_count}/{total_stage_1} configurations")
    print(f"[Stage 1] Complete! Shortlisting top configurations...\n")
    shortlisted = _best(
        state["stages"]["stage_1"]["results"],
        min(int(config["stage_2"]["shortlist_count"]), len(state["stages"]["stage_1"]["results"])),
    )
    print(f"[Stage 2] Starting - will test {len(shortlisted)} configs × {len(config['stage_2']['depths'])} depths = "
          f"{len(shortlisted) * len(config['stage_2']['depths'])} configurations")
    stage_2_count = 0
    for parent in shortlisted:
        base = _configuration_from_result(parent)
        for depth in config["stage_2"]["depths"]:
            _run_configuration(
                "stage_2",
                ModelConfiguration(**{**asdict(base), "depth": int(depth)}),
                dataset,
                runtime,
                state,
                run_dir,
            )
            stage_2_count += 1
            print(f"[Stage 2] Completed {stage_2_count}/{len(shortlisted) * len(config['stage_2']['depths'])} configurations")
    print(f"[Stage 2] Complete! Selecting best configuration...\n")
    best_stage_2 = _best(state["stages"]["stage_2"]["results"])[0]
    base = _configuration_from_result(best_stage_2)

    print(f"[Stage 3] Starting - will test {len(config['stage_3']['depth_episode_pairs'])} depth-episode pairs")
    stage_3_count = 0
    for depth, episode_count in config["stage_3"]["depth_episode_pairs"]:
        _run_configuration(
            "stage_3",
            ModelConfiguration(
                **{
                    **asdict(base),
                    "depth": int(depth),
                    "episode_count": int(episode_count),
                }
            ),
            dataset,
            runtime,
            state,
            run_dir,
        )
        stage_3_count += 1
        print(f"[Stage 3] Completed {stage_3_count}/{len(config['stage_3']['depth_episode_pairs'])} pairs")
    print(f"[Stage 3] Complete! Selecting best configuration...\n")
    best_stage_3 = _best(state["stages"]["stage_3"]["results"])[0]
    base = _configuration_from_result(best_stage_3)

    print(f"[Stage 4] Starting - will test {base.mode_count // 2} photon counts (1 to {base.mode_count // 2})")
    stage_4_count = 0
    for photon_count in range(1, base.mode_count // 2 + 1):
        _run_configuration(
            "stage_4",
            ModelConfiguration(**{**asdict(base), "photon_count": photon_count}),
            dataset,
            runtime,
            state,
            run_dir,
        )
        stage_4_count += 1
        print(f"[Stage 4] Completed {stage_4_count}/{base.mode_count // 2} photon counts")
    print(f"[Stage 4] Complete! Selecting best configuration...\n")
    best_stage_4 = _best(state["stages"]["stage_4"]["results"])[0]
    
    print(f"[Stage 5] Starting - testing best configuration on test set with {len(READOUT_NAMES)} readout types")
    stage_5 = _run_stage_5(best_stage_4, dataset, runtime)
    print(f"[Stage 5] Complete! Best readout: {stage_5['best_qks_readout']} "
          f"(AUROC: {stage_5['qks_readouts'][stage_5['best_qks_readout']]['test_auroc']:.4f})\n")
    state["stages"]["stage_5"]["results"].append(stage_5)
    state["best_model"] = best_stage_4
    state["status"] = "complete"

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    stage_1_results = state["stages"]["stage_1"]["results"]
    mode_counts = sorted(
        {result["configuration"]["mode_count"] for result in stage_1_results}
    )
    episode_counts = sorted(
        {result["configuration"]["episode_count"] for result in stage_1_results}
    )
    _plot_stage_1_heatmap(
        stage_1_results,
        entangling_enabled=False,
        mode_counts=mode_counts,
        episode_counts=episode_counts,
        output_path=figures_dir / "stage_1_validation_auroc_without_entangling.png",
    )
    _plot_stage_1_heatmap(
        stage_1_results,
        entangling_enabled=True,
        mode_counts=mode_counts,
        episode_counts=episode_counts,
        output_path=figures_dir / "stage_1_validation_auroc_with_entangling.png",
    )
    _plot_stage_2_curves(
        state["stages"]["stage_2"]["results"],
        figures_dir / "stage_2_validation_auroc_depth.png",
    )
    _plot_stage_3_bars(
        state["stages"]["stage_3"]["results"],
        figures_dir / "stage_3_validation_scores.png",
    )
    _plot_stage_4_lines(
        state["stages"]["stage_4"]["results"],
        figures_dir / "stage_4_validation_scores_photons.png",
    )
    # Keep the pre-existing smoke-test artifact name while exposing the
    # paper-compatible filename above.
    shutil.copy2(
        figures_dir / "stage_4_validation_scores_photons.png",
        figures_dir / "stage_4.png",
    )
    _plot_stage_5_readouts(
        state["stages"]["stage_5"]["results"][0],
        figures_dir / "stage_5_regression_functions.png",
    )
    _write_state(state, run_dir)
    print("="*80)
    print(f"Ablation Complete! Results saved to {run_dir}")
    print("="*80)
    print()
    return state
