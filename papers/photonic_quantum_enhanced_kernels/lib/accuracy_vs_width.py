"""
Reproduce simulated results in Supplementary Fig. 2.

We consider three kernels: perfect indistinguishability with bunching,
fully distinguishable photons, and perfect indistinguishability with
no bunching events.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

DEFAULTS = {
    "widths": [4, 5, 6, 7],
    "superset_size": 300,
    "reg": 0.02,
    "reps": 10,
    "test_size": 0.33,
    "seed": 42,
    "data_sizes": {"min": 40, "max": 200, "step": 20},
    "plotting": {"use_tex": False, "show": True},
}


def _resolve_plotting(plot_cfg: Any) -> dict[str, Any]:
    merged = dict(DEFAULTS["plotting"])
    if isinstance(plot_cfg, dict):
        merged.update(plot_cfg)
    return merged


def _resolve_config(exp_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if exp_cfg:
        for key, value in exp_cfg.items():
            if key != "plotting":
                cfg[key] = value
    cfg["plotting"] = _resolve_plotting(exp_cfg.get("plotting") if exp_cfg else None)
    return cfg


def _resolve_data_sizes(cfg: dict[str, Any]) -> np.ndarray:
    data_sizes_cfg = cfg.get("data_sizes")
    if isinstance(data_sizes_cfg, (list, tuple, np.ndarray)):
        return np.array([int(value) for value in data_sizes_cfg], dtype=int)

    if isinstance(data_sizes_cfg, dict):
        min_size = int(data_sizes_cfg.get("min", DEFAULTS["data_sizes"]["min"]))
        max_size = int(data_sizes_cfg.get("max", DEFAULTS["data_sizes"]["max"]))
        step = int(data_sizes_cfg.get("step", DEFAULTS["data_sizes"]["step"]))
    else:
        min_size = DEFAULTS["data_sizes"]["min"]
        max_size = DEFAULTS["data_sizes"]["max"]
        step = DEFAULTS["data_sizes"]["step"]

    return np.arange(min_size, max_size + step, step)


def run_accuracy_vs_width(exp_cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    cfg = _resolve_config(exp_cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    widths = list(cfg["widths"])
    superset_size = int(cfg["superset_size"])
    reg = float(cfg["reg"])
    reps = int(cfg["reps"])
    test_size = float(cfg["test_size"])
    seed = int(cfg["seed"])
    data_sizes = _resolve_data_sizes(cfg)

    np.random.seed(seed)

    plotting = cfg["plotting"]
    use_tex = bool(plotting.get("use_tex", True))
    show_plots = bool(plotting.get("show", False))

    import matplotlib

    if not show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from merlin.algorithms import FeatureMap, FidelityKernel
    from perceval import GenericInterferometer
    from sklearn.svm import SVC
    from tqdm import tqdm
    from utils.feature_map import circuit_func
    from utils.generate_data import generate_data

    scores_q_width = []
    scores_c_width = []
    scores_u_width = []

    all_indices = np.arange(superset_size)
    np.set_printoptions(linewidth=500)
    for width in tqdm(widths, desc="Training curve per width"):
        input_state = [1, 1] + [0] * (width - 2)

        circuit = GenericInterferometer(width, circuit_func)
        input_size = len(circuit.get_parameters())

        feature_map = FeatureMap(circuit, input_size, input_parameters=["phi"])

        unbunching_kernel = FidelityKernel(
            feature_map,
            input_state,
            no_bunching=True,
            force_psd=False,
        )

        X, y, _, kernel_matrix_q, kernel_matrix_c = generate_data(
            superset_size, reg, input_state
        )

        kernel_matrix_u = unbunching_kernel(X)

        scores_q_data_size = []
        scores_c_data_size = []
        scores_u_data_size = []

        for data_size in data_sizes:
            scores_q_rep = []
            scores_c_rep = []
            scores_u_rep = []

            for _ in range(reps):
                y_train = [1]
                while len(np.unique(y_train)) == 1:
                    np.random.shuffle(all_indices)

                    test_idx = all_indices[: int(test_size * data_size)]
                    train_idx = all_indices[int(test_size * data_size) : data_size]

                    _X_train, _X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                kernel_train_idx = np.ix_(train_idx, train_idx)
                kernel_test_idx = np.ix_(test_idx, train_idx)

                kernel_matrix_q_train = kernel_matrix_q[kernel_train_idx]
                kernel_matrix_c_train = kernel_matrix_c[kernel_train_idx]
                kernel_matrix_u_train = kernel_matrix_u[kernel_train_idx]
                kernel_matrix_q_test = kernel_matrix_q[kernel_test_idx]
                kernel_matrix_c_test = kernel_matrix_c[kernel_test_idx]
                kernel_matrix_u_test = kernel_matrix_u[kernel_test_idx]

                svc_q = SVC(kernel="precomputed")
                svc_c = SVC(kernel="precomputed")
                svc_u = SVC(kernel="precomputed")

                svc_q.fit(kernel_matrix_q_train, y_train)
                svc_c.fit(kernel_matrix_c_train, y_train)
                svc_u.fit(kernel_matrix_u_train, y_train)

                score_q = svc_q.score(kernel_matrix_q_test, y_test)
                score_c = svc_c.score(kernel_matrix_c_test, y_test)
                score_u = svc_u.score(kernel_matrix_u_test, y_test)

                scores_q_rep.append(score_q)
                scores_c_rep.append(score_c)
                scores_u_rep.append(score_u)

            scores_q_data_size.append(scores_q_rep)
            scores_c_data_size.append(scores_c_rep)
            scores_u_data_size.append(scores_u_rep)

        scores_q_width.append(scores_q_data_size)
        scores_c_width.append(scores_c_data_size)
        scores_u_width.append(scores_u_data_size)

    scores_q_width = np.array(scores_q_width)
    scores_c_width = np.array(scores_c_width)
    scores_u_width = np.array(scores_u_width)

    mean_scores_q = np.mean(scores_q_width, axis=2)
    mean_scores_c = np.mean(scores_c_width, axis=2)
    mean_scores_u = np.mean(scores_u_width, axis=2)

    std_scores_q = np.std(scores_q_width, axis=2)
    std_scores_c = np.std(scores_c_width, axis=2)
    std_scores_u = np.std(scores_u_width, axis=2)

    hyperparameters = {
        "widths": widths,
        "test_size": test_size,
        "reg": reg,
        "data_sizes": [int(value) for value in data_sizes],
        "reps": reps,
        "superset_size": superset_size,
        "seed": seed,
    }

    hyper_path = output_dir / "hyperparameters.json"
    with hyper_path.open("w", encoding="utf-8") as handle:
        json.dump(hyperparameters, handle, indent=4)

    data_path = output_dir / "data.npz"
    np.savez(
        data_path,
        kernel_matrix_q=kernel_matrix_q,
        kernel_matrix_c=kernel_matrix_c,
        kernel_matrix_u=kernel_matrix_u,
        X=X,
        y=y,
        scores_q_width=scores_q_width,
        scores_c_width=scores_c_width,
        scores_u_width=scores_u_width,
    )

    min_q = mean_scores_q - std_scores_q
    min_c = mean_scores_c - std_scores_c
    min_u = mean_scores_u - std_scores_u
    abs_min_val = min([np.min(min_q), np.min(min_c), np.min(min_u)]) - 0.05

    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["axes.labelsize"] = 15
    plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    plt.rc("text", usetex=use_tex)

    fig, axes = plt.subplots(len(widths), 1, figsize=(5.5, 6.5), sharex=True)
    fig.subplots_adjust(hspace=0.27, right=0.85)
    if len(widths) == 1:
        axes = [axes]

    for i, (ax, width) in enumerate(zip(axes, widths)):
        ax.errorbar(
            data_sizes,
            mean_scores_q[i],
            yerr=std_scores_q[i],
            label="Indistinguishable",
            linewidth=1.5,
            marker="o",
            capsize=5,
            markersize=5,
            linestyle="--",
        )
        ax.errorbar(
            data_sizes,
            mean_scores_c[i],
            yerr=std_scores_c[i],
            label="Distinguishable",
            linewidth=1.5,
            marker="o",
            capsize=5,
            markersize=5,
            linestyle="--",
        )
        ax.errorbar(
            data_sizes,
            mean_scores_u[i],
            yerr=std_scores_u[i],
            label="No bunching",
            linewidth=1.5,
            marker="o",
            capsize=5,
            markersize=5,
            linestyle="--",
        )

        ax.set_title(f"$w={width}$")
        ax.set_ylim(abs_min_val, 1.0)
        if i < len(widths) - 1:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", bottom=False, top=False)

    axes[-1].set_xlabel("Dataset Size")
    axes[-1].set_xticks(data_sizes)
    axes[-1].set_xticklabels(data_sizes)
    fig.text(0.00, 0.5, "Accuracy", va="center", rotation="vertical", fontsize=15)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.22, 0.895))

    plot_path = output_dir / "plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)

    LOGGER.info("Saved artifacts to %s", output_dir)
    return {
        "output_dir": str(output_dir),
        "plot_path": str(plot_path),
        "data_path": str(data_path),
    }


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    run_accuracy_vs_width({}, Path("results") / "accuracy_vs_width")
