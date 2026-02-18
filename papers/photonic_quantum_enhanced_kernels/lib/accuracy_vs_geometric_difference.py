"""
Reproduce simulated results in Supplementary Fig. 1.

We consider kernels generated with fully indistinguishable and fully
distinguishable photons and include bunching events.
"""

from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

DEFAULTS = {
    "num_points": 800,
    "n": 2,
    "data_size": 150,
    "test_size": 0.33,
    "reg": 0.02,
    "m_min": 4,
    "m_max": 7,
    "seed": 42,
    "spline_smoothing": 5,
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


def run_accuracy_vs_geometric_difference(
    exp_cfg: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    cfg = _resolve_config(exp_cfg)

    num_points = int(cfg["num_points"])
    n = int(cfg["n"])
    data_size = int(cfg["data_size"])
    test_size = float(cfg["test_size"])
    reg = float(cfg["reg"])
    m_min = int(cfg["m_min"])
    m_max = int(cfg["m_max"])
    seed = int(cfg["seed"])
    spline_smoothing = float(cfg["spline_smoothing"])

    plotting = cfg["plotting"]
    use_tex = bool(plotting.get("use_tex", True))
    show_plots = bool(plotting.get("show", False))

    np.random.seed(seed)
    random.seed(seed)

    import matplotlib

    if not show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import UnivariateSpline
    from sklearn.svm import SVC
    from tqdm import tqdm
    from utils.generate_data import generate_data

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_dir = output_dir / f"n={n}"
    target_dir.mkdir(parents=True, exist_ok=True)

    geometric_differences = []
    scores_q = []
    scores_c = []

    for _ in tqdm(range(num_points), desc="Sampling geometric differences"):
        m = random.randint(m_min, m_max)

        input_state = [1] * n + [0] * (m - n)
        random.shuffle(input_state)

        X, y, geometric_difference, kernel_matrix_q, kernel_matrix_c = generate_data(
            data_size, reg, input_state
        )

        all_indices = np.arange(data_size)
        np.random.shuffle(all_indices)

        test_idx = all_indices[: int(test_size * data_size)]
        train_idx = all_indices[int(test_size * data_size) :]

        y_train, y_test = y[train_idx], y[test_idx]

        while len(np.unique(y_train)) == 1:
            np.random.shuffle(all_indices)

            test_idx = all_indices[: int(test_size * data_size)]
            train_idx = all_indices[int(test_size * data_size) :]

            y_train, y_test = y[train_idx], y[test_idx]

        kernel_train_idx = np.ix_(train_idx, train_idx)
        kernel_test_idx = np.ix_(test_idx, train_idx)

        kernel_matrix_q_train = kernel_matrix_q[kernel_train_idx]
        kernel_matrix_c_train = kernel_matrix_c[kernel_train_idx]
        kernel_matrix_q_test = kernel_matrix_q[kernel_test_idx]
        kernel_matrix_c_test = kernel_matrix_c[kernel_test_idx]

        svc_q = SVC(kernel="precomputed")
        svc_c = SVC(kernel="precomputed")

        svc_q.fit(kernel_matrix_q_train, y_train)
        svc_c.fit(kernel_matrix_c_train, y_train)

        score_q = svc_q.score(kernel_matrix_q_test, y_test)
        score_c = svc_c.score(kernel_matrix_c_test, y_test)

        geometric_differences.append(geometric_difference)
        scores_q.append(score_q)
        scores_c.append(score_c)

    sort_indices = np.argsort(geometric_differences)
    geometric_differences = np.array(geometric_differences, dtype=float)[sort_indices]
    scores_q = np.array(scores_q, dtype=float)[sort_indices]
    scores_c = np.array(scores_c, dtype=float)[sort_indices]

    interpolated_gcq_vals = np.linspace(
        min(geometric_differences), max(geometric_differences), 100
    )
    spline_q = UnivariateSpline(geometric_differences, scores_q, s=spline_smoothing)
    spline_c = UnivariateSpline(geometric_differences, scores_c, s=spline_smoothing)

    trend_q = spline_q(interpolated_gcq_vals)
    trend_c = spline_c(interpolated_gcq_vals)

    hyperparameters = {
        "num_points": num_points,
        "n": n,
        "data_size": data_size,
        "test_size": test_size,
        "reg": reg,
        "m_min": m_min,
        "m_max": m_max,
        "seed": seed,
        "spline_smoothing": spline_smoothing,
    }

    hyper_path = target_dir / "hyperparameters.json"
    with hyper_path.open("w", encoding="utf-8") as handle:
        json.dump(hyperparameters, handle, indent=4)

    data_path = target_dir / "data.npz"
    np.savez(
        data_path,
        geometric_differences=geometric_differences,
        X=X,
        y=y,
    )

    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["axes.labelsize"] = 15
    plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    plt.rc("text", usetex=use_tex)

    fig = plt.figure()
    plt.scatter(
        geometric_differences,
        scores_q,
        s=5,
        label="Indistinguishable",
        alpha=0.95,
        zorder=2,
    )
    plt.scatter(
        geometric_differences,
        scores_c,
        s=5,
        label="Distinguishable",
        alpha=0.95,
        zorder=1,
    )
    plt.plot(
        interpolated_gcq_vals,
        trend_q,
        linestyle="dashed",
        linewidth=2,
        c="#15517c",
        zorder=3,
    )
    plt.plot(
        interpolated_gcq_vals,
        trend_c,
        linestyle="dashed",
        linewidth=2,
        c="#c3620d",
        zorder=3,
    )

    plt.xlabel("Geometric difference $g_{CQ}$")
    plt.ylabel("Accuracy")
    plt.legend()

    plot_path = target_dir / "plot.png"
    plt.savefig(plot_path, dpi=300)
    if show_plots:
        plt.show()
    plt.close(fig)

    LOGGER.info("Saved artifacts to %s", target_dir)
    return {
        "output_dir": str(target_dir),
        "plot_path": str(plot_path),
        "data_path": str(data_path),
    }


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    run_accuracy_vs_geometric_difference(
        {}, Path("results") / "accuracy_vs_geometric_difference"
    )
