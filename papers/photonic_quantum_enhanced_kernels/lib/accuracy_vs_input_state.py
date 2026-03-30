"""
Reproduce experimental results in Fig. 4a and 4b.

We consider experimental conditions with indistinguishability = 0.9720.
Ad-hoc data is generated ideally with perfect indistinguishability.
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
    "input_states": [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]],
    "superset_size": 2000,
    "reg": 0.02,
    "reps": 5,
    "test_size": 0.33,
    "indistinguishability": 0.972,
    "shots": None,
    "force_psd": True,
    "no_bunching": False,
    "seed": 42,
    "data_sizes": {"min": 40, "max": 100, "step": 20},
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


def run_accuracy_vs_input_state(
    exp_cfg: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    cfg = _resolve_config(exp_cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_states = cfg["input_states"]
    superset_size = int(cfg["superset_size"])
    reg = float(cfg["reg"])
    reps = int(cfg["reps"])
    test_size = float(cfg["test_size"])
    train_size = 1 - test_size
    indistinguishability = float(cfg["indistinguishability"])
    shots = cfg["shots"]
    force_psd = bool(cfg["force_psd"])
    no_bunching = bool(cfg["no_bunching"])
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
    from utils.noise import NoisySLOSComputeGraph

    circuit = GenericInterferometer(m=len(input_states[0]), fun_gen=circuit_func)
    input_size = len(circuit.get_parameters())
    feature_map = FeatureMap(circuit, input_size, input_parameters=["phi"])

    scores_q_input_state = []
    scores_c_input_state = []

    all_indices = np.arange(superset_size)
    pbar_size = len(input_states) * len(data_sizes)

    with tqdm(total=pbar_size, desc="Generating training curves") as pbar:
        for input_state in input_states:
            X, y, _, _, _ = generate_data(
                superset_size,
                reg,
                input_state,
            )
            quantum_kernel = FidelityKernel(
                feature_map,
                input_state,
                shots=shots,
                force_psd=force_psd,
                no_bunching=no_bunching,
            )
            coherent_kernel = FidelityKernel(
                feature_map,
                input_state,
                shots=shots,
                force_psd=force_psd,
                no_bunching=no_bunching,
            )

            quantum_kernel._slos_graph = NoisySLOSComputeGraph(
                indistinguishability=indistinguishability
            )
            coherent_kernel._slos_graph = NoisySLOSComputeGraph(
                indistinguishability=0.0
            )

            np.random.shuffle(all_indices)
            train_idx = all_indices[int(test_size * superset_size) :]
            test_idx = all_indices[: int(test_size * superset_size)]

            kernel_matrix_q = quantum_kernel(X)
            kernel_matrix_c = coherent_kernel(X)

            scores_q_data_size = []
            scores_c_data_size = []

            for data_size in data_sizes:
                scores_q_rep = []
                scores_c_rep = []

                for _ in range(reps):
                    y_train = [1]

                    while len(np.unique(y_train)) == 1:
                        np.random.shuffle(train_idx)
                        np.random.shuffle(test_idx)

                        train_idx_rep = train_idx[: int(data_size * train_size)]
                        test_idx_rep = test_idx[: int(data_size * test_size)]

                        _X_train, _X_test = X[train_idx_rep], X[test_idx_rep]
                        y_train, y_test = y[train_idx_rep], y[test_idx_rep]

                    kernel_train_idx = np.ix_(train_idx_rep, train_idx_rep)
                    kernel_test_idx = np.ix_(test_idx_rep, train_idx_rep)

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

                    scores_q_rep.append(score_q)
                    scores_c_rep.append(score_c)

                scores_q_data_size.append(scores_q_rep)
                scores_c_data_size.append(scores_c_rep)
                pbar.update(1)

            scores_q_input_state.append(scores_q_data_size)
            scores_c_input_state.append(scores_c_data_size)

    scores_q_input_state = np.array(scores_q_input_state)
    scores_c_input_state = np.array(scores_c_input_state)

    mean_scores_q = np.mean(scores_q_input_state, axis=2)
    mean_scores_c = np.mean(scores_c_input_state, axis=2)

    std_scores_q = np.std(scores_q_input_state, axis=2) / 2
    std_scores_c = np.std(scores_c_input_state, axis=2) / 2

    hyperparameters = {
        "input_states": input_states,
        "superset_size": superset_size,
        "reg": reg,
        "reps": reps,
        "test_size": test_size,
        "data_sizes": [int(value) for value in data_sizes],
        "seed": seed,
        "indistinguishability": indistinguishability,
        "shots": shots,
    }

    hyper_path = output_dir / "hyperparameters.json"
    with hyper_path.open("w", encoding="utf-8") as handle:
        json.dump(hyperparameters, handle, indent=4)

    data_path = output_dir / "data.npz"
    np.savez(
        data_path,
        scores_q_input_state=scores_q_input_state,
        scores_c_input_state=scores_c_input_state,
    )

    min_q = mean_scores_q - std_scores_q
    min_c = mean_scores_c - std_scores_c
    max_q = mean_scores_q + std_scores_q
    max_c = mean_scores_c + std_scores_c

    abs_min_val = min([np.min(min_q), np.min(min_c)]) - 0.01
    abs_max_val = max([np.max(max_q), np.max(max_c)]) + 0.01

    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["axes.titlesize"] = 15
    plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    plt.rc("text", usetex=use_tex)

    fig, axes = plt.subplots(1, len(input_states), figsize=(7, 2.5), sharex=True)
    if len(input_states) == 1:
        axes = [axes]

    for i, (ax, input_state) in enumerate(zip(axes, input_states)):
        ax.errorbar(
            data_sizes,
            mean_scores_q[i],
            yerr=std_scores_q[i],
            label="Indistinguishable",
            linewidth=1.5,
            marker="o",
            capsize=5,
            markersize=5,
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
        )
        if i != 0:
            ax.set_yticks([])
        else:
            ax.set_ylabel("Accuracy")

        ax.set_title(r"$|" + str(input_state)[1:-1] + r"\rangle$")
        ax.set_ylim(abs_min_val, abs_max_val)
        ax.set_xlabel("Dataset Size")

        if i == len(input_states) - 1:
            ax.legend(loc="upper right", bbox_to_anchor=(1.82, 1.05))

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
    run_accuracy_vs_input_state({}, Path("results") / "accuracy_vs_input_state")
