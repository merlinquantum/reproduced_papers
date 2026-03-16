"""
Reproduce experimental results in Fig. 4c.

We consider experimental conditions with imperfect indistinguishability = 0.9720.
Ad-hoc data is generated ideally with bunching and perfect indistinguishability.
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
    "input_state": [1, 1, 0, 0, 0, 0],
    "superset_size": 1000,
    "reg": 0.02,
    "reps": 10,
    "test_size": 0.33,
    "indistinguishability": 0.972,
    "no_bunching": False,
    "force_psd": False,
    "shots": None,
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


def run_accuracy_vs_kernel(exp_cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    cfg = _resolve_config(exp_cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_state = cfg["input_state"]
    superset_size = int(cfg["superset_size"])
    reg = float(cfg["reg"])
    reps = int(cfg["reps"])
    test_size = float(cfg["test_size"])
    train_size = 1 - test_size
    indistinguishability = float(cfg["indistinguishability"])
    no_bunching = bool(cfg["no_bunching"])
    force_psd = bool(cfg["force_psd"])
    shots = cfg["shots"]
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
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from utils.feature_map import circuit_func
    from utils.generate_data import generate_data
    from utils.noise import NoisySLOSComputeGraph

    circuit = GenericInterferometer(m=len(input_state), fun_gen=circuit_func)
    input_size = len(circuit.get_parameters())

    feature_map = FeatureMap(circuit, input_size, input_parameters=["phi"])

    quantum_kernel = FidelityKernel(
        feature_map,
        input_state,
        shots=shots,
        no_bunching=no_bunching,
        force_psd=force_psd,
    )
    coherent_kernel = FidelityKernel(
        feature_map,
        input_state,
        shots=shots,
        no_bunching=no_bunching,
        force_psd=force_psd,
    )

    quantum_kernel._slos_graph = NoisySLOSComputeGraph(
        indistinguishability=indistinguishability
    )
    coherent_kernel._slos_graph = NoisySLOSComputeGraph(indistinguishability=0.0)

    nt_available = True
    try:
        import jax.numpy as jnp
        import neural_tangents as nt
    except ImportError as exc:
        nt_available = False
        jnp = None  # type: ignore[assignment]
        nt = None  # type: ignore[assignment]
        LOGGER.warning("Skipping Neural Tangent baseline due to import error: %s", exc)

    if nt_available:
        _, _, neural_tangent_kernel = nt.stax.serial(
            nt.stax.Dense(30, W_std=1.5, b_std=0.05),
            nt.stax.Erf(),
            nt.stax.Dense(30, W_std=1.5, b_std=0.05),
            nt.stax.Erf(),
        )

    X, y, _, _, _ = generate_data(
        superset_size,
        reg,
        input_state,
    )
    all_indices = np.arange(superset_size)
    np.random.shuffle(all_indices)

    def tune_gaussian(X_tune, y_tune):
        param_grid = {"gamma": [1, 0.1, 0.01, 0.001], "kernel": ["rbf"]}
        grid_search = GridSearchCV(
            SVC(), param_grid, cv=5, scoring="accuracy", n_jobs=-1
        )
        grid_search.fit(X_tune, y_tune)
        return grid_search.best_params_["gamma"]

    def tune_poly(X_tune, y_tune):
        param_grid = {
            "coef0": [0.0, 0.5, 1.0],
            "degree": [2, 3, 4],
            "kernel": ["poly"],
        }
        grid_search = GridSearchCV(
            SVC(), param_grid, cv=5, scoring="accuracy", n_jobs=-1
        )
        grid_search.fit(X_tune, y_tune)
        coeff = grid_search.best_params_["coef0"]
        degree = grid_search.best_params_["degree"]
        return coeff, degree

    kernel_matrix_q = quantum_kernel(X)
    kernel_matrix_c = coherent_kernel(X)

    scores_q = []
    scores_c = []
    scores_g = []
    scores_l = []
    scores_p = []
    scores_nt = []

    for data_size in data_sizes:
        scores_q_rep = []
        scores_c_rep = []
        scores_g_rep = []
        scores_l_rep = []
        scores_p_rep = []
        scores_nt_rep = []

        for _ in range(reps):
            y_train = [1]
            while True:
                idx = np.random.choice(all_indices, size=data_size, replace=False)
                train_idx_rep = idx[: int(train_size * data_size)]
                test_idx_rep = idx[int(train_size * data_size) :]

                X_train = X[train_idx_rep]
                X_test = X[test_idx_rep]

                y_train = y[train_idx_rep]
                y_test = y[test_idx_rep]

                if len(np.unique(y_train)) != 1:
                    break

            kernel_matrix_q_train = kernel_matrix_q[
                np.ix_(train_idx_rep, train_idx_rep)
            ]
            kernel_matrix_c_train = kernel_matrix_c[
                np.ix_(train_idx_rep, train_idx_rep)
            ]

            kernel_matrix_q_test = kernel_matrix_q[np.ix_(test_idx_rep, train_idx_rep)]
            kernel_matrix_c_test = kernel_matrix_c[np.ix_(test_idx_rep, train_idx_rep)]

            gamma = tune_gaussian(X_train, y_train)
            coeff, degree = tune_poly(X_train, y_train)

            svc_q = SVC(kernel="precomputed")
            svc_c = SVC(kernel="precomputed")
            svc_g = SVC(kernel="rbf", gamma=gamma)
            svc_l = SVC(kernel="linear")
            svc_p = SVC(kernel="poly", degree=degree, coef0=coeff)

            svc_q.fit(kernel_matrix_q_train, y_train)
            svc_c.fit(kernel_matrix_c_train, y_train)
            svc_g.fit(X_train, y_train)
            svc_l.fit(X_train, y_train)
            svc_p.fit(X_train, y_train)

            score_q = svc_q.score(kernel_matrix_q_test, y_test)
            score_c = svc_c.score(kernel_matrix_c_test, y_test)
            score_g = svc_g.score(X_test, y_test)
            score_l = svc_l.score(X_test, y_test)
            score_p = svc_p.score(X_test, y_test)

            if nt_available:
                X_train_jnp = jnp.array(X_train)
                X_test_jnp = jnp.array(X_test)
                y_train_jnp = jnp.array(y_train).reshape(-1, 1)

                predict_fn = nt.predict.gradient_descent_mse_ensemble(
                    neural_tangent_kernel, X_train_jnp, y_train_jnp, diag_reg=1e-4
                )
                y_pred, _ = predict_fn(x_test=X_test_jnp, get="ntk", compute_cov=True)
                score_nt = np.mean(np.sign(y_pred.flatten()) == y_test)

            scores_q_rep.append(score_q)
            scores_c_rep.append(score_c)
            scores_g_rep.append(score_g)
            scores_l_rep.append(score_l)
            scores_p_rep.append(score_p)
            if nt_available:
                scores_nt_rep.append(score_nt)

        scores_q.append(scores_q_rep)
        scores_c.append(scores_c_rep)
        scores_g.append(scores_g_rep)
        scores_l.append(scores_l_rep)
        scores_p.append(scores_p_rep)
        if nt_available:
            scores_nt.append(scores_nt_rep)

    scores_q = np.array(scores_q)
    scores_c = np.array(scores_c)
    scores_g = np.array(scores_g)
    scores_l = np.array(scores_l)
    scores_p = np.array(scores_p)
    scores_nt = np.array(scores_nt) if nt_available else np.array([])

    mean_scores_q = np.mean(scores_q, axis=1)
    mean_scores_c = np.mean(scores_c, axis=1)
    mean_scores_g = np.mean(scores_g, axis=1)
    mean_scores_l = np.mean(scores_l, axis=1)
    mean_scores_p = np.mean(scores_p, axis=1)
    mean_scores_nt = np.mean(scores_nt, axis=1) if nt_available else None

    std_scores_q = np.std(scores_q, axis=1) / 2
    std_scores_c = np.std(scores_c, axis=1) / 2
    std_scores_g = np.std(scores_g, axis=1) / 2
    std_scores_l = np.std(scores_l, axis=1) / 2
    std_scores_p = np.std(scores_p, axis=1) / 2
    std_scores_nt = np.std(scores_nt, axis=1) / 2 if nt_available else None

    hyperparameters = {
        "input_state": input_state,
        "superset_size": superset_size,
        "reg": reg,
        "seed": seed,
        "reps": reps,
        "test_size": test_size,
        "data_sizes": [int(value) for value in data_sizes],
        "indistinguishability": indistinguishability,
        "shots": shots,
    }

    hyper_path = output_dir / "hyperparameters.json"
    with hyper_path.open("w", encoding="utf-8") as handle:
        json.dump(hyperparameters, handle, indent=4)

    data_path = output_dir / "data.npz"
    save_payload = {
        "scores_q": scores_q,
        "scores_c": scores_c,
        "scores_g": scores_g,
        "scores_l": scores_l,
        "scores_p": scores_p,
    }
    if nt_available:
        save_payload["scores_nt"] = scores_nt
    np.savez(data_path, **save_payload)

    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["axes.labelsize"] = 15
    plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    plt.rc("text", usetex=use_tex)

    fig = plt.figure(figsize=(5.5, 3.0))

    plt.errorbar(
        data_sizes,
        mean_scores_q,
        yerr=std_scores_q,
        label="Quantum",
        linewidth=1.5,
        marker="o",
        capsize=5,
        markersize=5,
        linestyle="--",
    )
    plt.errorbar(
        data_sizes,
        mean_scores_c,
        yerr=std_scores_c,
        label="Coherent",
        linewidth=1.5,
        marker="o",
        capsize=5,
        markersize=5,
        linestyle="--",
    )
    plt.errorbar(
        data_sizes,
        mean_scores_g,
        yerr=std_scores_g,
        label="Gaussian",
        linewidth=1.5,
        marker="o",
        capsize=5,
        markersize=5,
        linestyle="--",
    )
    plt.errorbar(
        data_sizes,
        mean_scores_l,
        yerr=std_scores_l,
        label="Linear",
        linewidth=1.5,
        marker="o",
        capsize=5,
        markersize=5,
        linestyle="--",
    )
    plt.errorbar(
        data_sizes,
        mean_scores_p,
        yerr=std_scores_p,
        label="Polynomial",
        linewidth=1.5,
        marker="o",
        capsize=5,
        markersize=5,
        linestyle="--",
    )
    if nt_available:
        plt.errorbar(
            data_sizes,
            mean_scores_nt,
            yerr=std_scores_nt,
            label="Neural Tangent",
            linewidth=1.5,
            marker="o",
            capsize=5,
            markersize=5,
            linestyle="--",
        )

    plt.xlabel("Data size")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right", bbox_to_anchor=(1.45, 1.025))

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
    run_accuracy_vs_kernel({}, Path("results") / "accuracy_vs_kernel")
