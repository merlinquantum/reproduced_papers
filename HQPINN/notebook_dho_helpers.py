from __future__ import annotations

import csv
import importlib
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from IPython import get_ipython
    from IPython.display import Markdown, display
except ImportError:
    def get_ipython():
        return None

    def display(obj):
        print(obj)

    class Markdown(str):
        pass

from .config import (
    DEFAULT_N_OUTPUTS,
    DHO_HIDDEN_WIDTH,
    DHO_LR,
    DHO_N_EPOCHS,
    DHO_N_SAMPLES,
    DHO_NUM_HIDDEN_LAYERS,
    DHO_PLOT_EVERY,
    DTYPE,
    K,
    LAMBDA1,
    LAMBDA2,
    M,
    MU,
)
from .lib.DHO.core_dho import oscillator_loss, u_exact
from .runtime import seed_everything
from .utils import count_trainable_params, make_optimizer, make_time_grid

ip = get_ipython()
if ip is not None:
    ip.run_line_magic("matplotlib", "inline")

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
RESULTS_ROOT = REPO_ROOT / "HQPINN" / "results" / "DHO"
SUMMARY_PATH = RESULTS_ROOT / "dho_summary.csv"

MODEL_SPECS = [
    {
        "name": "cc",
        "paper_label": "CC",
        "branch_1": "classical",
        "branch_2": "classical",
        "branches": "MLP + MLP",
        "size_hint": "16-2",
        "case_prefix": "dho_cc",
        "factory_path": "HQPINN.lib.DHO.dho_cc:CC_PINN",
        "factory_kwargs": {
            "num_hidden_layers": DHO_NUM_HIDDEN_LAYERS,
            "hidden_width": DHO_HIDDEN_WIDTH,
        },
        "color": "#4C78A8",
    },
    {
        "name": "qq-pl",
        "paper_label": "QQ-PL",
        "branch_1": "quantum PennyLane",
        "branch_2": "quantum PennyLane",
        "branches": "PQC + PQC",
        "size_hint": str(DEFAULT_N_OUTPUTS),
        "case_prefix": "dho_qq_pl",
        "factory_path": "HQPINN.lib.DHO.dho_qq_pl:PP_PINN",
        "factory_kwargs": {"n_qubits": DEFAULT_N_OUTPUTS},
        "color": "#72B7B2",
    },
    {
        "name": "hy-pl",
        "paper_label": "HY-PL",
        "branch_1": "quantum PennyLane",
        "branch_2": "classical",
        "branches": "PQC + MLP",
        "size_hint": f"{DHO_HIDDEN_WIDTH}-{DHO_NUM_HIDDEN_LAYERS}-{DEFAULT_N_OUTPUTS}",
        "case_prefix": "dho_hy_pl",
        "factory_path": "HQPINN.lib.DHO.dho_hy_pl:CQ_PINN",
        "factory_kwargs": {
            "num_hidden_layers": DHO_NUM_HIDDEN_LAYERS,
            "hidden_width": DHO_HIDDEN_WIDTH,
            "n_qubits": DEFAULT_N_OUTPUTS,
        },
        "color": "#F58518",
    },
    {
        "name": "qq-m",
        "paper_label": "QQ-M",
        "branch_1": "quantum Merlin",
        "branch_2": "quantum Merlin",
        "branches": "interferometer + interferometer",
        "size_hint": "1",
        "case_prefix": "dho_qq_m",
        "factory_path": "HQPINN.lib.DHO.dho_qq_m:MM_PINN",
        "factory_kwargs": {"n_photons": 1},
        "color": "#54A24B",
    },
    {
        "name": "qq-mp",
        "paper_label": "QQ-MP",
        "branch_1": "quantum Perceval",
        "branch_2": "quantum Perceval",
        "branches": "Perceval + Perceval",
        "size_hint": "default",
        "case_prefix": "dho_qq_mp",
        "factory_path": "HQPINN.lib.DHO.dho_qq_mp:MM_PINN",
        "factory_kwargs": {},
        "color": "#B279A2",
    },
    {
        "name": "hy-m",
        "paper_label": "HY-M",
        "branch_1": "quantum Merlin",
        "branch_2": "classical",
        "branches": "interferometer + MLP",
        "size_hint": f"{DHO_HIDDEN_WIDTH}-{DHO_NUM_HIDDEN_LAYERS}-1",
        "case_prefix": "dho_hy_m",
        "factory_path": "HQPINN.lib.DHO.dho_hy_m:CI_PINN",
        "factory_kwargs": {
            "num_hidden_layers": DHO_NUM_HIDDEN_LAYERS,
            "hidden_width": DHO_HIDDEN_WIDTH,
            "n_photons": 1,
        },
        "color": "#E45756",
    },
    {
        "name": "hy-mp",
        "paper_label": "HY-MP",
        "branch_1": "quantum Perceval",
        "branch_2": "classical",
        "branches": "Perceval + MLP",
        "size_hint": f"{DHO_HIDDEN_WIDTH}-{DHO_NUM_HIDDEN_LAYERS}",
        "case_prefix": "dho_hy_mp",
        "factory_path": "HQPINN.lib.DHO.dho_hy_mp:CM_PINN",
        "factory_kwargs": {
            "num_hidden_layers": DHO_NUM_HIDDEN_LAYERS,
            "hidden_width": DHO_HIDDEN_WIDTH,
        },
        "color": "#FF9DA6",
    },
]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (7, 4)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def load_symbol(path: str):
    module_name, attr_name = path.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def build_model(spec: dict[str, object]):
    try:
        ctor = load_symbol(str(spec["factory_path"]))
    except Exception as exc:
        raise RuntimeError(
            f"Could not build model {spec['name']}. "
            "Check the quantum dependencies if you enable RUN_TRAINING=True."
        ) from exc
    return ctor(**dict(spec["factory_kwargs"]))


def checkpoint_path(case_prefix: str, run_id: str) -> Path:
    return REPO_ROOT / "HQPINN" / "models" / "DHO" / f"{case_prefix}_{run_id}.pt"


def prediction_png_path(model_name: str, run_id: str) -> Path:
    folder = RESULTS_ROOT / f"dho_{model_name.replace('-', '_')}"
    return folder / f"dho-{model_name}_{run_id}.png"


def load_summary_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with SUMMARY_PATH.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "run_id": raw["run_id"],
                        "model": raw["Model"],
                        "size": raw["Size"],
                        "epoch": int(raw["epoch"]),
                        "elapsed_s": float(raw["elapsed time (s)"]),
                        "params": int(raw["Trainable parameters"]),
                        "loss": float(raw["Loss"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def latest_saved_row(model_name: str) -> dict[str, object]:
    latest = None
    for row in load_summary_rows():
        if row["model"] != model_name:
            continue
        if latest is None or str(row["run_id"]) > str(latest["run_id"]):
            latest = row
    if latest is None:
        raise FileNotFoundError(f"No summary row found for {model_name} in {SUMMARY_PATH}.")
    return latest


def history_csv_path(model_name: str, run_id: str) -> Path:
    folder = RESULTS_ROOT / f"dho_{model_name.replace('-', '_')}"
    return folder / f"dho-{model_name}_{run_id}.csv"


def load_history(model_name: str, run_id: str) -> list[dict[str, float]]:
    path = history_csv_path(model_name, run_id)
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")
    history: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            history.append(
                {
                    "epoch": int(raw["epoch"]),
                    "elapsed_s": float(raw["elapsed time (s)"]),
                    "loss": float(raw["Loss"]),
                    "ic_u": float(raw["IC_u"]),
                    "ic_du": float(raw["IC_du"]),
                    "pde": float(raw["PDE"]),
                }
            )
    return history


def load_saved_case(spec: dict[str, object]) -> dict[str, object]:
    row = latest_saved_row(str(spec["name"]))
    history = load_history(str(spec["name"]), str(row["run_id"]))
    last_point = history[-1]
    return {
        "name": spec["name"],
        "paper_label": spec["paper_label"],
        "branch_1": spec["branch_1"],
        "branch_2": spec["branch_2"],
        "branches": spec["branches"],
        "color": spec["color"],
        "case_prefix": spec["case_prefix"],
        "spec": spec,
        "size": row["size"],
        "params": row["params"],
        "elapsed_s": last_point["elapsed_s"],
        "final_loss": last_point["loss"],
        "history": history,
        "source": f"latest saved run ({row['run_id']})",
        "run_id": row["run_id"],
    }


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def model_overview_markdown() -> str:
    lines = [
        "| Order | Code | Branch 1 | Branch 2 | Size used in this reproduction |",
        "|---|---|---|---|---|",
    ]
    for index, spec in enumerate(MODEL_SPECS, start=1):
        lines.append(
            f"| {index} | `{spec['name']}` | {spec['branch_1']} | {spec['branch_2']} | `{spec['size_hint']}` |"
        )
    return "\n".join(lines)


def plot_loss_curve(result: dict[str, object]) -> None:
    epochs = [point["epoch"] for point in result["history"]]
    losses = [point["loss"] for point in result["history"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(epochs, losses, marker="o", lw=2.2, color=result["color"])
    ax.set_title(f"Learning curve - {result['paper_label']} ({result['name']})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    plt.show()


def load_model_for_result(result: dict[str, object]):
    if result.get("model") is not None:
        model = result["model"]
        model.eval()
        return model

    run_id = result.get("run_id")
    if run_id is None:
        raise FileNotFoundError("No run_id available to reload the saved checkpoint.")

    ckpt_path = checkpoint_path(str(result["case_prefix"]), str(run_id))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model(dict(result["spec"]))
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def plot_ut_curve(result: dict[str, object]) -> None:
    t_np = np.linspace(0.0, 1.0, 400)
    t_eval = torch.tensor(t_np, dtype=DTYPE).reshape(-1, 1)
    u_exact_np = u_exact(t_np)

    try:
        model = load_model_for_result(result)
        with torch.no_grad():
            u_pred = model(t_eval).detach().cpu().numpy().reshape(-1)

        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        ax.plot(t_np, u_exact_np, "--", color="#1f1f1f", lw=2.0, label="Exact")
        ax.plot(
            t_np,
            u_pred,
            color=result["color"],
            lw=2.4,
            label=f"Prediction ({result['name']})",
        )
        ax.set_title(f"Trajectory u(t) - {result['paper_label']} ({result['name']})")
        ax.set_xlabel("t")
        ax.set_ylabel("u(t)")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        plt.show()
        return
    except Exception as exc:
        png_path = prediction_png_path(str(result["name"]), str(result.get("run_id", "")))
        if png_path.exists():
            print(f"u(t) curve reloaded from the saved figure because checkpoint reload failed: {exc}")
            image = plt.imread(png_path)
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.imshow(image)
            ax.set_title(f"Trajectory u(t) - {result['paper_label']} ({result['name']})")
            ax.axis("off")
            fig.tight_layout()
            plt.show()
            return
        print(f"u(t) curve unavailable for {result['name']}: {exc}")


def display_case_report(result: dict[str, object]) -> None:
    display(Markdown(f"### {result['paper_label']} (`{result['name']}`)"))
    print(f"Branches : {result['branch_1']} + {result['branch_2']}")
    print(f"Source   : {result['source']}")
    print(
        "Runtime  : "
        f"{format_seconds(float(result['elapsed_s']))} | "
        f"Parameters : {int(result['params'])} | "
        f"Final loss : {float(result['final_loss']):.4e}"
    )
    plot_loss_curve(result)
    plot_ut_curve(result)


def display_final_comparison(results: list[dict[str, object]]) -> None:
    headers = [
        "Order",
        "Model",
        "Branches",
        "Size",
        "Parameters",
        "Runtime (s)",
        "Final loss",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for index, result in enumerate(results, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{result['name']}`",
                    str(result["branches"]),
                    f"`{result['size']}`",
                    str(int(result["params"])),
                    f"{float(result['elapsed_s']):.2f}",
                    f"{float(result['final_loss']):.4e}",
                ]
            )
            + " |"
        )
    display(Markdown("\n".join(lines)))

    fastest = min(results, key=lambda item: float(item["elapsed_s"]))
    smallest = min(results, key=lambda item: int(item["params"]))
    lowest_loss = min(results, key=lambda item: float(item["final_loss"]))
    display(
        Markdown(
            "\n".join(
                [
                    f"- **Fastest runtime**: `{fastest['name']}` ({float(fastest['elapsed_s']):.2f} s)",
                    f"- **Smallest model**: `{smallest['name']}` ({int(smallest['params'])} parameters)",
                    f"- **Lowest final loss**: `{lowest_loss['name']}` ({float(lowest_loss['final_loss']):.4e})",
                ]
            )
        )
    )

    labels = [str(item["name"]) for item in results]
    colors = [str(item["color"]) for item in results]
    params = [int(item["params"]) for item in results]
    times = [float(item["elapsed_s"]) for item in results]
    losses = [float(item["final_loss"]) for item in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    axes[0].bar(labels, times, color=colors)
    axes[0].set_title("Runtime")
    axes[0].set_ylabel("seconds")
    axes[0].set_yscale("log")

    axes[1].bar(labels, params, color=colors)
    axes[1].set_title("Trainable parameters")
    axes[1].set_ylabel("count")

    axes[2].bar(labels, losses, color=colors)
    axes[2].set_title("Final loss")
    axes[2].set_ylabel("loss")
    axes[2].set_yscale("log")

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    plt.show()


__all__ = [
    "Markdown",
    "display",
    "plt",
    "np",
    "torch",
    "perf_counter",
    "REPO_ROOT",
    "RESULTS_ROOT",
    "SUMMARY_PATH",
    "MODEL_SPECS",
    "DHO_LR",
    "DHO_N_EPOCHS",
    "DHO_N_SAMPLES",
    "DHO_PLOT_EVERY",
    "DTYPE",
    "K",
    "LAMBDA1",
    "LAMBDA2",
    "M",
    "MU",
    "oscillator_loss",
    "u_exact",
    "seed_everything",
    "count_trainable_params",
    "make_optimizer",
    "make_time_grid",
    "build_model",
    "load_saved_case",
    "model_overview_markdown",
    "display_case_report",
    "display_final_comparison",
]
