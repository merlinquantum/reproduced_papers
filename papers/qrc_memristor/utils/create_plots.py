import argparse
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# --- STYLE CONFIGURATION ---
sns.set_style("whitegrid")

mpl.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX for all text
        "font.family": "serif",  # Serif font (Computer Modern)
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)


def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)


def compute_linear_baseline(x, y_true):
    """
    Computes a simple Linear Regression on the raw input x.
    """
    # Reshape for sklearn
    x_np = x.reshape(-1, 1)

    # Use same split as quantum models (90%)
    split = int(0.9 * len(x))
    x_train = x_np[:split]
    y_train = y_true[:split]

    # Fit Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)

    # Predict Full Range
    y_pred = lin_reg.predict(x_np)

    # Calculate MSE on Test Split (last 10%)
    y_test_true = y_true[split:]
    y_test_pred = y_pred[split:]
    mse = np.mean((y_test_true - y_test_pred) ** 2)

    return y_pred, mse


def plot_nonlinear_from_file(result_dir, plot_data):
    """
    Generates Figure 3 (Target vs Prediction) using saved plot_data.
    """
    print(f"Plotting Nonlinear Results from {result_dir}...")

    # Load Data
    x = np.array(plot_data["x"]) if "x" in plot_data else None
    y_target = np.array(plot_data["y_target"])
    y_pred = np.array(plot_data["y_pred"])
    mse = plot_data["mse"]

    # If x wasn't saved, regenerate it (standard linspace)
    if x is None:
        x = np.linspace(0, 1, len(y_target))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        x, y_target, color="black", linestyle="-", linewidth=3, label="Target ($x^4$)"
    )
    ax.plot(
        x, y_pred, color="#d62728", linestyle="--", linewidth=2.5, label="Prediction"
    )

    ax.set_title(f"Non-linear Task Results\nMSE: {mse:.2e}", fontweight="bold")
    ax.set_xlabel("Input $x$")
    ax.set_ylabel("Output $y$")
    ax.legend(loc="upper left", frameon=True)

    save_path = os.path.join(result_dir, "Figure_3_Result.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.show()
    plt.close()  # Close to free memory if running in loop


def plot_narma_from_file(result_dir, plot_data):
    """
    Generates Figure 4 (Time Series) using saved plot_data.
    """
    print(f"Plotting NARMA Results from {result_dir}...")

    y_test = np.array(plot_data["y_test"])
    y_pred = np.array(plot_data["y_pred"])
    mse = plot_data["mse"]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Zoom relative to TEST set start
    start_step = 50
    end_step = 250

    if end_step > len(y_test):
        end_step = len(y_test)

    t = np.arange(500 + start_step, 500 + end_step)

    ax.plot(
        t,
        y_test[start_step:end_step],
        color="#808080",
        linestyle="-",
        label="Target",
        linewidth=2,
    )
    ax.plot(
        t,
        y_pred[start_step:end_step],
        color="g",
        linestyle="-",
        label="Prediction",
        linewidth=2,
    )

    ax.set_title("NARMA10 Prediction (Test Phase)", fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left")

    # MSE Box
    ax.text(
        0.95,
        0.95,
        f"MSE: {mse:.2e}",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "gray"},
    )

    save_path = os.path.join(result_dir, "Figure_4_Result.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.show()
    plt.close()


def plot_from_dir(result_dir):
    """
    Plot results for a specific directory.
    """
    plot_data_path = os.path.join(result_dir, "plot_data.json")
    config_path = os.path.join(result_dir, "config.json")

    if not os.path.exists(plot_data_path) or not os.path.exists(config_path):
        print(f"Skipping plot: plot_data.json or config.json missing in {result_dir}")
        return

    config = load_json(config_path)
    plot_data = load_json(plot_data_path)

    # Determine Task
    task = config.get("task", config.get("experiment", {}).get("task"))

    if task == "nonlinear":
        plot_nonlinear_from_file(result_dir, plot_data)
    elif task == "narma":
        plot_narma_from_file(result_dir, plot_data)
    elif task in ["mackey_glass", "santa_fe"]:
        # Re-use NARMA plotter for general time series
        plot_narma_from_file(result_dir, plot_data)
    else:
        print(f"Unknown task '{task}', cannot plot.")


def plot_nonlinear_notebook(
    results_dict_mem, results_dict_nomem, title="Nonlinear Task"
):
    """
    Plots Nonlinear task results from the dictionary returned by run_experiment.
    """
    if "plot_data" not in results_dict_mem or not results_dict_nomem:
        print("No plot data found in results.")
        return

    # 1. Get Data from Models
    data_mem = results_dict_mem["plot_data"]
    data_nomem = results_dict_nomem["plot_data"]

    x = np.array(data_mem["x"])
    y_true = np.array(data_mem["y_target"])

    y_pred_mem = np.array(data_mem["y_pred"])
    mse_mem = data_mem["mse"]

    y_pred_nomem = np.array(data_nomem["y_pred"])
    mse_nomem = data_nomem["mse"]

    # 2. Compute Linear Baseline On-The-Fly
    y_pred_lin, mse_lin = compute_linear_baseline(x, y_true)

    # 3. Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Target
    ax.plot(
        x, y_true, color="black", linestyle="-", linewidth=2, label="Target ($x^4$)"
    )

    plt.axvline(
        x[int(0.9 * len(y_pred_mem))],
        color="black",
        linestyle="dotted",
        linewidth=2,
        label="Train/Test Split",
    )

    # Linear Regression
    ax.plot(
        x,
        y_pred_lin,
        color="green",
        linestyle="-.",
        linewidth=2.5,
        label=f"Linear Reg. (MSE: {mse_lin:.1e})",
    )

    # No Memristor
    # ax.plot(x, y_pred_nomem, color='orange', linestyle='--', linewidth=2.5,
    #         label=f'No Memristor (MSE: {mse_nomem:.1e})')
    ax.scatter(
        x, y_pred_nomem, color="orange", label=f"No Memristor (MSE: {mse_nomem:.1e})"
    )

    # Memristor
    # ax.plot(x, y_pred_mem, color='#1f77b4', linestyle='-.', linewidth=2.5,
    #         label=f'Q. Memristor (MSE: {mse_mem:.1e})')
    ax.scatter(
        x, y_pred_mem, color="#1f77b4", label=f"Q. Memristor (MSE: {mse_mem:.1e})"
    )

    ax.set_title("Non-linear Transformation Task ($x^4$)", fontweight="bold")
    ax.set_xlabel("Input $x$")
    ax.set_ylabel("Output $y$")
    ax.legend(loc="upper left", frameon=True)

    plt.title(title)
    plt.xlabel("Input (x)")
    plt.ylabel("Output ($x^4$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_narma_notebook(results_dict, title="NARMA Prediction"):
    """
    Plots NARMA results from the results dictionary returned by run_experiment.
    """
    if "plot_data" not in results_dict or not results_dict["plot_data"]:
        print("No plot data found in results.")
        return

    data = results_dict["plot_data"]
    y_test = np.array(data["y_test"])
    y_pred = np.array(data["y_pred"])
    mse = data["mse"]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot similar points as in the paper
    start_step = 50
    end_step = 250

    if end_step > len(y_test):
        end_step = len(y_test)

    t = np.arange(500 + start_step, 500 + end_step)

    ax.plot(
        t,
        y_test[start_step:end_step],
        color="#808080",
        linestyle="-",
        label="Target",
        linewidth=2,
    )
    ax.plot(
        t,
        y_pred[start_step:end_step],
        color="g",
        linestyle="-",
        label="Prediction",
        linewidth=2,
    )

    ax.set_title("NARMA10 Prediction (Test Phase)", fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left")

    # MSE Box
    ax.text(
        0.95,
        0.95,
        f"MSE: {mse:.2e}",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "gray"},
    )

    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    args = parser.parse_args()
    plot_from_dir(args.result_dir)
