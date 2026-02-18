import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Any, Optional

import numpy as np
import torch
from sklearn.linear_model import Ridge

# Path fix: 3 levels up (lib -> qmem -> reproduced_papers -> ROOT)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import utils.create_plots as plot_module
from lib.quantum_reservoir import (
    QuantumReservoirFeedback,
    QuantumReservoirFeedbackTimeSeries,
    QuantumReservoirNoMem,
)
from lib.training import (
    extract_features_sequence,
    fit_readout_mse,
    get_param_snapshot,
    run_narma_multiple,
    train_sequence,
)
from utils.datasets import get_dataset
from utils.utils import (
    get_clean_config_quantum,
    load_config_file,
    save_json,
    setup_logging,
)


def run_narma_task(args: argparse.Namespace) -> dict[str, Any]:
    """
    Executes the NARMA benchmark task using the Quantum Reservoir with Feedback.

    Args:
        args (argparse.Namespace): Parsed arguments containing model hyperparameters
            (memory, n_runs, data_size, washout, etc.).

    Returns:
        Dict[str, Any]: A dictionary containing results:
            - 'mean_mse' (float): Average Mean Squared Error across runs.
            - 'std_mse' (float): Standard deviation of MSE.
            - 'all_mses' (List[float]): List of MSEs for each run.
            - 'plot_data' (Dict): Data required to generate plots (best targets/predictions).
    """
    logging.info("=== Starting NARMA Task ===")
    logging.info(f"Model: QuantumReservoirFeedbackTimeSeries | Memory: {args.memory}")

    def model_builder():
        return QuantumReservoirFeedbackTimeSeries(
            input_dim=1, n_modes=3, memory=args.memory
        )

    mean_mse, std_mse, all_mses, best_y_target, best_y_pred = run_narma_multiple(
        model_builder=model_builder,
        n_runs=args.n_runs,
        N=args.data_size,
        washout=args.washout,
        train=args.train_len,
        base_seed=args.seed,
        device=args.device,
    )

    logging.info(f"FINAL RESULT >> Mean MSE: {mean_mse:.6f} +/- {std_mse:.6f}")

    plot_data = {
        "y_test": best_y_target,
        "y_pred": best_y_pred,
        "mse": np.min(all_mses),
    }

    return {
        "mean_mse": mean_mse,
        "std_mse": std_mse,
        "all_mses": all_mses,
        "plot_data": plot_data,
    }


def run_general_timeseries_task(args: argparse.Namespace) -> dict[str, Any]:
    """
    Executes general time-series prediction tasks (Mackey-Glass, Santa Fe)
    using the Quantum Reservoir with Feedback.

    Args:
        args (argparse.Namespace): Parsed arguments containing task name and hyperparameters.

    Returns:
        Dict[str, Any]: A dictionary containing results:
            - 'mean_mse' (float): Average Mean Squared Error.
            - 'std_mse' (float): Standard deviation of MSE.
            - 'all_mses' (List[float]): MSEs for each run.
            - 'plot_data' (Dict): Best prediction data for plotting.
    """
    logging.info(f"=== Starting Task: {args.task} ===")
    logging.info(f"Model: QuantumReservoirFeedbackTimeSeries | Memory: {args.memory}")

    all_mses = []
    best_plot_data = None
    min_mse = float("inf")

    for i in range(args.n_runs):
        seed = args.seed + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        u_raw, y_raw = get_dataset(args.task, args.data_size)
        u_tensor = (
            torch.tensor(u_raw, dtype=torch.float32).reshape(1, -1, 1).to(args.device)
        )
        y_tensor = (
            torch.tensor(y_raw, dtype=torch.float32).reshape(1, -1, 1).to(args.device)
        )

        washout = args.washout
        train_len = args.train_len
        if args.task == "santa_fe" and train_len > u_tensor.shape[1] - 50:
            train_len = int(u_tensor.shape[1] * 0.7)

        model = QuantumReservoirFeedbackTimeSeries(
            input_dim=1, n_modes=3, memory=args.memory
        ).to(args.device)

        with torch.no_grad():
            X_feat = extract_features_sequence(model, u_tensor, memory=True)

        X_train = X_feat[0, washout:train_len, :].cpu().numpy()
        y_train = y_tensor[0, washout:train_len, :].cpu().numpy()
        X_test = X_feat[0, train_len:, :].cpu().numpy()
        y_test = y_tensor[0, train_len:, :].cpu().numpy()

        ridge = Ridge(alpha=1e-5)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        mse = np.mean((y_pred - y_test) ** 2)

        logging.info(f"Run {i + 1}/{args.n_runs} MSE: {mse:.6f}")
        all_mses.append(mse)

        if mse < min_mse:
            min_mse = mse
            best_plot_data = {"y_test": y_test, "y_pred": y_pred, "mse": mse}

    mean_mse = np.mean(all_mses)
    std_mse = np.std(all_mses)
    logging.info(f"FINAL RESULT >> Mean MSE: {mean_mse:.6f} +/- {std_mse:.6f}")

    return {
        "mean_mse": mean_mse,
        "std_mse": std_mse,
        "all_mses": all_mses,
        "plot_data": best_plot_data,
    }


def run_nonlinear_task(args: argparse.Namespace) -> dict[str, Any]:
    """
    Executes the Non-Linear function prediction task (e.g., y = x^4).
    Compares models with and without memristive feedback.

    Args:
        args (argparse.Namespace): Parsed arguments containing model type ('memristor' or 'nomem'),
            training epochs, learning rate, etc.

    Returns:
        Dict[str, Any]: A dictionary containing results:
            - 'mean_mse' (float): Average test MSE.
            - 'std_mse' (float): Standard deviation of MSE.
            - 'all_mses' (List[float]): MSEs for all runs.
            - 'best_run' (Dict): Details of the best performing run (index, params).
            - 'plot_data' (Dict): Data for plotting the best fit.
    """
    logging.info("=== Starting Non-Linear Function Task ===")

    def build_model_fn():
        if args.model_type == "nomem":
            return QuantumReservoirNoMem(input_dim=1, n_modes=3)
        else:
            return QuantumReservoirFeedback(input_dim=1, n_modes=3, memory=args.memory)

    best_run = {"mse": float("inf"), "run_index": -1, "params": None}
    best_plot_data = None
    all_mses = []

    for i in range(args.n_runs):
        current_seed = args.seed + i
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)

        n_samples = 85
        x = torch.linspace(0, 1, n_samples).reshape(-1, 1)
        y = x.flatten() ** 4
        split = int(0.9 * len(x))
        x_dev, y_dev = x.to(args.device), y.to(args.device)
        x_train, y_train = x_dev[:split], y_dev[:split]

        model = build_model_fn().to(args.device)
        mem_flag = args.model_type == "memristor"

        train_sequence(
            model,
            x_train,
            y_train,
            model_name=f"Run {i + 1}",
            n_epochs=args.epochs,
            lr=args.lr,
            memory=mem_flag,
        )

        with torch.no_grad():
            X_feat = extract_features_sequence(model, x_dev, memory=mem_flag)

        test_mse = fit_readout_mse(X_feat, y_dev, split=0.9)
        logging.info(f"[Run {i:02d}] Seed={current_seed} | Test MSE={test_mse:.6f}")
        all_mses.append(test_mse)

        if test_mse < best_run["mse"]:
            best_run["mse"] = test_mse
            best_run["run_index"] = i
            best_run["params"] = get_param_snapshot(model)

            if X_feat.ndim == 3:
                X_feat = X_feat[0]
            X_train_np = X_feat[:split].cpu().numpy()
            y_train_np = y_dev[:split].cpu().numpy()
            ridge = Ridge(alpha=1e-5)
            ridge.fit(X_train_np, y_train_np)
            y_pred_full = ridge.predict(X_feat.cpu().numpy())

            best_plot_data = {
                "x": x_dev.cpu().numpy().flatten(),
                "y_target": y_dev.cpu().numpy().flatten(),
                "y_pred": y_pred_full,
                "mse": test_mse,
            }

    mean_mse = np.mean(all_mses)
    std_mse = np.std(all_mses)
    logging.info(f"FINAL RESULT >> Mean MSE: {mean_mse:.6f} +/- {std_mse:.6f}")

    return {
        "mean_mse": mean_mse,
        "std_mse": std_mse,
        "all_mses": all_mses,
        "best_run": best_run,
        "plot_data": best_plot_data,
    }


def main(argv: Optional[list[str]] = None) -> int:
    """
    Main entry point for running quantum reservoir experiments.
    Parses arguments, sets up logging, dispatches tasks, and saves results.

    Args:
        argv (Optional[List[str]], optional): Command line arguments.
            Defaults to None (uses sys.argv).

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument(
        "--task", type=str, choices=["narma", "nonlinear", "mackey_glass", "santa_fe"]
    )
    parser.add_argument("--model-type", type=str, choices=["memristor", "nomem"])
    parser.add_argument("--n-runs", type=int)
    parser.add_argument("--memory", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--data-size", type=int)
    parser.add_argument("--washout", type=int)
    parser.add_argument("--train-len", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--exp-name", type=str)
    parser.add_argument("--plot", action="store_true")

    args, unknown = parser.parse_known_args(argv)

    defaults = {
        "task": "narma",
        "model_type": "memristor",
        "n_runs": 10,
        "memory": 5,
        "seed": 42,
        "device": "cpu",
        "data_size": 1000,
        "washout": 20,
        "train_len": 480,
        "epochs": 200,
        "lr": 0.01,
        "output_dir": "./results",
        "exp_name": "",
    }

    config_vals = {}
    if args.config:
        if os.path.exists(args.config):
            print(f"Loading configuration from {args.config}...")
            config_vals = load_config_file(args.config)
            if "size" in config_vals and "data_size" not in config_vals:
                config_vals["data_size"] = config_vals.pop("size")
        else:
            print(f"Warning: Config file {args.config} not found. Using defaults.")

    for key, default_val in defaults.items():
        arg_val = getattr(args, key, None)
        if arg_val is not None:
            pass
        elif key in config_vals:
            setattr(args, key, config_vals[key])
        else:
            setattr(args, key, default_val)

    if not args.task:
        # It's possible --task is missing if running purely from defaults, but usually required.
        print(
            "Error: The --task argument is required (or must be specified in --config)."
        )
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.task in ["narma", "mackey_glass", "santa_fe"]:
        exp_tag = args.exp_name if args.exp_name else f"{args.task}_mem{args.memory}"
    else:
        mem_str = f"_mem{args.memory}" if args.model_type == "memristor" else "_nomem"
        exp_tag = args.exp_name if args.exp_name else f"nonlinear{mem_str}"

    folder_name = f"{exp_tag}_{timestamp}"
    save_path = os.path.join(args.output_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)

    setup_logging(save_path)
    logging.info(f"Experiment initialized at {save_path}")
    if args.config:
        logging.info(f"Loaded config from: {args.config}")

    if args.task == "narma":
        results = run_narma_task(args)
    elif args.task in ["mackey_glass", "santa_fe"]:
        results = run_general_timeseries_task(args)
    elif args.task == "nonlinear":
        results = run_nonlinear_task(args)

    plot_data = results.pop("plot_data", None)

    save_json(get_clean_config_quantum(args), os.path.join(save_path, "config.json"))
    save_json(results, os.path.join(save_path, "metrics.json"))

    if plot_data:
        save_json(plot_data, os.path.join(save_path, "plot_data.json"))
        if args.plot:
            logging.info("Generating plots from plot_data.json...")
            try:
                plot_module.plot_from_dir(save_path)
            except Exception as e:
                logging.error(f"Failed to generate plots: {e}")

    if args.task == "nonlinear" and results.get("best_run"):
        save_json(
            results["best_run"]["params"], os.path.join(save_path, "best_params.json")
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
