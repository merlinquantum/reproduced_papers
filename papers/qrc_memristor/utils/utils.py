import argparse
import json
import logging
import os
import sys
from typing import Any

import numpy as np
import torch


def setup_logging(output_dir: str) -> None:
    """
    Configures the logging system to write logs to 'classical_experiment.log' and stdout.

    Args:
        output_dir (str): The directory where the log file will be created.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "classical_experiment.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )


def save_json(data: Any, filepath: str) -> None:
    """
    Saves data to a JSON file, handling NumPy types and PyTorch tensors.

    Args:
        data (Any): The dictionary or list to save.
        filepath (str): The full path to the output JSON file.
    """

    def convert(o: Any) -> Any:
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.item() if o.numel() == 1 else o.tolist()
        return str(o)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4, default=convert)


def load_config_file(config_path: str) -> dict[str, Any]:
    """
    Loads configuration from a JSON file and flattens 'experiment' and 'data' sections.

    Args:
        config_path (str): Path to the JSON configuration file.

    Returns:
        Dict[str, Any]: A dictionary of configuration parameters. Returns an empty dict if file not found.
    """
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found.")
        return {}

    with open(config_path) as f:
        config = json.load(f)

    flat_config = {}
    if "experiment" in config:
        flat_config.update(config["experiment"])
    if "data" in config:
        flat_config.update(config["data"])

    for k, v in config.items():
        if k not in ["experiment", "data"] and not isinstance(v, dict):
            flat_config[k] = v

    return flat_config


def get_clean_config_classical(args: argparse.Namespace) -> dict[str, Any]:
    """
    Extracts relevant configuration parameters from parsed arguments for saving.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        Dict[str, Any]: A sanitized dictionary of the configuration used.
    """
    config = {
        "task": args.task,
        "model_type": args.model_type,
        "model_class": f"ClassicalBenchmark_{args.model_type}",
        "n_runs": args.n_runs,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "device": args.device,
        "output_dir": args.output_dir,
        "exp_name": args.exp_name,
    }
    if args.task == "narma":
        config.update(
            {
                "data_size": args.data_size,
                "washout": args.washout,
                "train_len": args.train_len,
            }
        )
    elif args.task == "mackey_glass":
        config.update(
            {
                "data_size": args.data_size,
                "washout": args.washout,
                "train_len": args.train_len,
            }
        )
    elif args.task == "santa_fe":
        config.update(
            {
                "data_size": args.data_size,
                "washout": args.washout,
                "train_len": args.train_len,
            }
        )

    return config


def get_clean_config_quantum(args: argparse.Namespace) -> dict[str, Any]:
    """
    Extracts relevant configuration parameters from parsed arguments to create
    a clean dictionary for saving.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        Dict[str, Any]: A dictionary containing the sanitized configuration used for the run.
    """
    config = {
        "task": args.task,
        "seed": args.seed,
        "device": args.device,
        "exp_name": args.exp_name,
        "output_dir": args.output_dir,
    }
    if args.task in ["narma", "mackey_glass", "santa_fe"]:
        config.update(
            {
                "model": "QuantumReservoirFeedbackTimeSeries",
                "n_runs": args.n_runs,
                "memory": args.memory,
                "data_size": args.data_size,
                "washout": args.washout,
                "train_len": args.train_len,
            }
        )
    elif args.task == "nonlinear":
        config.update(
            {
                "model_type": args.model_type,
                "n_runs": args.n_runs,
                "epochs": args.epochs,
                "lr": args.lr,
                "nonlinear_data_size": 85,
                "split_ratio": 0.9,
            }
        )
        if args.model_type == "memristor":
            config["memory"] = args.memory

    return config
