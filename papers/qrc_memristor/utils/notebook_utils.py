import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from lib.datasets import generate_narma
except ModuleNotFoundError:
    # Ensure the project root is on sys.path so local `lib` imports work.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from lib.datasets import generate_narma


class NotebookArgs:
    """
    Helper class to manage experiment arguments.
    Ensures defaults match 'run_classical.py' exactly.
    """

    def __init__(
        self,
        task="narma",
        model_type="memristor",
        memory=4,
        epochs=200,
        lr=0.05,
        n_runs=10,
    ):
        # Experiment Settings
        self.task = task
        self.model_type = model_type
        self.memory = memory

        # --- CRITICAL DEFAULTS ---
        self.n_runs = n_runs  # Default: 10
        self.lr = lr  # Default: 0.05 (Crucial for classical models)
        self.epochs = epochs if epochs > 0 else 200  # Default: 200

        self.device = "cpu"  # Ensure this matches your script run
        self.seed = 42

        # Data Settings
        self.data_size = 1000
        self.washout = 20
        self.train_len = 480  # Used as Split Index [washout : train_len]

        # Logging
        self.exp_name = "notebook_run"
        self.output_dir = "./results_notebook"


def run_experiment(args, run_function):
    """
    Executes the run_function with args and returns the dictionary directly.
    """
    model_tag = "Classical" if "Classical" in str(run_function) else args.model_type
    print(f"--- Starting Experiment: {model_tag} on {args.task} ---")
    return run_function(args)


def run_classical_benchmarks(args):
    """
    Runs the 4 classical models (L, Q, L+M, Q+M) on the NARMA task.
    Logic is identical to 'run_classical.py'.
    """
    # Import locally to avoid top-level failures
    from lib.classical_models import ClassicalBenchmark

    results_summary = {}
    models_to_run = ["L", "Q", "L+M", "Q+M"]

    for m_type in models_to_run:
        type_mses = []

        for i in range(args.n_runs):
            seed = args.seed + i
            torch.manual_seed(seed)
            np.random.seed(seed)

            # 1. GENERATE DATA (Exact same function as script)
            u_tens, y_tens, _ = generate_narma(
                data_size=args.data_size, seed=seed, device=args.device
            )

            # 2. RESHAPE (1, Seq, 1)
            u_tens = u_tens.reshape(1, -1, 1).to(args.device)
            y_tens = y_tens.reshape(1, -1, 1).to(args.device)

            # 3. SLICING (Matches run_classical.py exactly)
            # Train: [washout : train_len] (Indices 20 to 480)
            u_train = u_tens[:, args.washout : args.train_len, :]
            y_train = y_tens[:, args.washout : args.train_len, :]

            # Test: [train_len : end] (Indices 480 to 1000)
            u_test = u_tens[:, args.train_len :, :]
            y_test = y_tens[:, args.train_len :, :]

            # 4. MODEL & OPTIMIZER
            model = ClassicalBenchmark(model_type=m_type, input_dim=1).to(args.device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.MSELoss()

            # 5. TRAINING LOOP
            model.train()
            for _ep in range(args.epochs):
                optimizer.zero_grad()
                pred = model(u_train)
                loss = criterion(pred, y_train)
                loss.backward()
                optimizer.step()

            # 6. EVALUATION
            model.eval()
            with torch.no_grad():
                test_pred = model(u_test)
                test_mse = torch.mean((test_pred - y_test) ** 2).item()
                type_mses.append(test_mse)

        mean = np.mean(type_mses)
        std = np.std(type_mses)
        results_summary[m_type] = {"mean": mean, "std": std}

    return results_summary
