import argparse
import datetime as dt
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestCentroid
from src.classifier import QuantumNearestCentroid, MLQuantumNearestCentroid
from sklearn.datasets import load_iris
import torchvision
from torchvision import transforms
import torch

# Import config utilities
from default_config import default_config, load_config, deep_update


# -----------------------------------------------------
# Helper utilities
# -----------------------------------------------------

def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------
# Main experiment logic
# -----------------------------------------------------

def run_subset_experiment(X, y, classes, n_samples, n_repeats=10, n_components=8, run_dir=None):
    """Runs repeated subset classification experiments with PCA and various classifiers."""
    accs = []
    ml_accs = []
    c_accs = []


    for r in range(n_repeats):
        mask = np.isin(y, classes)
        X_sub, y_sub = X[mask], y[mask]

        # balance classes & subsample
        X_sel, _, y_sel, _ = train_test_split(
            X_sub, y_sub, train_size=2 * n_samples, stratify=y_sub
        )

        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_sel, y_sel, test_size=0.5, stratify=y_sel
        )

        # Apply PCA
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Scaling
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca)
        X_test_scaled = scaler.transform(X_test_pca)

        # Classifiers
        c = NearestCentroid()
        clf = QuantumNearestCentroid()
        clf_ml = MLQuantumNearestCentroid(n=n_components)

        # Fit and evaluate
        c.fit(X_train_pca, y_train)
        clf.fit(X_train_scaled, y_train)
        clf_ml.fit(X_train_scaled, y_train)

        c_acc = c.score(X_test_pca, y_test)
        acc = clf.score(X_test_scaled, y_test)
        ml_acc = clf_ml.score(X_test_scaled, y_test)

        c_accs.append(c_acc)
        accs.append(acc)
        ml_accs.append(ml_acc)

    # Aggregate results
    result = {
        "classes": classes,
        "n_samples": n_samples,
        "acc_mean": np.mean(accs),
        "acc_std": np.std(accs),
        "ml_acc_mean": np.mean(ml_accs),
        "ml_acc_std": np.std(ml_accs),
        "c_acc_mean": np.mean(c_accs),
        "c_acc_std": np.std(c_accs),
    }

    # Save per-experiment results if run_dir provided
    if run_dir is not None:
        out_file = run_dir / f"results_{'_'.join(map(str, classes))}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

    return result


def train_and_evaluate(cfg, run_dir: Path) -> None:
    """Run training/evaluation logic depending on dataset."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment with dataset: {cfg['dataset']['name']}")

    dataset_name = cfg["dataset"]["name"].lower()
    n_repeats = cfg["training"]["n_repeats"]
    n_components = cfg["dataset"]["n_components"]
    experiments = cfg["experiments"]

    # -----------------------------------------------------
    # Dataset loading
    # -----------------------------------------------------
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 -> 784
        ])
        mnist = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        X = mnist.data.numpy().reshape(len(mnist), -1)
        y = mnist.targets.numpy()

    elif dataset_name == "iris":
        iris = load_iris()
        X, y = iris.data, iris.target

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # -----------------------------------------------------
    # Run experiments
    # -----------------------------------------------------
    all_results = []
    for exp in experiments:
        result = run_subset_experiment(
            X=X,
            y=y,
            classes=exp["classes"],
            n_samples=exp["n_samples"],
            n_repeats=n_repeats,
            n_components=n_components,
            run_dir=run_dir
        )
        print(result)
        all_results.append(result)
        logger.info(f"Finished experiment: {exp['classes']} -> mean acc: {result['acc_mean']:.3f}")

    # -----------------------------------------------------
    # Save global results
    # -----------------------------------------------------
    results_path = run_dir / "summary_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"All experiments completed. Results saved to {results_path}")


# -----------------------------------------------------
# CLI and runner
# -----------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper reproduction runner")
    p.add_argument("--config", type=str, help="Path to JSON config", default=None)
    p.add_argument("--seed", type=int, help="Random seed", default=None)
    p.add_argument("--outdir", type=str, help="Base output directory", default=None)
    p.add_argument("--device", type=str, help="Device string (cpu, cuda:0, mps)", default=None)
    return p


def configure_logging(level: str = "info", log_file: Path | None = None) -> None:
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    log_level = level_map.get(str(level).lower(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)


def resolve_config(args: argparse.Namespace):
    cfg = default_config()

    # Load from file if provided
    if args.config:
        file_cfg = load_config(Path(args.config))
        cfg = deep_update(cfg, file_cfg)

    # Apply CLI overrides
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.outdir is not None:
        cfg["outdir"] = args.outdir
    if args.device is not None:
        cfg["device"] = args.device


    return cfg


def main(argv: list[str] | None = None) -> int:
    configure_logging("info")
    script_dir = Path(__file__).resolve().parent
    if Path.cwd().resolve() != script_dir:
        logging.info("Switching working directory to %s", script_dir)
        os.chdir(script_dir)

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = resolve_config(args)
    setup_seed(cfg["seed"])

    # Prepare output dir
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_out = Path(cfg["outdir"])
    run_dir = base_out / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(cfg.get("logging", {}).get("level", "info"), run_dir / "run.log")

    (run_dir / "config_snapshot.json").write_text(json.dumps(cfg, indent=2))
    print(cfg.keys())
    train_and_evaluate(cfg, run_dir)

    logging.info("Finished. Artifacts in: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
