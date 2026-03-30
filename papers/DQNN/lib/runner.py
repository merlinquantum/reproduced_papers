"""
Command-line runner for DQNN experiment variants.

This module parses CLI arguments and dispatches to the selected experiment
routine (default, bond dimension, or ablation).
"""

import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

try:
    from papers.DQNN.lib.ablation_exp import run_ablation_exp
    from papers.DQNN.lib.bond_dimension_exp import run_bond_dimension_exp
    from papers.DQNN.lib.default_exp import run_default_exp
    from papers.DQNN.utils.utils import parse_args
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from papers.DQNN.lib.ablation_exp import run_ablation_exp
    from papers.DQNN.lib.bond_dimension_exp import run_bond_dimension_exp
    from papers.DQNN.lib.default_exp import run_default_exp
    from papers.DQNN.utils.utils import parse_args

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _parse_bond_dimensions(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        parts = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
        return [int(part) for part in parts]
    return value


def train_and_evaluate(cfg, run_dir: Path) -> None:
    exp_to_run = cfg.get("exp_to_run", "DEFAULT")
    generate_graph = not cfg.get("dont_generate_graph", False)
    bond_dimensions_to_test = _parse_bond_dimensions(cfg.get("bond_dimensions_to_test"))

    if exp_to_run == "DEFAULT":
        print("Running the DEFAULT experiment")
        run_default_exp(
            bond_dim=cfg.get("bond_dim", 7),
            num_training_rounds=cfg.get("num_training_rounds", 2),
            num_epochs=cfg.get("num_epochs", 5),
            classical_epochs=cfg.get("classical_epochs", 5),
            pruning=cfg.get("pruning", False),
            pruning_amount=cfg.get("pruning_amount", 0.5),
            weight_sharing=cfg.get("weight_sharing", False),
            shared_rows=cfg.get("shared_rows", 10),
            qu_train_with_cobyla=cfg.get("qu_train_with_cobyla", False),
            num_qnn_train_step=cfg.get("num_qnn_train_step", 12),
            generate_graph=generate_graph,
            run_dir=run_dir,
        )
    elif exp_to_run == "BOND":
        print("Running the BOND experiment")
        run_bond_dimension_exp(
            bond_dimensions_to_test=bond_dimensions_to_test or list(range(1, 11)),
            num_training_rounds=cfg.get("num_training_rounds", 2),
            num_epochs=cfg.get("num_epochs", 5),
            qu_train_with_cobyla=cfg.get("qu_train_with_cobyla", False),
            num_qnn_train_step=cfg.get("num_qnn_train_step", 12),
            generate_graph=generate_graph,
            run_dir=run_dir,
        )
    elif exp_to_run == "ABLATION":
        print("Running the ABLATION experiment")
        run_ablation_exp(
            bond_dimensions_to_test=bond_dimensions_to_test or list(range(1, 11)),
            num_training_rounds=cfg.get("num_training_rounds", 2),
            num_epochs=cfg.get("num_epochs", 5),
            qu_train_with_cobyla=cfg.get("qu_train_with_cobyla", False),
            num_qnn_train_step=cfg.get("num_qnn_train_step", 12),
            generate_graph=generate_graph,
            run_dir=run_dir,
        )
    else:
        raise NameError("No experiment with that name")


def main():
    """
    Entry point for running the selected experiment.

    Returns
    -------
    None
    """
    args = parse_args()

    if args.exp_to_run == "DEFAULT":
        print("Running the DEFAULT experiment")
        run_default_exp(
            bond_dim=args.bond_dim,
            num_training_rounds=args.num_training_rounds,
            num_epochs=args.num_epochs,
            classical_epochs=args.classical_epochs,
            pruning=args.pruning,
            pruning_amount=args.pruning_amount,
            weight_sharing=args.weight_sharing,
            shared_rows=args.shared_rows,
            qu_train_with_cobyla=args.qu_train_with_cobyla,
            num_qnn_train_step=args.num_qnn_train_step,
            generate_graph=not args.dont_generate_graph,
        )
    elif args.exp_to_run == "BOND":
        print("Running the BOND experiment")
        run_bond_dimension_exp(
            bond_dimensions_to_test=args.bond_dimensions_to_test,
            num_training_rounds=args.num_training_rounds,
            num_epochs=args.num_epochs,
            qu_train_with_cobyla=args.qu_train_with_cobyla,
            num_qnn_train_step=args.num_qnn_train_step,
            generate_graph=not args.dont_generate_graph,
        )
    elif args.exp_to_run == "ABLATION":
        print("Running the ABLATION experiment")
        run_ablation_exp(
            bond_dimensions_to_test=args.bond_dimensions_to_test,
            num_training_rounds=args.num_training_rounds,
            num_epochs=args.num_epochs,
            qu_train_with_cobyla=args.qu_train_with_cobyla,
            num_qnn_train_step=args.num_qnn_train_step,
            generate_graph=not args.dont_generate_graph,
        )
    else:
        raise NameError("No experiment with that name")


if __name__ == "__main__":
    main()
