#!/usr/bin/env python3
"""Reproduce the classical MLP baseline for the HQNN study."""
# ruff:  noqa: E402

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from models.classical_baseline import (
    MLP,
    BaselineConfig,
    evaluate_architecture,
    generate_mlp_architectures,
)

# noqa: E402
from utils.io import save_experiment_results  # noqa: E402
from utils.training import count_parameters  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classical MLP baseline runner")
    parser.add_argument(
        "--features", type=str, default=None, help="Comma separated feature counts"
    )
    parser.add_argument(
        "--samples", type=int, default=1875, help="Number of dataset samples"
    )
    parser.add_argument("--classes", type=int, default=3, help="Number of classes")
    parser.add_argument(
        "--repetitions", type=int, default=5, help="Repetitions per architecture"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="Accuracy threshold for early stop",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/classical_baseline.json",
        help="Output JSON file",
    )
    return parser


def _parse_features(arg: str | None) -> Sequence[int]:
    if not arg:
        return [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    return [int(token.strip()) for token in arg.split(",") if token.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    feature_grid = _parse_features(args.features)
    output_path = Path(args.out)

    for feature_dim in feature_grid:
        print(f"\nEvaluating feature_dim={feature_dim}")
        for hidden_dims in generate_mlp_architectures():
            cfg = BaselineConfig(
                nb_features=feature_dim,
                hidden_dims=hidden_dims,
                nb_classes=args.classes,
                nb_samples=args.samples,
                repetitions=args.repetitions,
                lr=args.lr,
                batch_size=args.batch_size,
            )

            mean_acc, std_acc = evaluate_architecture(cfg)
            model = MLP(cfg.nb_features, cfg.hidden_dims, cfg.nb_classes)
            param_count = count_parameters(model)

            result = {
                "dataset": "spiral",
                "nb_features": feature_dim,
                "hidden_dims": list(hidden_dims),
                "nb_classes": args.classes,
                "samples": args.samples,
                "repetitions": args.repetitions,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "param_count": param_count,
            }

            run_count = save_experiment_results(result, output_path)
            print(
                f"Arch {hidden_dims} -> {mean_acc:.2f} ± {std_acc:.2f} "
                f"(params={param_count}, recorded_run={run_count})"
            )

            if mean_acc >= args.threshold:
                print(
                    f"Threshold {args.threshold} reached, moving to next feature size."
                )
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
