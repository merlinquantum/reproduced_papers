#!/usr/bin/env python3
"""
Generate Figures from Experiment Results
=========================================

This utility generates figures from saved experiment results.
It can be run after experiments complete to create publication-quality figures.

Usage:
    # Generate figures for a specific run
    python utils/generate_figures.py results/train_quantum/run_20240120-123456

    # Generate figures for all runs in a directory
    python utils/generate_figures.py results/ --all

    # Regenerate specific figure types
    python utils/generate_figures.py results/attack_bim/run_xxx --type attack
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Handle imports for both direct execution and module execution
if __name__ == "__main__":
    # When run directly, add parent to path for sibling imports
    _parent = Path(__file__).resolve().parent
    if str(_parent.parent) not in sys.path:
        sys.path.insert(0, str(_parent.parent))

    from utils.plot_attacks import (
        plot_attack_from_results,
        plot_robustness_from_results,
    )
    from utils.plot_comparison import (
        plot_model_comparison_from_results,
        plot_noise_comparison_from_results,
        plot_transfer_from_results,
    )
    from utils.plot_training import plot_training_from_results
else:
    # When imported as a module
    from .plot_attacks import plot_attack_from_results, plot_robustness_from_results
    from .plot_comparison import (
        plot_model_comparison_from_results,
        plot_noise_comparison_from_results,
        plot_transfer_from_results,
    )
    from .plot_training import plot_training_from_results

logger = logging.getLogger(__name__)


def load_results(run_dir: Path) -> dict[str, Any] | None:
    """Load results from a run directory.

    Args:
        run_dir: Path to run directory

    Returns:
        Results dictionary or None if not found
    """
    # Try different result file names
    result_files = [
        "results.json",
        "training_results.json",
        "attack_results.json",
        "noise_comparison_results.json",
        "transfer_results.json",
        "comparison_results.json",
        "summary_results.json",
    ]

    results = {}
    for filename in result_files:
        filepath = run_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
                results.update(data)

    if not results:
        logger.warning(f"No results found in {run_dir}")
        return None

    return results


def load_config(run_dir: Path) -> dict[str, Any] | None:
    """Load config from a run directory.

    Args:
        run_dir: Path to run directory

    Returns:
        Config dictionary or None if not found
    """
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


def detect_experiment_type(run_dir: Path, results: dict[str, Any]) -> str:
    """Detect the type of experiment from results.

    Args:
        run_dir: Path to run directory
        results: Results dictionary

    Returns:
        Experiment type string
    """
    # Check for specific result keys
    if "noise_comparison_results.json" in [f.name for f in run_dir.glob("*.json")]:
        return "noise"
    if "transfer_results.json" in [f.name for f in run_dir.glob("*.json")]:
        return "transfer"
    if "comparison_results.json" in [f.name for f in run_dir.glob("*.json")]:
        return "comparison"

    # Check result content
    if "fooling_rate" in results or "adversarial_accuracy" in results:
        return "attack"
    if "history" in results or "train_loss" in results:
        return "training"
    if "robustness" in results:
        return "robustness"

    # Try to infer from config
    config = load_config(run_dir)
    if config:
        return config.get("experiment", "unknown")

    return "unknown"


def generate_figures_for_run(
    run_dir: Path,
    figure_types: list[str] | None = None,
    force: bool = False,
) -> list[Path]:
    """Generate figures for a single run.

    Args:
        run_dir: Path to run directory
        figure_types: List of figure types to generate (None = auto-detect)
        force: Overwrite existing figures

    Returns:
        List of generated figure paths
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return []

    results = load_results(run_dir)
    if results is None:
        return []

    config = load_config(run_dir)
    exp_type = detect_experiment_type(run_dir, results)

    generated = []

    # Auto-detect figure types if not specified
    if figure_types is None:
        if exp_type == "training" or "history" in results:
            figure_types = ["training"]
        elif exp_type == "attack":
            figure_types = ["attack"]
        elif exp_type == "noise":
            figure_types = ["noise"]
        elif exp_type == "transfer":
            figure_types = ["transfer"]
        elif exp_type == "comparison":
            figure_types = ["comparison"]
        else:
            figure_types = ["training"]  # Default

    for fig_type in figure_types:
        try:
            if fig_type == "training":
                path = plot_training_from_results(run_dir, results, config, force)
                if path:
                    generated.append(path)

            elif fig_type == "attack":
                path = plot_attack_from_results(run_dir, results, config, force)
                if path:
                    generated.append(path)

            elif fig_type == "robustness":
                path = plot_robustness_from_results(run_dir, results, config, force)
                if path:
                    generated.append(path)

            elif fig_type == "noise":
                path = plot_noise_comparison_from_results(
                    run_dir, results, config, force
                )
                if path:
                    generated.append(path)

            elif fig_type == "transfer":
                path = plot_transfer_from_results(run_dir, results, config, force)
                if path:
                    generated.append(path)

            elif fig_type == "comparison":
                path = plot_model_comparison_from_results(
                    run_dir, results, config, force
                )
                if path:
                    generated.append(path)

        except Exception as e:
            logger.error(f"Error generating {fig_type} figure: {e}")

    return generated


def generate_all_figures(
    base_dir: Path,
    force: bool = False,
) -> dict[str, list[Path]]:
    """Generate figures for all runs in a directory.

    Args:
        base_dir: Base results directory
        force: Overwrite existing figures

    Returns:
        Dictionary mapping run paths to generated figures
    """
    base_dir = Path(base_dir)
    all_generated = {}
    processed_dirs = set()

    # Find all run directories (contain results files)
    for run_dir in base_dir.rglob("run_*"):
        if run_dir.is_dir() and str(run_dir) not in processed_dirs:
            processed_dirs.add(str(run_dir))
            figures = generate_figures_for_run(run_dir, force=force)
            if figures:
                all_generated[str(run_dir)] = figures
                logger.info(f"Generated {len(figures)} figures for {run_dir}")

    return all_generated


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate figures from experiment results"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to run directory or base results directory",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate figures for all runs in directory",
    )
    parser.add_argument(
        "--type",
        type=str,
        nargs="+",
        choices=["training", "attack", "robustness", "noise", "transfer", "comparison"],
        help="Specific figure types to generate",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing figures",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.all:
        results = generate_all_figures(args.path, force=args.force)
        total = sum(len(v) for v in results.values())
        print(f"\nGenerated {total} figures across {len(results)} runs")
    else:
        figures = generate_figures_for_run(
            args.path,
            figure_types=args.type,
            force=args.force,
        )
        print(f"\nGenerated {len(figures)} figures:")
        for fig in figures:
            print(f"  - {fig}")


if __name__ == "__main__":
    main()
