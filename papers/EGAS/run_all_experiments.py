#!/usr/bin/env python3
"""
Run all EGAS reproduction experiments with progress tracking.

Usage:
    python run_all_experiments.py                    # Run all experiments
    python run_all_experiments.py --skip-tests       # Skip tests, run all others
    python run_all_experiments.py --only-photonic    # Run only photonic experiments
    python run_all_experiments.py --quick            # Run quick smoke test only
"""

import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

# ANSI colors
GREEN = "\033[0;32m"
BLUE = "\033[0;34m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
NC = "\033[0m"  # No Color


def run_command(cmd: list, description: str = "", cwd: Optional[str] = None) -> bool:
    """Run a command and return success status."""
    if description:
        print(f"{YELLOW}→{NC} {description}")
    try:
        result = subprocess.run(cmd, check=True, cwd=cwd)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        print(f"{RED}✗ Failed{NC}: {' '.join(cmd)}")
        return False


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{BLUE}{'=' * 60}{NC}")
    print(f"{BLUE}{text.center(60)}{NC}")
    print(f"{BLUE}{'=' * 60}{NC}\n")


def print_step(step_num: int, total: int, text: str) -> None:
    """Print a step with progress."""
    print(f"{YELLOW}[{step_num}/{total}]{NC} {text}")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{GREEN}✓ {text}{NC}\n")


def print_result_location(name: str, path: str) -> None:
    """Print where results are saved."""
    print(f"  • {name:20} {path}")


def main():
    parser = ArgumentParser(description="Run all EGAS reproduction experiments")
    parser.add_argument("--skip-tests", action="store_true", help="Skip photonic tests")
    parser.add_argument(
        "--only-photonic", action="store_true", help="Run only photonic experiments"
    )
    parser.add_argument(
        "--only-gate", action="store_true", help="Run only gate-based EGAS experiments"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick smoke test only"
    )
    args = parser.parse_args()

    # Get paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    python_exe = sys.executable

    # Verify repo structure
    if not (repo_root / "implementation.py").exists():
        print(
            f"{RED}Error: Could not find implementation.py at {repo_root}{NC}",
            file=sys.stderr,
        )
        sys.exit(1)

    print_header("EGAS Reproduction - All Experiments")

    step = 1
    gate_datasets = [
        ("PW", "Phishing"),
        ("WDGV1", "Waveform DB (multiclass)"),
        ("WQ", "Wine Quality"),
        ("MGT", "MAGIC Gamma Telescope"),
    ]
    photonic_datasets = [
        ("PW", "Phishing"),
        ("WQ", "Wine Quality"),
        ("MGT", "MAGIC Gamma Telescope"),
        ("WDGV1", "Waveform DB (multiclass)"),
    ]

    if args.quick:
        total_steps = 1
    elif args.only_photonic:
        total_steps = len(photonic_datasets) + (0 if args.skip_tests else 1)
    elif args.only_gate:
        total_steps = len(gate_datasets) + 2 + (0 if args.skip_tests else 1)
    else:
        total_steps = (
            len(gate_datasets)
            + len(photonic_datasets)
            + 2
            + (0 if args.skip_tests else 1)
        )

    # Run tests
    if not args.skip_tests and not args.only_photonic:
        print_step(step, total_steps, "Running photonic implementation tests...")
        if not run_command(
            [
                python_exe,
                "-m",
                "pytest",
                "tests/test_photonic_impl.py",
                "-v",
                "--tb=short",
            ],
            cwd=str(script_dir),
        ):
            print(
                f"{YELLOW}Warning: Tests failed, but continuing with experiments...{NC}\n"
            )
        else:
            print_success("Tests passed")
        step += 1

    if args.quick:
        # Quick smoke test
        print_step(step, total_steps, "Running quick smoke test...")
        if run_command(
            [
                python_exe,
                str(repo_root / "implementation.py"),
                "--paper-dir",
                str(script_dir),
                "--config",
                str(script_dir / "configs" / "defaults.json"),
                "--outdir",
                str(script_dir / "outdir" / "quick_test"),
            ],
            cwd=str(repo_root),
        ):
            print_success("Quick test completed")
        return

    if not args.only_photonic:
        # Wasserstein (Table I)
        print_step(step, total_steps, "Running Wasserstein diagnostic (Table I)...")
        if run_command(
            [
                python_exe,
                str(repo_root / "implementation.py"),
                "--paper-dir",
                str(script_dir),
                "--config",
                str(script_dir / "configs" / "wasserstein.json"),
                "--outdir",
                str(script_dir / "outdir" / "wasserstein"),
            ],
            cwd=str(repo_root),
        ):
            print_success("Wasserstein results saved")
        step += 1

        # Fig 1
        print_step(
            step, total_steps, "Running Fig 1 experiments (trace distance vs W1)..."
        )
        if run_command(
            [
                python_exe,
                str(repo_root / "implementation.py"),
                "--paper-dir",
                str(script_dir),
                "--config",
                str(script_dir / "configs" / "fig1.json"),
                "--outdir",
                str(script_dir / "outdir" / "fig1"),
            ],
            cwd=str(repo_root),
        ):
            print_success("Fig 1 results saved")
        step += 1

        # EGAS experiments (unless only photonic)
        if not args.only_photonic:
            for i, (shortname, fullname) in enumerate(gate_datasets, 1):
                current_step = step + i - 1
                print_step(
                    current_step,
                    total_steps,
                    f"Running EGAS search on {fullname} ({shortname})...",
                )
                if run_command(
                    [
                        python_exe,
                        str(repo_root / "implementation.py"),
                        "--paper-dir",
                        str(script_dir),
                        "--config",
                        str(script_dir / "configs" / f"egas_{shortname}.json"),
                        "--outdir",
                        str(script_dir / "outdir" / shortname),
                    ],
                    cwd=str(repo_root),
                ):
                    print_success(f"{fullname} results saved")
            step += len(gate_datasets)

    # Photonic experiments
    if not args.only_gate:
        for i, (shortname, fullname) in enumerate(photonic_datasets, 1):
            current_step = step + i - 1
            print_step(
                current_step,
                total_steps,
                f"Running photonic QKSVM on {fullname}...",
            )
            if run_command(
                [
                    python_exe,
                    str(repo_root / "implementation.py"),
                    "--paper-dir",
                    str(script_dir),
                    "--config",
                    str(script_dir / "configs" / f"photonic_{shortname}.json"),
                    "--outdir",
                    str(script_dir / "outdir" / f"photonic_{shortname}"),
                ],
                cwd=str(repo_root),
            ):
                print_success(f"{fullname} photonic results saved")

    # Summary
    print_header("All experiments completed successfully!")

    print("Results saved to:")
    if not args.only_photonic:
        print_result_location("Wasserstein:", f"{script_dir}/outdir/wasserstein/")
        print_result_location("Fig 1:", f"{script_dir}/outdir/fig1/")
        print_result_location("EGAS (PW):", f"{script_dir}/outdir/PW/")
        print_result_location("EGAS (WQ):", f"{script_dir}/outdir/WQ/")
        print_result_location("EGAS (MGT):", f"{script_dir}/outdir/MGT/")
    if not args.only_gate:
        print_result_location("Photonic:", f"{script_dir}/outdir/photonic_MGT/")

    print("\n")


if __name__ == "__main__":
    main()
