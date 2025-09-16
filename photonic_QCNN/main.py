#!/usr/bin/env python3
"""
Main script to run photonic QCNN experiments on all datasets.

Usage:
    python main.py --paper    # Run with paper's original implementation
    python main.py --merlin   # Run with MerLin framework implementation
"""

import argparse
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import torch

# Add parent directory to Python path so photonic_QCNN can be imported as a package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def run_merlin_experiments():
    """Run experiments using MerLin implementation"""
    print("=" * 60)
    print("Running experiments with MerLin's implementation")
    print("=" * 60)

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Import paper experiment modules
    runs_dir = os.path.join(os.path.dirname(__file__), "runs")
    sys.path.insert(0, runs_dir)

    results = {}

    print("\n" + "=" * 40)
    print("1. Running BAS dataset with MerLin implementation")
    print("=" * 40)
    try:
        # Change to runs directory and execute BAS paper experiment
        original_dir = os.getcwd()
        os.chdir(runs_dir)
        # Create a new execution context with the correct working directory
        exec_globals = {
            "__file__": os.path.join(runs_dir, "run_BAS.py"),
            "__name__": "__main__",
        }
        exec(open("run_BAS.py").read(), exec_globals)
        os.chdir(original_dir)
        results["BAS"] = "MerLin BAS experiment completed"
        print("MerLin BAS experiment completed")
    except Exception as e:
        # Restore original directory on error
        if "original_dir" in locals():
            os.chdir(original_dir)
        print(f"MerLin BAS experiment failed: {str(e)}")
        print(f"Error occurred in file: {os.path.join(runs_dir, 'run_BAS.py')}")
        print("Full traceback:")
        traceback.print_exc()
        results["BAS"] = None

    print("\n" + "=" * 40)
    print("2. Running Custom BAS dataset with MerLin implementation")
    print("=" * 40)
    try:
        # Change to runs directory and execute Custom BAS paper experiment
        original_dir = os.getcwd()
        os.chdir(runs_dir)
        # Create a new execution context with the correct working directory
        exec_globals = {
            "__file__": os.path.join(runs_dir, "run_custom_BAS.py"),
            "__name__": "__main__",
        }
        exec(open("run_custom_BAS.py").read(), exec_globals)
        os.chdir(original_dir)
        results["Custom_BAS"] = "MerLin Custom BAS experiment completed"
        print("MerLin Custom BAS experiment completed")
    except Exception as e:
        # Restore original directory on error
        if "original_dir" in locals():
            os.chdir(original_dir)
        print(f"MerLin Custom BAS experiment failed: {str(e)}")
        print(f"Error occurred in file: {os.path.join(runs_dir, 'run_custom_BAS.py')}")
        print("Full traceback:")
        traceback.print_exc()
        results["Custom_BAS"] = None

    print("\n" + "=" * 40)
    print("3. Running MNIST dataset with MerLin implementation")
    print("=" * 40)
    try:
        # Change to runs directory and execute MNIST paper experiment
        original_dir = os.getcwd()
        os.chdir(runs_dir)
        # Create a new execution context with the correct working directory
        exec_globals = {
            "__file__": os.path.join(runs_dir, "run_MNIST.py"),
            "__name__": "__main__",
        }
        exec(open("run_MNIST.py").read(), exec_globals)
        os.chdir(original_dir)
        results["MNIST"] = "MerLin MNIST experiment completed"
        print("MerLin MNIST experiment completed")
    except Exception as e:
        # Restore original directory on error
        if "original_dir" in locals():
            os.chdir(original_dir)
        print(f"MerLin MNIST experiment failed: {str(e)}")
        print(f"Error occurred in file: {os.path.join(runs_dir, 'run_MNIST.py')}")
        print("Full traceback:")
        traceback.print_exc()
        results["MNIST"] = None

    return results


def run_paper_experiments():
    """Run experiments using paper's original implementation"""
    print("=" * 60)
    print("Running experiments with Paper's original implementation")
    print("=" * 60)

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Import paper experiment modules
    runs_dir = os.path.join(os.path.dirname(__file__), "runs")
    sys.path.insert(0, runs_dir)

    results = {}

    print("\n" + "=" * 40)
    print("1. Running BAS dataset with Paper implementation")
    print("=" * 40)
    try:
        # Change to runs directory and execute BAS paper experiment
        original_dir = os.getcwd()
        os.chdir(runs_dir)
        # Create a new execution context with the correct working directory
        exec_globals = {"__file__": os.path.join(runs_dir, "run_BAS_paper.py")}
        exec(open("run_BAS_paper.py").read(), exec_globals)
        os.chdir(original_dir)
        results["BAS"] = "Paper BAS experiment completed"
        print("BAS paper experiment completed")
    except Exception as e:
        # Restore original directory on error
        if "original_dir" in locals():
            os.chdir(original_dir)
        print(f"BAS paper experiment failed: {str(e)}")
        print(f"Error occurred in file: {os.path.join(runs_dir, 'run_BAS_paper.py')}")
        print("Full traceback:")
        traceback.print_exc()
        results["BAS"] = None

    print("\n" + "=" * 40)
    print("2. Running Custom BAS dataset with Paper implementation")
    print("=" * 40)
    try:
        # Change to runs directory and execute Custom BAS paper experiment
        original_dir = os.getcwd()
        os.chdir(runs_dir)
        # Create a new execution context with the correct working directory
        exec_globals = {"__file__": os.path.join(runs_dir, "run_custom_BAS_paper.py")}
        exec(open("run_custom_BAS_paper.py").read(), exec_globals)
        os.chdir(original_dir)
        results["Custom_BAS"] = "Paper Custom BAS experiment completed"
        print("Custom BAS paper experiment completed")
    except Exception as e:
        # Restore original directory on error
        if "original_dir" in locals():
            os.chdir(original_dir)
        print(f"Custom BAS paper experiment failed: {str(e)}")
        print(
            f"Error occurred in file: {os.path.join(runs_dir, 'run_custom_BAS_paper.py')}"
        )
        print("Full traceback:")
        traceback.print_exc()
        results["Custom_BAS"] = None

    print("\n" + "=" * 40)
    print("3. Running MNIST dataset with Paper implementation")
    print("=" * 40)
    try:
        # Change to runs directory and execute MNIST paper experiment
        original_dir = os.getcwd()
        os.chdir(runs_dir)
        # Create a new execution context with the correct working directory
        exec_globals = {"__file__": os.path.join(runs_dir, "run_MNIST_paper.py")}
        exec(open("run_MNIST_paper.py").read(), exec_globals)
        os.chdir(original_dir)
        results["MNIST"] = "Paper MNIST experiment completed"
        print("MNIST paper experiment completed")
    except Exception as e:
        # Restore original directory on error
        if "original_dir" in locals():
            os.chdir(original_dir)
        print(f"MNIST paper experiment failed: {str(e)}")
        print(f"Error occurred in file: {os.path.join(runs_dir, 'run_MNIST_paper.py')}")
        print("Full traceback:")
        traceback.print_exc()
        results["MNIST"] = None

    return results


def print_summary(results, implementation_type):
    """Print a summary of all experiment results"""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT SUMMARY - {implementation_type.upper()} IMPLEMENTATION")
    print("=" * 60)

    for dataset, result in results.items():
        if result is None:
            print(f"X {dataset}: FAILED")
        elif implementation_type == "merlin" and isinstance(result, dict):
            print(
                f"✓ {dataset}: SUCCESS - Test Accuracy: {result['final_test_acc']:.4f}"
            )
        else:
            print(f"✓ {dataset}: SUCCESS")

    success_count = sum(1 for r in results.values() if r is not None)
    total_count = len(results)
    print(
        f"\nOverall: {success_count}/{total_count} experiments completed successfully"
    )

    if implementation_type == "merlin":
        print("\nResults saved to respective directories in ./results/")
    else:
        print("\nResults saved to respective directories in ./results/*_paper/")


def main():
    """Main function to parse arguments and run experiments"""
    parser = argparse.ArgumentParser(
        description="Run photonic QCNN experiments on all datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      python main.py --paper     Run with paper's original implementation
      python main.py --merlin    Run with MerLin framework implementation

    The script will run experiments on all three datasets:
    - Bars and Stripes (BAS): 4x4 images
    - Custom Bars and Stripes (Custom BAS): 4x4 images with noise
    - MNIST: 8x8 binary classification (digits 0 vs 1)
            """,
    )

    # Create mutually exclusive group for implementation choice
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--paper",
        action="store_true",
        help="Run experiments using paper's original implementation",
    )
    group.add_argument(
        "--merlin",
        action="store_true",
        help="Run experiments using MerLin framework implementation",
    )

    args = parser.parse_args()

    # Print header
    print("Photonic Quantum Convolutional Neural Network Experiments")
    print(
        "Paper: Photonic Quantum Convolutional Neural Networks with Adaptive State Injection"
    )
    print("Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Check CUDA availability
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("Running on CPU")

    # Run experiments based on chosen implementation
    if args.paper:
        results = run_paper_experiments()
        print_summary(results, "paper")

    elif args.merlin:
        results = run_merlin_experiments()
        print_summary(results, "merlin")

    print(
        f"\nAll experiments finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("Check the results/ directory for detailed outputs and trained models.")


if __name__ == "__main__":
    main()
