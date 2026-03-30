import argparse
import sys

from .run_classical import main as main_classical
from .run_quantum import main as main_quantum


def train_and_evaluate(runtime_cfg, run_dir):
    raw_argv = sys.argv[1:]

    # 1. Check for --mode flag to decide which script to run
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["quantum", "classical"], default="quantum")

    args, _ = parser.parse_known_args(raw_argv)

    # 2. Dispatch to the correct script
    if args.mode == "classical":
        return main_classical(raw_argv)
    else:
        return main_quantum(raw_argv)


if __name__ == "__main__":
    train_and_evaluate(None, ".")
