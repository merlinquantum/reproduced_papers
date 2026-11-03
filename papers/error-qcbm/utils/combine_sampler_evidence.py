"""Combine separate full-budget profile audits for the production sampler gate."""

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = next(path for path in PROJECT.parents if (path / "implementation.py").is_file())
sys.path[:0] = [str(PROJECT), str(ROOT)]

import torch  # noqa: E402
from lib.sampler_validation import combine_evidence  # noqa: E402
from lib.training import write_json  # noqa: E402


def main():
    """Validate complete, compatible evidence and write its combined artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    combined = combine_evidence(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, combined)
    print(f"Validated four-condition 50M evidence: {args.output}")


if __name__ == "__main__":
    main()
