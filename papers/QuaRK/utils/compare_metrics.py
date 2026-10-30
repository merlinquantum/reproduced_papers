"""Compare the task values of two runs' ``metrics.json``, ignoring timings.

A regression check: two runs of the same config and seed must give identical values.
Every number under ``tasks`` is compared exactly, except keys named ``seconds``.
Read-only.

Usage (from the repository root)::

    python papers/QuaRK/utils/compare_metrics.py <run_dir_a> <run_dir_b>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key != "seconds":
                out.update(flatten(item, f"{prefix}{key}."))
        return out
    if isinstance(value, list):
        out = {}
        for index, item in enumerate(value):
            out.update(flatten(item, f"{prefix}{index}."))
        return out
    return {prefix[:-1]: value}


def main() -> int:
    runs = [Path(arg) for arg in sys.argv[1:]]
    if len(runs) != 2:
        raise SystemExit("usage: compare_metrics.py <run_dir_a> <run_dir_b>")
    values = [
        flatten(json.loads((run / "metrics.json").read_text())["tasks"]) for run in runs
    ]
    keys = sorted(set(values[0]) | set(values[1]))
    differing = [k for k in keys if values[0].get(k) != values[1].get(k)]
    print(f"compared values: {len(keys)}; differing: {len(differing)}")
    for key in differing[:20]:
        print(f"  {key}: {values[0].get(key)!r} != {values[1].get(key)!r}")
    print("IDENTICAL" if not differing else "DIFFERENT")
    return 0 if not differing else 1


if __name__ == "__main__":
    raise SystemExit(main())
