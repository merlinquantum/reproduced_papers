"""Check that every included sweep run used the config now in ``configs/``.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/check_config_drift.py \
        papers/fermi_dirac_neurons/outdir/sweeps/<sweep> [...]

For every COMPLETED run, the run's ``config_snapshot.json`` is compared with
``configs/defaults.json`` overlaid by ``configs/<config>.json`` as they are
now. Keys the shared runtime injects per run (seed, dtype, device, logging,
data_root, outdir, and experiment.run_id / sweep_id / repetition) are left
out. Prints one line per sweep and every differing
key path; exits 1 if any run differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))
sys.path.insert(0, str(PROJECT_DIR.parents[1]))

from fermi_dirac_neurons.utils.run_sweep import resolved_config  # noqa: E402

RUNTIME_KEYS = {"seed", "dtype", "device", "logging", "data_root", "outdir"}
# The shared runtime writes the sweep identity of each run into ``experiment``.
RUNTIME_EXPERIMENT_KEYS = {"run_id", "sweep_id", "repetition"}


def differing_paths(first, second, prefix: str = "") -> list[str]:
    """Return the key paths where two JSON values differ."""
    if isinstance(first, dict) and isinstance(second, dict):
        paths = []
        for key in sorted(set(first) | set(second)):
            if not prefix and key in RUNTIME_KEYS:
                continue
            if prefix == "experiment." and key in RUNTIME_EXPERIMENT_KEYS:
                continue
            if key not in first or key not in second:
                paths.append(f"{prefix}{key} (missing on one side)")
                continue
            paths.extend(differing_paths(first[key], second[key], f"{prefix}{key}."))
        return paths
    return [] if first == second else [prefix.rstrip(".")]


def main() -> int:
    """Compare every completed run's snapshot with the current config."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    any_drift = False
    for sweep_dir in args.sweep_dirs:
        status = json.loads((sweep_dir / "sweep_status.json").read_text())
        checked, drifted = 0, {}
        for run in status["runs"]:
            if run["status"] != "COMPLETED":
                continue
            snapshot = json.loads((sweep_dir / run["config_path"]).read_text())
            paths = differing_paths(snapshot, resolved_config(run["config"]))
            checked += 1
            if paths:
                drifted[run["run_id"]] = paths
        any_drift = any_drift or bool(drifted)
        print(
            f"{sweep_dir.name}: {checked} completed runs checked, "
            f"{len(drifted)} differ from the current configs"
        )
        for run_id, paths in list(drifted.items())[:5]:
            print(f"  {run_id}: {', '.join(paths[:6])}")
    return 1 if any_drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
