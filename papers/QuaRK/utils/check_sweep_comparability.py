"""Check that the runs of each sweep differ only in their declared dimensions.

For every completed sweep given, the script flattens each run's
``config_snapshot.json`` and lists the keys whose value is not the same in all runs.
Keys outside ``DECLARED`` make the check fail. It also checks that all runs carry the
same code state (``run_status.json``) and that every metric in ``metrics.json`` is
finite. Read-only.

Usage (from the repository root)::

    python papers/QuaRK/utils/check_sweep_comparability.py papers/QuaRK/outdir/sweep_*
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Keys allowed to differ between the runs of a sweep: identity and the varied model
# settings. Everything else (data, readout, tuning, grids) must be identical.
DECLARED = {
    "seed",
    "run_id",
    "candidate.name",
    "candidate.config",
    "candidate.args",
    "description",
    "model.kind",
    "model.esn.mode",
    "model.quark.estimator",
    "model.quark.shots",
    "model.quark.num_qubits",
    "model.quark.num_reservoirs",
    "model.merlin.estimator",
    "model.merlin.shots",
    "model.merlin.distinguishable",
    "outputs.save_features",
    "data.check_release",
    "outdir",
}


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}{key}."))
        return out
    return {prefix[:-1]: json.dumps(value, sort_keys=True)}


def numbers(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from numbers(item)
    elif isinstance(value, float):
        yield value


def check(sweep_dir: Path) -> bool:
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    configs, codes, bad_numbers = [], set(), []
    for run in status["runs"]:
        run_dir = (sweep_dir / run["metrics_path"]).parent
        configs.append(
            flatten(json.loads((run_dir / "config_snapshot.json").read_text()))
        )
        code = json.loads((run_dir / "run_status.json").read_text())["code"]
        codes.add(json.dumps(code, sort_keys=True))
        metrics = json.loads((run_dir / "metrics.json").read_text())
        if not all(math.isfinite(x) for x in numbers(metrics["tasks"])):
            bad_numbers.append(run["run_id"])
    keys = set().union(*configs)
    varying = sorted(k for k in keys if len({c.get(k) for c in configs}) > 1)
    undeclared = [k for k in varying if k not in DECLARED]
    print(f"== {status['sweep_id']}: {len(configs)} runs")
    print(f"   keys that vary: {varying}")
    print(f"   undeclared varying keys: {undeclared or 'none'}")
    print(f"   code states: {sorted(codes)}")
    print(f"   runs with non-finite metrics: {bad_numbers or 'none'}")
    return not undeclared and len(codes) == 1 and not bad_numbers


def main() -> int:
    results = [check(Path(arg)) for arg in sys.argv[1:]]
    print("PASS" if results and all(results) else "FAIL")
    return 0 if results and all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
