"""Shared-runner entrypoint (see repo-root ``implementation.py``).

Adapts the shared runtime's resolved config into a call to
:func:`lib.benchmark.run_sweep_from_config`, so ``python implementation.py
--paper ObliQ_photonic_QUBO --config configs/obliq_maxclique.json`` and
``python -m lib.benchmark sweep --config configs/obliq_maxclique.json`` run the
exact same sweep and land in the exact same ``results/<hash>/`` directory.

For a single instance -- no sweep, no config file -- use
``python -m lib.benchmark run ...`` directly; the shared runner has no
single-instance mode.
"""

from __future__ import annotations

from pathlib import Path

from lib.benchmark import run_sweep_from_config

#: ObliQ's own experiment-identity shape (what every ``configs/*.json`` file has
#: always contained standalone). Everything the shared runtime injects on top
#: (``seed``, ``dtype``, ``device``, ``logging``, ``data_root``, ``outdir``) is
#: deliberately excluded here so the content hash -- and therefore
#: ``results/<hash>/`` -- never moves. See ``tests/test_config.py``, which pins
#: the hash each shipped config must keep producing.
_EXPERIMENT_KEYS = (
    "problem_type",
    "solver",
    "name",
    "sweep",
    "provider",
    "backend",
    "solver_options",
    "output",
)


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Run an ObliQ sweep from the shared runtime's resolved config.

    ``run_dir`` -- the shared runtime's own timestamped bookkeeping folder
    (``config_snapshot.json`` / ``run.log``) -- is unused here: the sweep is
    content-addressed by :func:`lib.benchmark.run_sweep_from_config` instead,
    independent of which entrypoint triggered it.

    ``cfg["seed"]`` (set by the global ``--seed`` flag, or its default of 1337)
    is a *fallback* for ``sweep.seed``, not an override: an experiment config
    that pins its own ``sweep.seed`` -- every named ``configs/*.json`` does --
    is left untouched, so results stay reproducible regardless of what
    ``--seed`` happens to be. It only fills in when a config leaves
    ``sweep.seed`` unset, which today is just ``configs/defaults.json``.
    """
    experiment_cfg = {key: cfg[key] for key in _EXPERIMENT_KEYS if key in cfg}
    sweep = experiment_cfg.setdefault("sweep", {})
    if sweep.get("seed") is None and cfg.get("seed") is not None:
        sweep["seed"] = cfg["seed"]
    run_sweep_from_config(experiment_cfg)
