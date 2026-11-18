"""Time vectorised DQN training on one device for several agent counts.

Used to decide whether the remote L4 GPU or CPU workers run each experiment
(LOG.md, Remote-execution decision). Each measurement trains ``--agents``
agents of the given config for ``--episodes`` episodes and reports the cost
per lockstep iteration and per agent-environment-step.

Example (from the paper folder)::

    python utils/benchmark_devices.py --configs configs/e2e3_cartpole_optimal_full_original.json \\
        --devices cpu --agents 1 10 60 --episodes 20 --label laptop --output-dir outdir/benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR.parent.parent))

from lib.dqn import DQNTrainer  # noqa: E402
from lib.experiment_logging import configure_logging, write_json  # noqa: E402

logger = logging.getLogger("benchmark_devices")


def benchmark(
    config_path: Path, device: str, agent_counts: list[int], episodes: int, dtype: str
) -> dict:
    """Run one timing per agent count and return the measurements.

    Parameters
    ----------
    config_path : pathlib.Path
        Experiment config (complete JSON as in ``configs/``).
    device : str
        Torch device string.
    agent_counts : list of int
        Numbers of agents trained together.
    episodes : int
        Episodes per agent (all agents run exactly this many: solving is
        disabled by an unreachable threshold).
    dtype : str
        ``float32`` or ``float64``.

    Returns
    -------
    dict
        Device information and one record per agent count.
    """
    base_cfg = json.loads(config_path.read_text())
    real_dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
    records = []
    for n_agents in agent_counts:
        cfg = deepcopy(base_cfg)
        cfg["agents"]["seeds"] = list(range(n_agents))
        cfg["training"]["max_episodes"] = episodes
        cfg["environment"]["solve"]["mean_threshold"] = float("inf")
        trainer = DQNTrainer(cfg, torch.device(device), real_dtype)
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = trainer.train()
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        agent_steps = sum(result.total_env_steps)
        record = {
            "agents": n_agents,
            "episodes_per_agent": episodes,
            "seconds": elapsed,
            "lockstep_iterations": result.lockstep_iterations,
            "agent_env_steps": agent_steps,
            "ms_per_lockstep_iteration": 1e3 * elapsed / result.lockstep_iterations,
            "ms_per_agent_env_step": 1e3 * elapsed / agent_steps,
        }
        logger.info("BENCHMARK_COMPLETED | %s", record)
        records.append(record)
    return {
        "config": str(config_path),
        "device": device,
        "device_name": torch.cuda.get_device_name(0)
        if device.startswith("cuda")
        else platform.processor(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "dtype": dtype,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--agents", type=int, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--label", required=True, help="Machine label used in output file names"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(args.output_dir / f"{args.label}_benchmark.log")
    logger.info(
        "BENCHMARK_STARTED | label=%s | configs=%s | devices=%s | agents=%s",
        args.label,
        [str(path) for path in args.configs],
        args.devices,
        args.agents,
    )
    for device in args.devices:
        for config_path in args.configs:
            results = benchmark(
                config_path, device, args.agents, args.episodes, args.dtype
            )
            device_tag = device.replace(":", "")
            output = (
                args.output_dir / f"{args.label}_{device_tag}_{config_path.stem}.json"
            )
            write_json(output, results)
            logger.info("BENCHMARK_WRITTEN | path=%s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
