"""Runtime entry point: ``train_and_evaluate(cfg, run_dir)``.

One run solves ``problem.instances`` independent instances of one problem family
at one size with one method, and writes

    metrics.json   one row per instance
    summary.json   the aggregate the paper's tables report
    run.log        via the shared runtime

The paper's tables report "% optimal" and "average % error" over a set of random
instances, so those two numbers are what ``summary.json`` carries.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from lib.baselines import BASELINES
from lib.bbs import BosonicBinarySolver
from lib.metrics import relative_error
from lib.problems import build_problem
from lib.tbi import candidate_budget

logger = logging.getLogger(__name__)


def _require(cfg, *path):
    node = cfg
    for key in path:
        if key not in node:
            raise ValueError(f"missing required config key: {'.'.join(path)}")
        node = node[key]
    return node


def solve_instance(cfg, seed):
    """Solve one instance and return its result row."""
    problem = cfg["problem"]
    method = cfg["method"]
    m = problem["m"]
    delays = tuple(cfg["circuit"]["delays"])
    training = cfg["training"]

    instance, cost_batch, optimum = build_problem(
        problem["family"], m, seed, **problem.get("kwargs", {})
    )
    budget = candidate_budget(m, delays, training["updates"], training["samples"])

    started = time.time()
    if method["name"] == "bbs":
        solver = BosonicBinarySolver(
            m,
            delays=delays,
            updates=training["updates"],
            samples=training["samples"],
            lr_theta=training["lr_theta"],
            lr_alpha=training["lr_alpha"],
            shift=training["shift"],
            shift_scale=training.get("shift_scale", 1.0),
            source=method.get("source", "boson"),
            rate_scale=method.get("rate_scale", 1.0),
            topology=method.get("topology", "chain"),
            seed=seed,
        )
        result = solver.solve(cost_batch)
    elif method["name"] == "bbs_merlin":
        # Imported here rather than at module scope: MerLin is only needed by this
        # branch, and the GPU runner must not acquire a MerLin dependency through
        # an import chain it never uses.
        from lib.bbs_merlin import MerlinBinarySolver

        # exact-gradient MerLin variant; enumerates the cost table to form the
        # expectation, so it is limited to small m and its two counters mean
        # different things -- see lib/bbs_merlin.MerlinBinarySolver.solve
        solver = MerlinBinarySolver(
            m,
            delays=delays,
            updates=training["updates"],
            flip_samples=training["samples"],
            lr_theta=training["lr_theta"],
            lr_alpha=training["lr_alpha"],
            topology=method.get("topology", "chain"),
            seed=seed,
        )
        result = solver.solve(cost_batch)
    elif method["name"] in BASELINES:
        # every baseline gets exactly the solver's candidate budget, which is the
        # only way the comparison tests the paper's claim rather than its compute
        result = BASELINES[method["name"]](
            cost_batch, m, budget, np.random.default_rng(seed)
        )
        result.setdefault("mean_clicks", float("nan"))
    else:
        raise ValueError(f"unknown method {method['name']!r}")

    return {
        "instance_seed": seed,
        "family": problem["family"],
        "m": m,
        "method": method["name"],
        "source": method.get("source") if method["name"] == "bbs" else None,
        "rate_scale": method.get("rate_scale", 1.0)
        if method["name"] == "bbs"
        else None,
        "best_cost": result["best_cost"],
        "optimum": float(optimum),
        "found_optimum": bool(abs(result["best_cost"] - optimum) < 1e-9),
        "relative_error_percent": relative_error(result["best_cost"], optimum),
        "evaluations": result["evaluations"],
        "budget": int(budget),
        "circuit_evaluations": result.get("circuit_evaluations"),
        "simulator_cost_evaluations": result.get("simulator_cost_evaluations"),
        "mean_clicks": result.get("mean_clicks", float("nan")),
        "seconds": round(time.time() - started, 2),
    }


def train_and_evaluate(cfg, run_dir: Path) -> None:
    """Solve every instance of one experiment and write the run artifacts."""
    problem = _require(cfg, "problem")
    _require(cfg, "circuit", "delays")
    _require(cfg, "training", "updates")
    _require(cfg, "method", "name")

    run_dir = Path(run_dir)
    instances = problem["instances"]
    base = problem.get("instance_seed_base", 0)
    delays = tuple(cfg["circuit"]["delays"])
    budget = candidate_budget(
        problem["m"], delays, cfg["training"]["updates"], cfg["training"]["samples"]
    )

    logger.info(
        "%s m=%d, %d instances, method=%s, budget %d candidates = %.3g of the 2^%d space",
        problem["family"],
        problem["m"],
        instances,
        cfg["method"]["name"],
        budget,
        budget / 2 ** problem["m"],
        problem["m"],
    )

    metrics_path = run_dir / "metrics.json"
    # Resume: an interrupted run leaves a complete metrics.json for the instances
    # it finished, because the file is rewritten after every one. Long runs get
    # killed here (a cloud container reclaims background processes when the
    # session idles), so resuming is not a nicety.
    rows = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else []
    )
    done = {row["instance_seed"] for row in rows}
    if done:
        logger.info(
            "resuming: %d of %d instances already recorded", len(done), instances
        )
    for index in range(instances):
        if base + index in done:
            continue
        row = solve_instance(cfg, base + index)
        rows.append(row)
        # written every instance so a long run is never lost and can be inspected live
        metrics_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        logger.info(
            "instance %d/%d seed=%d best=%.4f optimum=%.4f %s (%.1fs)",
            index + 1,
            instances,
            row["instance_seed"],
            row["best_cost"],
            row["optimum"],
            "OPTIMAL"
            if row["found_optimum"]
            else f"err {row['relative_error_percent']:.2f}%",
            row["seconds"],
        )

    errors = np.array([r["relative_error_percent"] for r in rows], dtype=float)
    summary = {
        "family": problem["family"],
        "m": problem["m"],
        "method": cfg["method"]["name"],
        "source": cfg["method"].get("source")
        if cfg["method"]["name"] == "bbs"
        else None,
        "rate_scale": cfg["method"].get("rate_scale", 1.0)
        if cfg["method"]["name"] == "bbs"
        else None,
        "instances": len(rows),
        "percent_optimal": 100.0 * float(np.mean([r["found_optimum"] for r in rows])),
        "mean_percent_error": float(np.nanmean(errors)),
        "std_percent_error": float(np.nanstd(errors, ddof=1)) if len(rows) > 1 else 0.0,
        "candidate_budget": int(budget),
        "space_coverage": budget / 2 ** problem["m"],
        "mean_clicks": float(np.nanmean([r["mean_clicks"] for r in rows])),
        "median_seconds_per_instance": float(np.median([r["seconds"] for r in rows])),
        "training": cfg["training"],
        "circuit": cfg["circuit"],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info(
        "SUMMARY %s m=%d %s: %.1f%% optimal, mean error %.3f%% over %d instances",
        summary["family"],
        summary["m"],
        summary["method"],
        summary["percent_optimal"],
        summary["mean_percent_error"],
        summary["instances"],
    )
