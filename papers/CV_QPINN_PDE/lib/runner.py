"""Shared-runtime entry point for the CV-QPINN reproduction.

Dispatches across the experiment kinds based on ``cfg['experiment']``:

* ``poisson_qpinn``  : 1D Poisson with the CV-QPINN consistency-loss scheme.
* ``poisson_pinn``   : matched classical FFN PINN baseline on Poisson.
* ``heat_qpinn``     : 1D heat equation with CV-QPINN.
* ``heat_pinn``      : matched classical FFN PINN on the heat equation.
* ``poisson_merlin`` : MerLin photonic-linear-optics adaptation of the
                       consistency-loss PINN on Poisson (see lib.merlin_pinn).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .data import HeatProblem, PoissonProblem, regular_grid_2d, sobol_1d
from .losses import (
    heat_nested_loss,
    heat_total_loss,
    poisson_nested_loss,
    poisson_total_loss,
)
from .metrics import summarise
from .pinn_baseline import FCNN, hidden_layers_for_param_count
from .qpinn_model import QPINN, QPINNConfig
from .training import train_heat, train_poisson

_LOG = logging.getLogger(__name__)


def _build_qpinn(cfg: dict) -> QPINN:
    model_cfg = cfg["model"]
    qcfg = QPINNConfig(
        n_qumodes=model_cfg.get("n_qumodes", 2),
        n_multi_layers=model_cfg["n_multi_layers"],
        n_single_layers=model_cfg["n_single_layers"],
        cutoff=model_cfg["cutoff"],
        active_sd=model_cfg.get("active_sd", 0.001),
        passive_sd=model_cfg.get("passive_sd", 0.1),
        seed=cfg.get("seed", 42),
    )
    return QPINN(qcfg)


def _run_poisson_qpinn(cfg: dict, run_dir: Path) -> dict:
    problem = PoissonProblem()
    torch.manual_seed(cfg.get("seed", 42))
    model = _build_qpinn(cfg)
    n_train = cfg["training"]["collocation_points"]
    x_coll = sobol_1d(n_train, problem.x_min, problem.x_max, seed=cfg.get("seed", 42))
    x_bc_left = torch.tensor([problem.x_min], dtype=torch.float64)
    x_bc_right = torch.tensor([problem.x_max], dtype=torch.float64)
    lambdas = cfg["training"]["lambdas"]
    history_path = run_dir / "history.json"
    schedule = cfg["training"].get("lr_schedule")
    loss_fn = (poisson_nested_loss if cfg["training"].get("use_nested_loss")
               else poisson_total_loss)
    _LOG.info("QPINN trainable params: %d (loss=%s)", model.n_trainable(),
              loss_fn.__name__)
    result = train_poisson(
        model, x_coll, (x_bc_left, x_bc_right),
        loss_fn=loss_fn,
        lr=cfg["training"]["lr"], epochs=cfg["training"]["epochs"],
        lambdas=lambdas, history_path=history_path,
        log_every=cfg["training"].get("log_every", 50),
        lr_schedule=schedule,
    )
    x_eval = torch.linspace(problem.x_min, problem.x_max, 200, dtype=torch.float64)
    with torch.no_grad():
        u_pred, _, trace = model(x_eval)
    u_ref = problem.analytic(x_eval)
    metrics = summarise(u_pred, u_ref)
    metrics["mean_trace"] = float(trace.mean().item())
    artefacts = {
        "x": x_eval.tolist(),
        "u_pred": u_pred.tolist(),
        "u_ref": u_ref.tolist(),
    }
    (run_dir / "predictions.json").write_text(json.dumps(artefacts))
    summary = {
        "experiment": "poisson_qpinn",
        "n_params": model.n_trainable(),
        "metrics": metrics,
        "best": result["best"],
        "wall_time_sec": result["wall_time_sec"],
        "cfg": cfg,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    torch.save(model.state_dict(), run_dir / "model.pt")
    return summary


def _run_poisson_pinn(cfg: dict, run_dir: Path) -> dict:
    problem = PoissonProblem()
    torch.manual_seed(cfg.get("seed", 42))
    target = cfg["model"]["target_param_count"]
    hidden_layers = hidden_layers_for_param_count(target_params=target, in_features=1)
    model = FCNN(in_features=1, hidden_layers=hidden_layers).to(torch.float64)
    _LOG.info("PINN baseline trainable params: %d (target %d)", model.n_trainable(), target)
    n_train = cfg["training"]["collocation_points"]
    x_coll = sobol_1d(n_train, problem.x_min, problem.x_max, seed=cfg.get("seed", 42))
    x_bc_left = torch.tensor([problem.x_min], dtype=torch.float64)
    x_bc_right = torch.tensor([problem.x_max], dtype=torch.float64)
    lambdas = cfg["training"]["lambdas"]
    history_path = run_dir / "history.json"
    schedule = cfg["training"].get("lr_schedule")
    result = train_poisson(
        model, x_coll, (x_bc_left, x_bc_right),
        loss_fn=poisson_total_loss,
        lr=cfg["training"]["lr"], epochs=cfg["training"]["epochs"],
        lambdas=lambdas, history_path=history_path,
        log_every=cfg["training"].get("log_every", 50),
        lr_schedule=schedule,
    )
    x_eval = torch.linspace(problem.x_min, problem.x_max, 200, dtype=torch.float64)
    with torch.no_grad():
        u_pred, _, _ = model(x_eval)
    u_ref = problem.analytic(x_eval)
    metrics = summarise(u_pred, u_ref)
    artefacts = {
        "x": x_eval.tolist(),
        "u_pred": u_pred.tolist(),
        "u_ref": u_ref.tolist(),
    }
    (run_dir / "predictions.json").write_text(json.dumps(artefacts))
    summary = {
        "experiment": "poisson_pinn",
        "n_params": model.n_trainable(),
        "hidden_layers": hidden_layers,
        "metrics": metrics,
        "best": result["best"],
        "wall_time_sec": result["wall_time_sec"],
        "cfg": cfg,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    torch.save(model.state_dict(), run_dir / "model.pt")
    return summary


def _run_heat_qpinn(cfg: dict, run_dir: Path) -> dict:
    problem = HeatProblem()
    torch.manual_seed(cfg.get("seed", 42))
    model = _build_qpinn(cfg)
    nx = cfg["training"]["nx"]
    nt = cfg["training"]["nt"]
    xt_coll = regular_grid_2d(nx, nt, problem.x_min, problem.x_max,
                              problem.t_min, problem.t_max)
    n_ic = cfg["training"]["n_ic"]
    xt_ic = torch.stack([
        torch.linspace(problem.x_min, problem.x_max, n_ic, dtype=torch.float64),
        torch.zeros(n_ic, dtype=torch.float64),
    ], dim=1)
    T_ic = problem.initial(xt_ic[:, 0])
    n_bc = cfg["training"]["n_bc"]
    t_bc = torch.linspace(problem.t_min, problem.t_max, n_bc, dtype=torch.float64)
    xt_bc_left = torch.stack([torch.full((n_bc,), problem.x_min, dtype=torch.float64),
                              t_bc], dim=1)
    xt_bc_right = torch.stack([torch.full((n_bc,), problem.x_max, dtype=torch.float64),
                               t_bc], dim=1)
    lambdas = cfg["training"]["lambdas"]
    history_path = run_dir / "history.json"
    schedule = cfg["training"].get("lr_schedule")
    loss_fn = (heat_nested_loss if cfg["training"].get("use_nested_loss")
               else heat_total_loss)
    _LOG.info("QPINN trainable params: %d (loss=%s)", model.n_trainable(),
              loss_fn.__name__)
    result = train_heat(
        model, xt_coll, xt_ic, T_ic, (xt_bc_left, xt_bc_right),
        loss_fn=loss_fn, lr=cfg["training"]["lr"],
        pretrain_epochs=cfg["training"]["pretrain_epochs"],
        epochs=cfg["training"]["epochs"], lambdas=lambdas,
        alpha=problem.alpha, history_path=history_path,
        log_every=cfg["training"].get("log_every", 50),
        lr_schedule=schedule,
    )
    nx_ref = cfg["training"].get("eval_nx", 41)
    nt_ref = cfg["training"].get("eval_nt", 11)
    ref = problem.reference_solution(nx=nx_ref, nt=nt_ref)
    x_arr = torch.tensor(ref["x"])
    t_arr = torch.tensor(ref["t"])
    xx, tt = torch.meshgrid(x_arr, t_arr, indexing="xy")
    eval_xt = torch.stack([xx.flatten(), tt.flatten()], dim=1).to(torch.float64)
    with torch.no_grad():
        u_pred, _, trace = model(eval_xt)
    u_pred = u_pred.reshape(nt_ref, nx_ref)
    u_ref = torch.tensor(ref["T"], dtype=torch.float64)
    metrics = summarise(u_pred, u_ref)
    metrics["mean_trace"] = float(trace.mean().item())
    artefacts = {
        "x": ref["x"].tolist(),
        "t": ref["t"].tolist(),
        "u_pred": u_pred.tolist(),
        "u_ref": u_ref.tolist(),
    }
    (run_dir / "predictions.json").write_text(json.dumps(artefacts))
    summary = {
        "experiment": "heat_qpinn",
        "n_params": model.n_trainable(),
        "metrics": metrics,
        "best": result["best"],
        "wall_time_sec": result["wall_time_sec"],
        "cfg": cfg,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    torch.save(model.state_dict(), run_dir / "model.pt")
    return summary


def _run_heat_pinn(cfg: dict, run_dir: Path) -> dict:
    problem = HeatProblem()
    torch.manual_seed(cfg.get("seed", 42))
    target = cfg["model"]["target_param_count"]
    hidden_layers = hidden_layers_for_param_count(target_params=target, in_features=2)
    model = FCNN(in_features=2, hidden_layers=hidden_layers).to(torch.float64)
    _LOG.info("PINN baseline trainable params: %d (target %d)", model.n_trainable(), target)
    nx = cfg["training"]["nx"]
    nt = cfg["training"]["nt"]
    xt_coll = regular_grid_2d(nx, nt, problem.x_min, problem.x_max,
                              problem.t_min, problem.t_max)
    n_ic = cfg["training"]["n_ic"]
    xt_ic = torch.stack([
        torch.linspace(problem.x_min, problem.x_max, n_ic, dtype=torch.float64),
        torch.zeros(n_ic, dtype=torch.float64),
    ], dim=1)
    T_ic = problem.initial(xt_ic[:, 0])
    n_bc = cfg["training"]["n_bc"]
    t_bc = torch.linspace(problem.t_min, problem.t_max, n_bc, dtype=torch.float64)
    xt_bc_left = torch.stack([torch.full((n_bc,), problem.x_min, dtype=torch.float64),
                              t_bc], dim=1)
    xt_bc_right = torch.stack([torch.full((n_bc,), problem.x_max, dtype=torch.float64),
                               t_bc], dim=1)
    lambdas = cfg["training"]["lambdas"]
    history_path = run_dir / "history.json"
    schedule = cfg["training"].get("lr_schedule")
    result = train_heat(
        model, xt_coll, xt_ic, T_ic, (xt_bc_left, xt_bc_right),
        loss_fn=heat_total_loss, lr=cfg["training"]["lr"],
        pretrain_epochs=cfg["training"]["pretrain_epochs"],
        epochs=cfg["training"]["epochs"], lambdas=lambdas,
        alpha=problem.alpha, history_path=history_path,
        log_every=cfg["training"].get("log_every", 50),
        lr_schedule=schedule,
    )
    nx_ref = cfg["training"].get("eval_nx", 41)
    nt_ref = cfg["training"].get("eval_nt", 11)
    ref = problem.reference_solution(nx=nx_ref, nt=nt_ref)
    x_arr = torch.tensor(ref["x"])
    t_arr = torch.tensor(ref["t"])
    xx, tt = torch.meshgrid(x_arr, t_arr, indexing="xy")
    eval_xt = torch.stack([xx.flatten(), tt.flatten()], dim=1).to(torch.float64)
    with torch.no_grad():
        u_pred, _, _ = model(eval_xt)
    u_pred = u_pred.reshape(nt_ref, nx_ref)
    u_ref = torch.tensor(ref["T"], dtype=torch.float64)
    metrics = summarise(u_pred, u_ref)
    artefacts = {
        "x": ref["x"].tolist(),
        "t": ref["t"].tolist(),
        "u_pred": u_pred.tolist(),
        "u_ref": u_ref.tolist(),
    }
    (run_dir / "predictions.json").write_text(json.dumps(artefacts))
    summary = {
        "experiment": "heat_pinn",
        "n_params": model.n_trainable(),
        "hidden_layers": hidden_layers,
        "metrics": metrics,
        "best": result["best"],
        "wall_time_sec": result["wall_time_sec"],
        "cfg": cfg,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    torch.save(model.state_dict(), run_dir / "model.pt")
    return summary


def _run_poisson_merlin(cfg: dict, run_dir: Path) -> dict:
    """MerLin photonic linear-optics adaptation of the consistency-loss PINN."""
    from .merlin_pinn import run_merlin_poisson

    return run_merlin_poisson(cfg, run_dir)


_EXPERIMENTS = {
    "poisson_qpinn": _run_poisson_qpinn,
    "poisson_pinn": _run_poisson_pinn,
    "heat_qpinn": _run_heat_qpinn,
    "heat_pinn": _run_heat_pinn,
    "poisson_merlin": _run_poisson_merlin,
}


def train_and_evaluate(cfg: dict, run_dir: Path) -> dict:
    experiment = cfg.get("experiment", "poisson_qpinn")
    if experiment not in _EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{experiment}'. "
                         f"Available: {sorted(_EXPERIMENTS)}")
    _LOG.info("Running experiment '%s' in %s", experiment, run_dir)
    return _EXPERIMENTS[experiment](cfg, run_dir)
