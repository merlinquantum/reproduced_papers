"""QARIMA reproduction runner.

For one dataset config this:
  1. loads the series and applies the paper's train/OOS split;
  2. builds classical baselines (pmdarima auto non-seasonal = paper comparator;
     the paper's fixed order; optionally a *fair* seasonal ARIMA);
  3. fits every hard-coded candidate ``(p,d,q)`` with each refiner
     (classical OLS / gate VQC / photonic MerLin VQC), multi-seed for the VQCs;
     NOTE: quantum_acf/quantum_pacf are library functions available for order
     selection but the runner currently uses pre-configured candidate lists;
  4. runs a classical AR-order sweep (the "quantum gain is an order effect" probe);
  5. computes OOS MSE/MAPE vs the classical baseline (Diebold--Mariano p-values
     omitted for multi-step forecasts with mixed lead times, where they are
     undefined; see metrics.py for details);
  6. writes ``results.json``, ``metrics.csv``, ``forecasts.npz`` and figures.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.classical import auto_order, dynamic_forecast  # noqa: E402
from lib.data import load_series, split_series  # noqa: E402
from lib.metrics import mape, mse  # noqa: E402
from lib.qarima import LossWeights, fit_and_forecast  # noqa: E402
from lib.refiners import make_refiner  # noqa: E402
from utils import plot_qarima  # noqa: E402


def _loss_weights(cfg) -> LossWeights:
    lc = cfg.get("loss", {})
    return LossWeights(
        lambda_cos=lc.get("lambda_cos", 0.1),
        lambda_ent=lc.get("lambda_ent", 0.05),
        lambda_l2=lc.get("lambda_l2", 0.0),
        omega=lc.get("omega", 1.0),
        shots=lc.get("shots", None),
    )


def train_and_evaluate(cfg, run_dir: Path) -> None:
    log = logging.getLogger("qarima")
    t_start = time.time()

    ds = cfg["dataset"]["name"]
    y, meta = load_series(ds, data_root=cfg.get("data_root"))
    n_oos = meta["oos"]
    y_train, y_true = split_series(y, n_oos)
    log.info("Dataset %s: total=%d train=%d oos=%d", ds, y.size, y_train.size, n_oos)

    weights = _loss_weights(cfg)
    vqc = cfg.get("vqc", {})
    seeds = cfg.get("seeds", [0, 1, 2])
    refiner_names = cfg.get("refiners", ["classical", "gate", "merlin"])

    results: dict = {
        "dataset": ds,
        "unit": meta.get("unit", "value"),
        "n_total": int(y.size),
        "n_train": int(y_train.size),
        "n_oos": int(n_oos),
        "meta": {k: v for k, v in meta.items() if k not in ("oos",)},
    }

    # ---- classical baselines -------------------------------------------------
    order_auto, so_auto = auto_order(y_train)
    cpred_auto, _, _ = dynamic_forecast(y, n_oos, order_auto)
    results["classical_auto"] = {
        "order": list(order_auto),
        "mse": mse(y_true, cpred_auto),
        "mape": mape(y_true, cpred_auto),
    }
    log.info(
        "classical auto order=%s MSE=%.4f MAPE=%.4f",
        order_auto,
        results["classical_auto"]["mse"],
        results["classical_auto"]["mape"],
    )

    forecasts = {
        "y_train_tail": y_train[-min(len(y_train), 3 * n_oos) :],
        "y_true": y_true,
        "classical_auto": cpred_auto,
    }

    pbo = cfg.get("paper_baseline_order")
    if pbo:
        try:
            cpred_paper, _, _ = dynamic_forecast(y, n_oos, tuple(pbo))
            results["classical_paper"] = {
                "order": list(pbo),
                "mse": mse(y_true, cpred_paper),
                "mape": mape(y_true, cpred_paper),
            }
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"paper baseline order {pbo} failed and is required by config: {e}"
            ) from e

    sb = cfg.get("seasonal_baseline", {})
    if sb.get("enabled"):
        try:
            spred, _, _ = dynamic_forecast(
                y, n_oos, tuple(sb["order"]), tuple(sb["seasonal_order"])
            )
            results["seasonal"] = {
                "order": sb["order"],
                "seasonal_order": sb["seasonal_order"],
                "mse": mse(y_true, spred),
                "mape": mape(y_true, spred),
            }
            forecasts["seasonal"] = spred
            log.info(
                "fair seasonal ARIMA MSE=%.4f MAPE=%.4f",
                results["seasonal"]["mse"],
                results["seasonal"]["mape"],
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"seasonal baseline (order={sb.get('order')}, "
                f"seasonal_order={sb.get('seasonal_order')}) failed and is "
                f"enabled in config: {e}"
            ) from e

    # ---- candidate (p,d,q) x refiner ----------------------------------------
    best = {"gate": (np.inf, None), "merlin": (np.inf, None)}
    cand_rows = []
    for p, d, q in cfg.get("candidates", []):
        row = {"p": p, "d": d, "q": q, "refiners": {}}
        for rname in refiner_names:
            use_seeds = seeds if rname in ("gate", "merlin") else [seeds[0]]
            preds_seeds, mses, mapes = [], [], []
            for sd in use_seeds:
                refiner = make_refiner(
                    rname,
                    reps=vqc.get("reps", 1),
                    max_iter=vqc.get("max_iter", 80),
                    step_frac=vqc.get("step_frac", 1.0),
                    n_train=vqc.get("n_train"),
                )
                fr = fit_and_forecast(y, n_oos, p, d, q, refiner, weights, seed=sd)
                preds_seeds.append(fr.y_pred)
                mses.append(mse(y_true, fr.y_pred))
                mapes.append(mape(y_true, fr.y_pred))
            mean_pred = np.mean(preds_seeds, axis=0)
            # NOTE: DM p-values are omitted here. The averaged predictions mix
            # lead times 1..n_oos, making a fixed-horizon DM test undefined.
            # A proper multi-step test would require rolling-origin fixed-horizon
            # errors; see metrics.py comments for details.
            entry = {
                "mse_mean": float(np.mean(mses)),
                "mse_std": float(np.std(mses)),
                "mape_mean": float(np.mean(mapes)),
                "n_seeds": len(use_seeds),
            }
            row["refiners"][rname] = entry
            if rname in best and entry["mse_mean"] < best[rname][0]:
                best[rname] = (entry["mse_mean"], mean_pred)
        cand_rows.append(row)
        log.info(
            "  Q(%d,%d,%d) %s",
            p,
            d,
            q,
            " ".join(
                f"{r}={row['refiners'][r]['mse_mean']:.3f}" for r in refiner_names
            ),
        )
    results["candidates"] = cand_rows
    for rn in ("gate", "merlin"):
        if best[rn][1] is not None:
            forecasts[f"best_{rn}"] = best[rn][1]

    # ---- classical AR-order sweep (order-effect diagnostic) ------------------
    osw = cfg.get("order_sweep", {})
    if osw.get("enabled", True):
        d_sw, q_sw = osw.get("d", 1), osw.get("q", 0)
        sweep = []
        cls = make_refiner("classical", max_iter=1)  # OLS only (1 COBYLA step ~ OLS)
        for p in range(1, osw.get("p_max", 15) + 1):
            fr = fit_and_forecast(
                y,
                n_oos,
                p,
                d_sw,
                q_sw,
                cls,
                LossWeights(lambda_cos=0, lambda_ent=0),
                seed=0,
            )
            sweep.append({"p": p, "mse": mse(y_true, fr.y_pred)})
        results["order_sweep"] = sweep
    if cfg.get("paper_best_quantum_mse"):
        results["paper_best_quantum_mse"] = cfg["paper_best_quantum_mse"]

    results["wall_clock_s"] = round(time.time() - t_start, 1)

    # ---- persist -------------------------------------------------------------
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    np.savez(run_dir / "forecasts.npz", **forecasts)
    with (run_dir / "metrics.csv").open("w", newline="") as fh:
        wri = csv.writer(fh)
        wri.writerow(["model", "p", "d", "q", "mse", "mape"])
        wri.writerow(
            [
                "classical_auto",
                *order_auto,
                results["classical_auto"]["mse"],
                results["classical_auto"]["mape"],
            ]
        )
        if "seasonal" in results:
            wri.writerow(
                [
                    "seasonal",
                    "",
                    "",
                    "",
                    results["seasonal"]["mse"],
                    results["seasonal"]["mape"],
                ]
            )
        for r in cand_rows:
            for rn, e in r["refiners"].items():
                wri.writerow(
                    [
                        f"qarima_{rn}",
                        r["p"],
                        r["d"],
                        r["q"],
                        f"{e['mse_mean']:.4f}",
                        f"{e['mape_mean']:.5f}",
                    ]
                )

    figs = plot_qarima.make_all(run_dir, ds)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    for f in figs:
        (results_dir / f.name).write_bytes(f.read_bytes())
    log.info(
        "Done %s in %.1fs. Figures: %s",
        ds,
        results["wall_clock_s"],
        [f.name for f in figs],
    )
