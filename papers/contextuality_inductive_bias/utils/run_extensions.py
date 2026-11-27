"""Phase 5 extensions X1 (contextuality witness) and X2 (stratified test KL).

Usage (from the repository root)::

    python3 papers/contextuality_inductive_bias/utils/run_extensions.py \\
        --config papers/contextuality_inductive_bias/configs/extensions_x1_x2.json

**X1 (characterization).** For every bias-encoding model of the final fair
sweep, the trained model is rebuilt from ``config_snapshot.json`` and
``model_state.pt`` and evaluated on a dense random input set: the 10000 test
strategies plus "corner" strategies near the deterministic profiles (all 27
deterministic profiles and Dirichlet(alpha) rows). A finite input set gives a
lower bound on the eta of the model's full behaviour set. Along each of the six hexagon-vertex
directions, a linear program finds the largest t with c + t (v_i - c) in the
convex hull of the model's behaviours. The witness is eta = min_i t_i: the
largest V_eta inside the behaviour hull. eta > 2/3 means the trained model
is contextual (Theorem 4.1). Models without the bias are outside the
theorem's premise and reported as n/a.

**X2 (advantage evidence).** Per-sample test KL from ``predictions.npz`` of
every final run, averaged within strata of max_k |P_k - 1/2| (the distance of
the true behaviour from the centre). Strata and comparisons are fixed in the
config; the fair baselines ran under identical conditions.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from scipy.optimize import linprog

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parents[1]))
sys.path.insert(0, str(PROJECT_DIR.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_io import describe, load_sweep  # noqa: E402

from contextuality_inductive_bias.lib.data import compress_strategies  # noqa: E402
from contextuality_inductive_bias.lib.rps_game import (  # noqa: E402
    HEXAGON_VERTICES,
    win_probabilities,
)
from contextuality_inductive_bias.lib.runner import build_model  # noqa: E402
from runtime_lib.dtypes import dtype_torch  # noqa: E402

BIASED_KINDS = {"biased_quantum", "bias_fourier", "mlp_biased"}
CENTRE = np.full(3, 0.5)


def corner_strategies(n: int, alpha: float, seed: int) -> np.ndarray:
    """The 27 deterministic profiles plus n Dirichlet(alpha) strategy matrices."""
    deterministic = np.array(
        [np.eye(3)[list(p)] for p in itertools.product(range(3), repeat=3)]
    )
    rng = np.random.default_rng(seed)
    return np.concatenate([deterministic, rng.dirichlet([alpha] * 3, size=(n, 3))])


def is_biased(candidate: dict) -> bool:
    model = candidate["model"]
    return model["kind"] in BIASED_KINDS or (
        model["kind"] == "photonic_merlin" and model["readout"] in ("vertex", "group")
    )


def witness_eta(behaviours: np.ndarray) -> float:
    """Largest eta with V_eta inside conv(behaviours) (behaviours in the zero-sum plane)."""
    n = behaviours.shape[0]
    etas = []
    for vertex in HEXAGON_VERTICES:
        direction = vertex - CENTRE
        # variables: weights w (n) >= 0, t; maximise t s.t. B^T w - t d = c, sum w = 1
        a_eq = np.vstack(
            [np.hstack([behaviours.T, -direction[:, None]]), np.r_[np.ones(n), 0.0]]
        )
        b_eq = np.r_[CENTRE, 1.0]
        result = linprog(
            np.r_[np.zeros(n), -1.0],
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=[(0, None)] * n + [(0, 1)],
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"Witness LP failed: {result.message}")
        etas.append(result.x[-1])
    return float(min(etas))


def run_x1(sweep_dir: Path, cfg: dict) -> dict:
    corners = corner_strategies(
        cfg["corner_samples"], cfg["corner_alpha"], cfg["corner_seed"]
    )
    per_model = defaultdict(list)
    for record in load_sweep(sweep_dir):
        name = describe(record.candidate)
        if not is_biased(record.candidate):
            per_model[name].append(None)
            continue
        run_cfg = json.loads(
            (record.run_dir / "config_snapshot.json").read_text(encoding="utf-8")
        )
        dtype = dtype_torch(run_cfg["dtype"])
        torch.manual_seed(0)
        model = build_model(
            run_cfg["model"],
            torch.Generator().manual_seed(0),
            dtype,
            torch.device("cpu"),
        )
        model.load_state_dict(
            torch.load(record.run_dir / "model_state.pt", weights_only=True)
        )
        model.eval()
        scale = math.pi / 2 if run_cfg["data"]["protocol"] == "paper" else 1.0
        predictions = np.load(record.run_dir / "predictions.npz")
        with torch.no_grad():
            corner_e = model(
                torch.as_tensor(compress_strategies(corners, scale), dtype=dtype)
            ).numpy()
        behaviours = (
            1.0 + np.vstack([predictions["test_expectations"], corner_e])
        ) / 2.0
        if np.abs(behaviours.sum(axis=1) - 1.5).max() > 1e-9:
            raise ValueError(
                f"{record.run_dir}: biased model leaves the zero-sum plane"
            )
        per_model[name].append(witness_eta(behaviours))
    truth = win_probabilities(corners)
    summary = {"truth_eta_on_inputs": witness_eta(truth), "models": {}}
    for name, values in per_model.items():
        if values[0] is None:
            summary["models"][name] = {
                "status": "n/a (no bias: Theorem 4.1 premise fails)"
            }
            continue
        arr = np.array(values)
        summary["models"][name] = {
            "n": int(arr.size),
            "eta_mean": float(arr.mean()),
            "eta_min": float(arr.min()),
            "eta_max": float(arr.max()),
            "contextual_runs": int((arr > 2 / 3 + 1e-9).sum()),
        }
    return summary


def run_x2(sweep_dir: Path, cfg: dict) -> dict:
    edges = cfg["strata_edges"]
    per_model = defaultdict(list)
    counts = None
    for record in load_sweep(sweep_dir):
        predictions = np.load(record.run_dir / "predictions.npz")
        true_win = predictions["test_win_probabilities"]
        model_win = np.clip(
            (1.0 + predictions["test_expectations"]) / 2.0, 1e-8, 1 - 1e-8
        )
        kl = (
            true_win * np.log(np.where(true_win > 0, true_win, 1) / model_win)
            + (1 - true_win)
            * np.log(np.where(true_win < 1, 1 - true_win, 1) / (1 - model_win))
        ).mean(axis=1)
        distance = np.abs(true_win - 0.5).max(axis=1)
        strata = np.digitize(distance, edges[1:-1])
        counts = np.bincount(strata, minlength=len(edges) - 1)
        per_model[describe(record.candidate)].append(
            [float(kl[strata == s].mean()) for s in range(len(edges) - 1)]
        )
    # np.digitize puts every distance >= edges[-2] in the last stratum, so it is closed: [0.2, 0.5].
    labels = [f"[{edges[i]}, {edges[i + 1]})" for i in range(len(edges) - 2)] + [
        f"[{edges[-2]}, {edges[-1]}]"
    ]
    table = {name: np.array(v) for name, v in per_model.items()}
    comparisons = []
    for reference, other in cfg["comparisons"]:
        a, b = table[reference], table[other]
        for s, label in enumerate(labels):
            welch = stats.ttest_ind(a[:, s], b[:, s], equal_var=False).pvalue
            mw = stats.mannwhitneyu(a[:, s], b[:, s], alternative="two-sided").pvalue
            comparisons.append(
                {
                    "reference": reference,
                    "other": other,
                    "stratum": label,
                    "diff": float(a[:, s].mean() - b[:, s].mean()),
                    "welch_p": float(welch),
                    "mw_p": float(mw),
                    "resolved": bool(welch < 0.05 and mw < 0.05),
                }
            )
    return {
        "strata": labels,
        "test_points_per_stratum": counts.tolist(),
        "models": {
            name: {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0, ddof=1).tolist(),
                "n": int(arr.shape[0]),
            }
            for name, arr in table.items()
        },
        "comparisons": comparisons,
    }


def write(x1: dict, x2: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "extension_x1_witness.json").write_text(
        json.dumps(x1, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "extension_x2_stratified.json").write_text(
        json.dumps(x2, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "### Extension X1 (characterization): contextuality witness of trained fair-final models",
        "",
        f"Inputs: test strategies + corner strategies. Ground truth on the same inputs: eta = {x1['truth_eta_on_inputs']:.3f}. Contextual iff eta > 2/3.",
        "",
        "| Model | n | eta mean | eta min | eta max | contextual runs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, info in sorted(x1["models"].items()):
        if "status" in info:
            lines.append(f"| {name} | | | | | {info['status']} |")
        else:
            lines.append(
                f"| {name} | {info['n']} | {info['eta_mean']:.3f} | {info['eta_min']:.3f} | {info['eta_max']:.3f} | {info['contextual_runs']}/{info['n']} |"
            )
    lines += [
        "",
        "### Extension X2 (advantage evidence): test KL by distance of the true behaviour from the centre",
        "",
        "Test points per stratum: "
        + ", ".join(
            f"{s}: {c}" for s, c in zip(x2["strata"], x2["test_points_per_stratum"])
        ),
        "",
        "| Model | " + " | ".join(x2["strata"]) + " |",
        "|---|" + "---:|" * len(x2["strata"]),
    ]
    for name, info in sorted(x2["models"].items(), key=lambda kv: kv[1]["mean"][-1]):
        lines.append(
            f"| {name} | "
            + " | ".join(
                f"{m:.2e} +- {s:.1e}" for m, s in zip(info["mean"], info["std"])
            )
            + " |"
        )
    lines += [
        "",
        "| Reference | Other | Stratum | Diff | Welch p | MW p | Resolved |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for c in x2["comparisons"]:
        lines.append(
            f"| {c['reference']} | {c['other']} | {c['stratum']} | {c['diff']:+.2e} | {c['welch_p']:.1e} | {c['mw_p']:.1e} | {'yes' if c['resolved'] else 'no'} |"
        )
    (out_dir / "extensions_x1_x2.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    repo = PROJECT_DIR.parents[1]
    sweep_dir = repo / cfg["final_sweep_dir"]
    x1 = run_x1(sweep_dir, cfg["x1"])
    x2 = run_x2(sweep_dir, cfg["x2"])
    write(x1, x2, repo / cfg["out_dir"])
    print(
        json.dumps(
            {
                "x1_truth_eta": x1["truth_eta_on_inputs"],
                "x1": {
                    k: v.get("eta_mean", v.get("status"))
                    for k, v in x1["models"].items()
                },
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
