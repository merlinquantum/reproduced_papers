"""E5: compare the QBGU gradient with per-sample batch gradients (paper Eq. 18-20).

Three exact (shot-free) objectives are differentiated at the same random
parameters:

- ``F_batch``: the QBGU fidelity ``|<target| U |input>|^2`` on the superposed input.
- ``J_basis``: the paper's Eq. 20 reading, the data average of the per-sample
  fidelity with the observed basis state, ``sum_{c,o} P(c,o) |<c,o| U |c,0>|^2``.
- ``J_cond``: the index-preserving average fidelity with the conditional target
  state, ``sum_c P(c) |<c,t_c| U |c,0>|^2`` with ``|t_c> = sum_o sqrt(p(o|c)) |o>``.

Eq. 20 claims the QBGU update equals the P-weighted per-sample update. The
script reports the cosine similarity and norm ratio between the gradients, and
the value of each objective at the true conditional and at the argmax
predictor. Training targets are read from a completed sweep.

Usage::

    python papers/QNN_STOCK/utils/analyze_qbgu_gradient.py <sweep_dir> --out results/e5_qbgu_gradient
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_DIR.parents[1]))
sys.path.insert(0, str(PAPER_DIR))

from lib.model import ContextualTask  # noqa: E402
from lib.statevector import basis_state  # noqa: E402
from lib.training import qstl_circuit_factory  # noqa: E402

ANSATZES = (
    ("HEA 4 layers (32)", {"ansatz": "hea", "layers": 4}),
    (
        "context-preserving order 1 (4)",
        {"ansatz": "context_controlled", "interaction_order": 1},
    ),
    (
        "context-preserving order 3 (8)",
        {"ansatz": "context_controlled", "interaction_order": 3},
    ),
)
DATASETS = (("raw", "e3_qstl_original"), ("MA5", "e3_qstl_ma5"))
STEP = 1e-6


def objectives(task: ContextualTask, joint: np.ndarray, circuit) -> dict[str, float]:
    """Exact values of the three objectives for one circuit."""

    n_contexts, levels = joint.shape
    context = joint.sum(axis=1)
    conditional = joint / context[:, None]
    j_basis = 0.0
    j_cond = 0.0
    for c in range(n_contexts):
        output = circuit(basis_state(c * levels, task.n_qubits)).reshape(
            n_contexts, levels
        )
        # Amplitudes that stay on context c; anything that left the context
        # register has zero overlap with a target that carries |c>.
        kept = output[c]
        j_basis += float(np.sum(joint[c] * np.abs(kept) ** 2))
        j_cond += float(context[c] * abs(np.vdot(np.sqrt(conditional[c]), kept)) ** 2)
    return {"F_batch": task.fidelity(circuit), "J_basis": j_basis, "J_cond": j_cond}


def gradients(task, joint, factory, params) -> dict[str, np.ndarray]:
    """Central-difference gradients of the three objectives."""

    grads = {name: np.zeros_like(params) for name in ("F_batch", "J_basis", "J_cond")}
    for index in range(len(params)):
        shift = np.zeros_like(params)
        shift[index] = STEP
        plus = objectives(task, joint, factory(params + shift))
        minus = objectives(task, joint, factory(params - shift))
        for name in grads:
            grads[name][index] = (plus[name] - minus[name]) / (2 * STEP)
    return grads


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def gradient_rows(joints: dict[str, np.ndarray], draws: int, seed: int) -> list[dict]:
    rows = []
    for data_label, joint in joints.items():
        task = ContextualTask(joint)
        for ansatz_label, model_cfg in ANSATZES:
            factory, count = qstl_circuit_factory(model_cfg, task.n_qubits, task.levels)
            rng = np.random.default_rng(seed)
            for draw in range(draws):
                params = rng.uniform(0.0, 2.0 * np.pi, size=count)
                values = objectives(task, joint, factory(params))
                grads = gradients(task, joint, factory, params)
                rows.append(
                    {
                        "data": data_label,
                        "ansatz": ansatz_label,
                        "draw": draw,
                        **values,
                        "cos_F_vs_J_basis": _cosine(grads["F_batch"], grads["J_basis"]),
                        "cos_F_vs_J_cond": _cosine(grads["F_batch"], grads["J_cond"]),
                        "norm_ratio_F_over_J_basis": float(
                            np.linalg.norm(grads["F_batch"])
                            / np.linalg.norm(grads["J_basis"])
                        ),
                        "norm_ratio_F_over_J_cond": float(
                            np.linalg.norm(grads["F_batch"])
                            / np.linalg.norm(grads["J_cond"])
                        ),
                    }
                )
    return rows


def optimum_rows(joints: dict[str, np.ndarray]) -> list[dict]:
    """Objectives of a context-preserving model at two candidate predictors.

    With the context kept and real non-negative amplitudes sqrt(q(o|c)):
    ``F_batch = (sum_c P(c) BC_c)^2``, ``J_basis = sum P(c,o) q(o|c)``,
    ``J_cond = sum_c P(c) BC_c^2`` with ``BC_c = sum_o sqrt(p(o|c) q(o|c))``.
    """

    rows = []
    for data_label, joint in joints.items():
        context = joint.sum(axis=1)
        conditional = joint / context[:, None]
        argmax = np.zeros_like(conditional)
        argmax[np.arange(len(conditional)), conditional.argmax(axis=1)] = 1.0
        for predictor_label, q in (
            ("true conditional q = p", conditional),
            ("argmax predictor", argmax),
        ):
            bc = np.sqrt(conditional * q).sum(axis=1)
            rows.append(
                {
                    "data": data_label,
                    "predictor": predictor_label,
                    "F_batch": float((context @ bc) ** 2),
                    "J_basis": float(np.sum(joint * q)),
                    "J_cond": float(context @ bc**2),
                }
            )
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--asset", default="AAPL")
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    status = json.loads(
        (args.sweep_dir / "sweep_status.json").read_text(encoding="utf-8")
    )
    run_dirs = {
        r["candidate"]["config"]: args.sweep_dir / r["run_dir"]
        for r in status["runs"]
        if r["seed"] == args.seed
    }
    joints = {}
    for label, config in DATASETS:
        tables = json.loads(
            (run_dirs[config] / "distributions.json").read_text(encoding="utf-8")
        )
        joints[label] = np.asarray(tables[args.asset]["train_joint"])

    args.out.mkdir(parents=True, exist_ok=True)
    grad_rows = gradient_rows(joints, args.draws, args.seed)
    _write(args.out / "gradient_comparison.csv", grad_rows)
    opt_rows = optimum_rows(joints)
    _write(args.out / "objective_optima.csv", opt_rows)

    print(
        f"asset={args.asset} draws={args.draws} (random parameters, exact statevector)"
    )
    for label, _ in DATASETS:
        for ansatz_label, _ in ANSATZES:
            subset = [
                r
                for r in grad_rows
                if r["data"] == label and r["ansatz"] == ansatz_label
            ]
            summary = []
            for key in (
                "cos_F_vs_J_basis",
                "cos_F_vs_J_cond",
                "norm_ratio_F_over_J_cond",
            ):
                values = [r[key] for r in subset]
                summary.append(
                    f"{key}={statistics.fmean(values):+.3f} [{min(values):+.3f},{max(values):+.3f}]"
                )
            print(f"{label:4s} {ansatz_label:32s} " + "  ".join(summary))
    for row in opt_rows:
        print(
            f"{row['data']:4s} {row['predictor']:24s} F_batch={row['F_batch']:.4f} "
            f"J_basis={row['J_basis']:.4f} J_cond={row['J_cond']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
