"""Evaluate the predeclared Phase 4 preservation criteria P1-P5 (LOG.md plan).

Usage (from the paper folder)::

    python utils/evaluate_photonic.py --photonic results/study_photonic_extended.json.gz \
        --gate results/study_extended.json.gz --output results/photonic_preservation_extended

Criteria (one verdict per photonic embedding; full-precision values from the
aggregates; CV protocol where a protocol applies):

* P1 (C3): d_FQK increases with n (Spearman rho(n, d_FQK) > 0.9) and
  d_PQK <= 0.1 N at every n (N = 800 subsample points).
* P2 (C4): mean g_PQK > mean g_FQK at every n >= 8 with an FQK.
* P3 (C1): at the largest n with an FQK, FQK MAE - best classical MAE > 0 on the
  photonic quantum dataset, with the paired 95% CI excluding 0.
* P4 (C5): on PQK-engineered labels at N = 600, PQK - best classical > 0 with
  the 95% CI excluding 0 at every n >= 8; and the embedding with the larger mean
  g_PQK (n >= 8) has the larger mean advantage (n >= 8).
* P5 (C9): Spearman rho(bound, best classical MAE) > 0.8 over all
  (embedding, n <= 12) points, and bound >= error at >= 90% of them.

Also writes a side-by-side table of photonic (PA, PB) and gate (E2, E3) values at
matched n for d, g, the C1 gap and the C5 advantage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(PROJECT_DIR)]

from lib.result_io import load_json  # noqa: E402

SUBSAMPLE = 800
CV = "cv"


def index(rows):
    return {
        (
            r["component"],
            r["quantity"],
            r["embedding"],
            r["n"],
            r["train_size"],
            r["protocol"],
        ): r
        for r in rows
    }


def series(idx, component, quantity, embedding, train_size, protocol):
    points = sorted(
        (key[3], row)
        for key, row in idx.items()
        if key[0] == component
        and key[1] == quantity
        and key[2] == embedding
        and key[4] == train_size
        and key[5] == protocol
    )
    return dict(points)


def evaluate(photonic_rows):
    idx = index(photonic_rows)
    embeddings = sorted({r["embedding"] for r in photonic_rows if r["embedding"]})
    verdicts, details = {}, {}
    for e in embeddings:
        d_fqk = series(idx, "dimension_geometry", "d_quantum_kernel", e, None, None)
        d_pqk = series(
            idx, "dimension_geometry", "d_projected_quantum_kernel", e, None, None
        )
        g_fqk = series(idx, "dimension_geometry", "g_quantum_kernel", e, None, None)
        g_pqk = series(
            idx, "dimension_geometry", "g_projected_quantum_kernel", e, None, None
        )
        rho = (
            stats.spearmanr(list(d_fqk), [r["mean"] for r in d_fqk.values()])[0]
            if len(d_fqk) > 2
            else float("nan")
        )
        p1 = rho > 0.9 and all(r["mean"] <= 0.1 * SUBSAMPLE for r in d_pqk.values())
        p2_sizes = [n for n in g_fqk if n >= 8]
        p2 = bool(p2_sizes) and all(
            g_pqk[n]["mean"] > g_fqk[n]["mean"] for n in p2_sizes
        )
        c1 = series(
            idx, "quantum_dataset", "quantum_kernel_minus_best_classical", e, 600, CV
        )
        p3_n = max(c1) if c1 else None
        p3 = p3_n is not None and c1[p3_n]["mean"] - c1[p3_n]["ci95_half_width"] > 0
        c5 = series(
            idx,
            "engineered_pqk",
            "projected_quantum_kernel_minus_best_classical",
            e,
            600,
            CV,
        )
        c5_sizes = [n for n in c5 if n >= 8]
        p4a = bool(c5_sizes) and all(
            c5[n]["mean"] - c5[n]["ci95_half_width"] > 0 for n in c5_sizes
        )
        details[e] = {
            "P1": {
                "spearman_n_dFQK": rho,
                "d_FQK_max_n": max(d_fqk) if d_fqk else None,
                "d_FQK_at_max_n_over_N": d_fqk[max(d_fqk)]["mean"] / SUBSAMPLE
                if d_fqk
                else None,
                "max_d_PQK": max(r["mean"] for r in d_pqk.values()),
            },
            "P2": {n: (g_pqk[n]["mean"], g_fqk[n]["mean"]) for n in p2_sizes},
            "P3": {
                "n": p3_n,
                "mean": c1[p3_n]["mean"] if p3_n else None,
                "ci95": c1[p3_n]["ci95_half_width"] if p3_n else None,
            },
            "P4": {n: (c5[n]["mean"], c5[n]["ci95_half_width"]) for n in c5_sizes},
            "mean_g_PQK_n_ge_8": sum(g_pqk[n]["mean"] for n in g_pqk if n >= 8)
            / max(1, len([n for n in g_pqk if n >= 8])),
            "mean_advantage_n_ge_8": sum(c5[n]["mean"] for n in c5_sizes)
            / max(1, len(c5_sizes)),
        }
        verdicts[e] = {"P1": p1, "P2": p2, "P3": p3, "P4_ci": p4a}
    ordering = sorted(embeddings, key=lambda e: details[e]["mean_g_PQK_n_ge_8"])
    by_advantage = sorted(embeddings, key=lambda e: details[e]["mean_advantage_n_ge_8"])
    p4_order = ordering[-1] == by_advantage[-1]
    pairs = []
    for e in embeddings:
        bound = series(idx, "bound", "prediction_error_bound", e, 600, None)
        error = series(idx, "quantum_dataset", "best_classical", e, 600, "paper")
        pairs += [
            (bound[n]["mean"], error[n]["mean"])
            for n in bound
            if n in error and n <= 12
        ]
    rho5 = (
        stats.spearmanr([b for b, _ in pairs], [x for _, x in pairs])[0]
        if len(pairs) > 2
        else float("nan")
    )
    above = sum(b >= x for b, x in pairs) / len(pairs) if pairs else float("nan")
    # None = not applicable: the bound is only computed for n <= 12, so an
    # aggregate restricted to larger n has no point to test.
    p5 = (rho5 > 0.8 and above >= 0.9) if pairs else None
    return (
        verdicts,
        details,
        {
            "P4_order": p4_order,
            "g_order": ordering,
            "advantage_order": by_advantage,
            "P5": p5,
            "P5_spearman": rho5,
            "P5_fraction_above": above,
            "P5_points": len(pairs),
        },
    )


def comparison_rows(photonic_rows, gate_rows, sizes):
    p, g = index(photonic_rows), index(gate_rows)
    out = []
    for n in sizes:
        for label, idx, e in (
            ("photonic", p, "PA"),
            ("photonic", p, "PB"),
            ("gate", g, "E2"),
            ("gate", g, "E3"),
        ):

            def mean(component, quantity, train_size, protocol, idx=idx, e=e, n=n):
                row = idx.get((component, quantity, e, n, train_size, protocol))
                return (
                    None if row is None else (row["mean"], row.get("ci95_half_width"))
                )

            out.append(
                {
                    "n": n,
                    "family": label,
                    "embedding": e,
                    "d_FQK": mean("dimension_geometry", "d_quantum_kernel", None, None),
                    "d_PQK": mean(
                        "dimension_geometry", "d_projected_quantum_kernel", None, None
                    ),
                    "g_FQK": mean("dimension_geometry", "g_quantum_kernel", None, None),
                    "g_PQK": mean(
                        "dimension_geometry", "g_projected_quantum_kernel", None, None
                    ),
                    "C1_FQK_minus_best_MAE_cv": mean(
                        "quantum_dataset",
                        "quantum_kernel_minus_best_classical",
                        600,
                        CV,
                    ),
                    "C5_PQK_minus_best_acc_cv_N600": mean(
                        "engineered_pqk",
                        "projected_quantum_kernel_minus_best_classical",
                        600,
                        CV,
                    ),
                }
            )
    return out


def fmt(value, digits=3, difference=False):
    """Mean, with the 95% CI half-width only for paired differences."""
    if value is None:
        return "—"
    mean, half = value
    if difference and half is not None and half == half:
        return f"{mean:+.{digits}f} [±{half:.{digits}f}]"
    return f"{mean:.{digits}f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photonic", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--sizes", default="4,8,12")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    photonic = load_json(args.photonic)
    gate = load_json(args.gate)
    verdicts, details, global_ = evaluate(photonic["summary"])
    sizes = [int(s) for s in args.sizes.split(",")]
    table = comparison_rows(photonic["summary"], gate["summary"], sizes)
    word = lambda ok: (  # noqa: E731
        "N/A (no bound points at n <= 12)"
        if ok is None
        else "PRESERVED"
        if ok
        else "NOT PRESERVED"
    )
    lines = [
        f"# Phase 4 preservation criteria [{args.photonic.name}; coverage {photonic['coverage']}]",
        "",
        "| Criterion | " + " | ".join(verdicts) + " |",
        "|---|" + "---|" * len(verdicts),
    ]
    for key in ("P1", "P2", "P3", "P4_ci"):
        lines.append(
            f"| {key} | " + " | ".join(word(verdicts[e][key]) for e in verdicts) + " |"
        )
    lines += [
        "",
        f"- P4 ordering (larger g_PQK has larger advantage): {word(global_['P4_order'])} "
        f"(g order {global_['g_order']}, advantage order {global_['advantage_order']})",
        f"- P5 (bound tracks error): {word(global_['P5'])} (Spearman {global_['P5_spearman']:.3f}, "
        f"bound >= error at {global_['P5_fraction_above']:.0%} of {global_['P5_points']} points)",
        "",
        "## Details",
        "",
        "```json",
        json.dumps(details, indent=2, default=str),
        "```",
        "",
        "## Photonic vs gate at matched n (mean; [±95% CI] for paired differences; CV protocol)",
        "",
        "| n | family | emb | d_FQK | d_PQK | g_FQK | g_PQK | FQK − best MAE | PQK − best acc (eng., N=600) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table:
        lines.append(
            f"| {r['n']} | {r['family']} | {r['embedding']} | {fmt(r['d_FQK'], 1)} | {fmt(r['d_PQK'], 1)} | "
            f"{fmt(r['g_FQK'], 2)} | {fmt(r['g_PQK'], 2)} | {fmt(r['C1_FQK_minus_best_MAE_cv'], difference=True)} | "
            f"{fmt(r['C5_PQK_minus_best_acc_cv_N600'], difference=True)} |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "verdicts": verdicts,
                "global": global_,
                "details": details,
                "comparison": table,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print("\n".join(lines[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
