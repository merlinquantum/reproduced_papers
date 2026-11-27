"""Claim-level statistics table from an aggregate written by aggregate_study.py.

Usage (from the paper folder)::

    python utils/report_claims.py --aggregate results/study_extended.json.gz \
        --sizes 8,12,16 --output results/claim_table

Writes ``<output>.md`` and ``<output>.json``. Every number comes from the
aggregate's full-precision summary rows (mean, sample std and 95% t-interval
over seeds); paired quantum-minus-classical differences are aggregated per seed
before averaging. Signs: for MAE and classification error, a positive
difference means the quantum model is worse; for accuracy, a positive
difference means the quantum model is better.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(PROJECT_DIR)]

from lib.result_io import load_json  # noqa: E402

# Set in main() from the embeddings present in the aggregate.
EMBEDDINGS: tuple[str, ...] = ("E1", "E2", "E3")


def index_rows(rows):
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


def lookup(idx, component, quantity, *, embedding, n, train_size, protocol):
    """Summary row for one identity, or None."""
    return idx.get((component, quantity, embedding, n, train_size, protocol))


def fmt(row, digits=3):
    if row is None:
        return "—"
    text = f"{row['mean']:.{digits}f} ± {row['std']:.{digits}f}"
    if row["quantity"].endswith("_minus_best_classical"):
        text += f" [±{row['ci95_half_width']:.{digits}f}]"
    return text


def paper(row, digits=3):
    return (
        "—"
        if row is None or row.get("paper_value") is None
        else f"{row['paper_value']:.{digits}f}"
    )


def table(lines, title, header, body):
    lines += [
        f"### {title}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in body]
    lines.append("")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aggregate", required=True, type=Path)
    parser.add_argument("--sizes", default="8,12,16")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    aggregate = load_json(args.aggregate)
    rows = aggregate["summary"]
    global EMBEDDINGS
    EMBEDDINGS = tuple(sorted({r["embedding"] for r in rows if r["embedding"]}))
    idx = index_rows(rows)
    sizes = [int(s) for s in args.sizes.split(",")]
    protocols = sorted({r["protocol"] for r in rows if r["protocol"]})

    def get(*key):
        return idx.get(key)

    lines = [
        f"# Claim table: {args.aggregate.name}",
        "",
        f"Coverage: {aggregate['coverage']} ({aggregate['observed_records']}/{aggregate['expected_records']} records). "
        "Values are mean ± sample std over seeds; [±x] is the 95% t-interval half-width of a paired difference. "
        "'paper' = journal Source Data.",
        "",
    ]
    machine = {}

    body = []
    for embedding in EMBEDDINGS:
        for n in sizes:
            body.append(
                [
                    embedding,
                    n,
                    fmt(
                        get(
                            "dimension_geometry",
                            "d_quantum_kernel",
                            embedding,
                            n,
                            None,
                            None,
                        ),
                        1,
                    ),
                    paper(
                        get(
                            "dimension_geometry",
                            "d_quantum_kernel",
                            embedding,
                            n,
                            None,
                            None,
                        ),
                        1,
                    ),
                    fmt(
                        get(
                            "dimension_geometry",
                            "d_projected_quantum_kernel",
                            embedding,
                            n,
                            None,
                            None,
                        ),
                        1,
                    ),
                    paper(
                        get(
                            "dimension_geometry",
                            "d_projected_quantum_kernel",
                            embedding,
                            n,
                            None,
                            None,
                        ),
                        1,
                    ),
                    fmt(
                        get(
                            "dimension_geometry",
                            "g_quantum_kernel",
                            embedding,
                            n,
                            None,
                            None,
                        ),
                        2,
                    ),
                    paper(
                        get(
                            "dimension_geometry",
                            "g_quantum_kernel",
                            embedding,
                            n,
                            None,
                            None,
                        ),
                        2,
                    ),
                    fmt(
                        get(
                            "dimension_geometry",
                            "g_projected_quantum_kernel",
                            embedding,
                            n,
                            None,
                            None,
                        ),
                        2,
                    ),
                    paper(
                        get(
                            "dimension_geometry",
                            "g_projected_quantum_kernel",
                            embedding,
                            n,
                            None,
                            None,
                        ),
                        2,
                    ),
                ]
            )
    table(
        lines,
        "C2–C4: effective dimension and geometric difference (Fig. 1a)",
        [
            "emb",
            "n",
            "d_QK",
            "paper",
            "d_PQK",
            "paper",
            "g_QK",
            "paper",
            "g_PQK",
            "paper",
        ],
        body,
    )

    for protocol in protocols:
        body = []
        for embedding in EMBEDDINGS:
            for n in sizes:
                key = functools.partial(
                    lookup,
                    idx,
                    "quantum_dataset",
                    embedding=embedding,
                    n=n,
                    train_size=600,
                    protocol=protocol,
                )
                if key("best_classical") is None:
                    continue
                body.append(
                    [
                        embedding,
                        n,
                        fmt(key("quantum_kernel")),
                        paper(key("quantum_kernel")),
                        fmt(key("projected_quantum_kernel")),
                        paper(key("projected_quantum_kernel")),
                        fmt(key("best_classical")),
                        paper(key("best_classical")),
                        fmt(key("quantum_kernel_minus_best_classical")),
                        fmt(key("projected_quantum_kernel_minus_best_classical")),
                    ]
                )
        table(
            lines,
            f"C1/C10: quantum datasets, test MAE, N=600, protocol `{protocol}` (Fig. 1b)",
            [
                "emb",
                "n",
                "QK",
                "paper",
                "PQK",
                "paper",
                "best classical",
                "paper",
                "QK − best",
                "PQK − best",
            ],
            body,
        )

        body = []
        for n in sizes:
            best = get("classical_dataset", "best_classical", None, n, 600, protocol)
            if best is None:
                continue
            cells = [n, fmt(best), paper(best)]
            for embedding in EMBEDDINGS:
                for family in ("quantum_kernel", "projected_quantum_kernel"):
                    row = get(
                        "classical_dataset",
                        f"{family}_{embedding}",
                        None,
                        n,
                        600,
                        protocol,
                    )
                    cells += [f"{fmt(row)} (paper {paper(row)})"]
            body.append(cells)
        table(
            lines,
            f"C10: Dataset (C) classification error, N=600, protocol `{protocol}`",
            [
                "n",
                "best classical",
                "paper",
                *[f"{kind} {e}" for e in EMBEDDINGS for kind in ("QK", "PQK")],
            ],
            body,
        )

        for component, reference in (
            ("engineered_pqk", "C5/C6 (Fig. 2)"),
            ("engineered_qk", "C7 (Fig. S1)"),
        ):
            body = []
            for embedding in EMBEDDINGS:
                for train_size in (100, 600):
                    for n in sizes:
                        key = functools.partial(
                            lookup,
                            idx,
                            component,
                            embedding=embedding,
                            n=n,
                            train_size=train_size,
                            protocol=protocol,
                        )
                        if key("best_classical") is None:
                            continue
                        difference = key(
                            "projected_quantum_kernel_minus_best_classical"
                        )
                        body.append(
                            [
                                embedding,
                                train_size,
                                n,
                                fmt(key("projected_quantum_kernel")),
                                paper(key("projected_quantum_kernel")),
                                fmt(key("best_classical")),
                                paper(key("best_classical")),
                                fmt(key("quantum_kernel")),
                                paper(key("quantum_kernel")),
                                fmt(difference),
                                "yes"
                                if difference
                                and difference["mean"] - difference["ci95_half_width"]
                                > 0
                                else "no",
                            ]
                        )
            table(
                lines,
                f"{reference}: {component} accuracy, protocol `{protocol}`",
                [
                    "emb",
                    "N",
                    "n",
                    "PQK",
                    "paper",
                    "best classical",
                    "paper",
                    "QK",
                    "paper",
                    "PQK − best",
                    "PQK > best (95% CI)",
                ],
                body,
            )

    bound_pairs = []
    for embedding in EMBEDDINGS:
        for n in range(3, 13):
            bound = get("bound", "prediction_error_bound", embedding, n, 600, None)
            error = get("quantum_dataset", "best_classical", embedding, n, 600, "paper")
            if bound and error:
                bound_pairs.append((bound["mean"], error["mean"], embedding, n))
    if bound_pairs:
        rho, p_value = stats.spearmanr(
            [b for b, *_ in bound_pairs], [e for _, e, *_ in bound_pairs]
        )
        above = sum(b >= e for b, e, *_ in bound_pairs)
        lines += [
            "### C9: bound vs best classical error (Fig. S3, n=3–12, all embeddings)",
            "",
            f"Spearman rho = {rho:.3f} (p = {p_value:.2g}) over {len(bound_pairs)} (embedding, n) points; "
            f"bound ≥ error at {above}/{len(bound_pairs)} points.",
            "",
        ]
        machine["c9"] = {
            "spearman_rho": rho,
            "p_value": p_value,
            "points": len(bound_pairs),
            "bound_above_error": above,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".md").write_text("\n".join(lines))
    args.output.with_suffix(".json").write_text(json.dumps(machine, indent=2) + "\n")
    print(f"wrote {args.output.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
