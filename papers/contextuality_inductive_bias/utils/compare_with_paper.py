"""Compare Fig. 8 reproduction results with the values read off the paper.

Usage (from the repository root)::

    python3 papers/contextuality_inductive_bias/utils/compare_with_paper.py \\
        --results papers/contextuality_inductive_bias/results/fig8_results.json \\
        --paper papers/contextuality_inductive_bias/results/paper_values_read_off.json \\
        --out papers/contextuality_inductive_bias/results/fig8_vs_paper.md

The paper's four curves are matched to reproduced candidates for each
protocol under two readings of the published settings (D2/D3): "paper text"
(quantum L = 2, generic with Fig. 10 CNOTs) and "author code" (quantum L = 1,
generic with code CNOTs). The relative difference is (ours - paper) / paper.
Paper values carry about +-5% read-off uncertainty (KL) and +-0.002 (NLL).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

READINGS = {
    "paper text (L=2, Fig. 10 CNOTs)": {
        "quantum (biased)": "quantum (biased) L=2",
        "quantum (no bias)": "quantum (no bias, Fig. 10 CNOTs) L=2",
        "surrogate L=1": "surrogate L=1",
        "surrogate L=2": "surrogate L=2",
    },
    "author code (L=1, code CNOTs)": {
        "quantum (biased)": "quantum (biased) L=1",
        "quantum (no bias)": "quantum (no bias, code CNOTs) L=1",
        "surrogate L=1": "surrogate L=1",
        "surrogate L=2": "surrogate L=2",
    },
}
METRICS = (
    ("final_test_kl", "final test KL"),
    ("min_test_kl", "min test KL"),
    ("final_train_nll", "final train NLL"),
)


def compare(results_path: Path, paper_path: Path, out: Path) -> str:
    """Write and return the Markdown comparison."""
    results = json.loads(results_path.read_text(encoding="utf-8"))
    paper = json.loads(paper_path.read_text(encoding="utf-8"))["fig8"]
    table = {
        (row["protocol"], row["model"]): row
        for row in results["candidates"]
        if row["l2"] == 0.0 and row["train"] == "author"
    }
    lines = [
        "### Fig. 8: reproduced vs paper (paper values read off the figure, about +-5% on KL)",
        "",
        "| Protocol | Reading | Paper curve | Metric | Paper | Ours (mean +- std, n) | Rel. diff |",
        "|---|---|---|---|---:|---|---:|",
    ]
    for protocol in ("code", "paper", "fork"):
        for reading, mapping in READINGS.items():
            for paper_name, ours_name in mapping.items():
                row = table[(protocol, ours_name)]
                for key, label in METRICS:
                    reported = paper[key].get(paper_name)
                    if reported is None:
                        continue
                    ours = row[f"{key}_mean"]
                    std = row[f"{key}_std"]
                    lines.append(
                        f"| {protocol} | {reading} | {paper_name} | {label} | {reported:.4g} | "
                        f"{ours:.4g} +- {std:.1g} (n={row['n']}) | {(ours - reported) / reported:+.1%} |"
                    )
    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    compare(args.results, args.paper, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
