"""Condense raw run rows into the curated `results/summary.json`.

The raw JSONL is not committed (see `.gitignore`); this summary is, because it is
the evidence behind every number in `README.md`. Regenerate with

    python utils/summarise.py results/*.jsonl --out results/summary.json

Aggregates per arm and recomputes the paired statistics quoted in the README, so
a reader can check a claim without the raw rows and a future run can be compared
against this one directly.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as st
from pathlib import Path


def arm_of(row):
    """Name an arm the way the README does."""
    if row["method"] != "bbs":
        return row["method"]
    if row.get("freeze_theta"):
        return "boson (theta frozen)"
    source = row.get("source") or "boson"
    if source == "bernoulli" and row.get("rate_scale", 1.0) != 1.0:
        return f"bernoulli@{row['rate_scale']}"
    return {
        "boson": "boson",
        "shuffled_boson": "shuffled boson",
        "distinguishable": "distinguishable",
        "bernoulli": "bernoulli@1.0",
    }[source]


def summarise(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[
            (row["family"], row["m"], row.get("topology", "chain"), row["config"])
        ].append(row)

    arms = []
    for (family, m, topology, config), members in sorted(groups.items()):
        errors = [r["relative_error_percent"] for r in members]
        arms.append(
            {
                "config": config,
                "family": family,
                "m": m,
                "topology": topology,
                "arm": arm_of(members[0]),
                "instances": len(members),
                "percent_optimal": round(
                    100 * sum(r["found_optimum"] for r in members) / len(members), 2
                ),
                "mean_percent_error": round(st.mean(errors), 5),
                "std_percent_error": round(st.stdev(errors), 5)
                if len(members) > 1
                else 0.0,
                "mean_click_density": round(
                    st.mean(r["mean_clicks"] for r in members) / m, 4
                ),
                "candidate_budget": members[0]["budget"],
                "space_coverage": members[0]["space_coverage"],
                "median_seconds_per_instance": round(
                    st.median(r["seconds"] for r in members), 3
                ),
            }
        )
    return arms


def paired(rows, config_a, config_b):
    """McNemar on found-optimum and a paired t on relative error, by instance."""
    index = collections.defaultdict(dict)
    for row in rows:
        index[row["config"]][row["instance_seed"]] = row
    left, right = index.get(config_a, {}), index.get(config_b, {})
    shared = sorted(set(left) & set(right))
    if len(shared) < 2:
        return None
    only_a = sum(
        1 for i in shared if left[i]["found_optimum"] and not right[i]["found_optimum"]
    )
    only_b = sum(
        1 for i in shared if right[i]["found_optimum"] and not left[i]["found_optimum"]
    )
    discordant = only_a + only_b
    z = (abs(only_a - only_b) - 1) / math.sqrt(discordant) if discordant else 0.0
    if only_a > only_b:
        z = -z
    gaps = [
        right[i]["relative_error_percent"] - left[i]["relative_error_percent"]
        for i in shared
    ]
    spread = st.stdev(gaps)
    return {
        "a": config_a,
        "b": config_b,
        "instances": len(shared),
        "a_only_optimal": only_a,
        "b_only_optimal": only_b,
        "mcnemar_z": round(z, 3),
        "mean_error_gap_percent": round(st.mean(gaps), 5),
        "paired_t": round(st.mean(gaps) / (spread / math.sqrt(len(gaps))), 3)
        if spread
        else None,
        "reading": "positive z and negative gap favour b",
    }


COMPARISONS = [
    ("knapsack_m25_original.json", "knapsack_m25_source_shuffled_boson.json"),
    ("knapsack_m25_original.json", "knapsack_m25_source_distinguishable.json"),
    ("knapsack_m25_original.json", "knapsack_m25_source_bernoulli.json"),
    ("knapsack_m30_original.json", "knapsack_m30_source_shuffled_boson.json"),
    ("knapsack_m30_original.json", "knapsack_m30_source_distinguishable.json"),
    ("knapsack_m30_original.json", "knapsack_m30_source_bernoulli.json"),
    ("knapsack_m30_original.json", "knapsack_m30_density_r087.json"),
    ("knapsack_m30_density_r087.json", "knapsack_m30_density_r15.json"),
    ("knapsack_m25_frozen_theta.json", "knapsack_m25_original.json"),
    ("knapsack_m30_frozen_theta.json", "knapsack_m30_original.json"),
    ("knapsack_m25_original.json", "knapsack_m25_shift_pi6.json"),
    ("knapsack_m25_original.json", "knapsack_m25_shift_scale05.json"),
    ("knapsack_m20_original.json", "knapsack_m20_topology_loop.json"),
    ("knapsack_m25_original.json", "knapsack_m25_topology_loop.json"),
    ("tsp_m29_original.json", "tsp_m29_source_shuffled_boson.json"),
    ("tsp_m29_original.json", "tsp_m29_source_distinguishable.json"),
    ("tsp_m29_original.json", "tsp_m29_source_bernoulli.json"),
    # extension beyond the paper: does the density result survive as coverage collapses?
    ("scaling_m34_source_boson.json", "scaling_m34_source_distinguishable.json"),
    ("scaling_m34_source_boson.json", "scaling_m34_source_bernoulli.json"),
    ("scaling_m34_source_boson.json", "scaling_m34_source_bernoulli_dense.json"),
    ("scaling_m38_source_boson.json", "scaling_m38_source_distinguishable.json"),
    ("scaling_m38_source_boson.json", "scaling_m38_source_bernoulli.json"),
    ("scaling_m38_source_boson.json", "scaling_m38_source_bernoulli_dense.json"),
    ("scaling_m42_source_boson.json", "scaling_m42_source_distinguishable.json"),
    ("scaling_m42_source_boson.json", "scaling_m42_source_bernoulli.json"),
    ("scaling_m42_source_boson.json", "scaling_m42_source_bernoulli_dense.json"),
    ("scaling_m34_source_bernoulli.json", "scaling_m34_source_bernoulli_dense.json"),
    ("scaling_m38_source_bernoulli.json", "scaling_m38_source_bernoulli_dense.json"),
    ("scaling_m42_source_bernoulli.json", "scaling_m42_source_bernoulli_dense.json"),
    ("scaling_m48_source_bernoulli.json", "scaling_m48_source_bernoulli_dense.json"),
    ("scaling_m54_source_bernoulli.json", "scaling_m54_source_bernoulli_dense.json"),
    ("scaling_m60_source_bernoulli.json", "scaling_m60_source_bernoulli_dense.json"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", default="results/summary.json")
    args = parser.parse_args()

    rows, sources = [], []
    for path in args.inputs:
        lines = [json.loads(line) for line in Path(path).open() if line.strip()]
        rows.extend(lines)
        sources.append({"file": Path(path).name, "rows": len(lines)})

    # a key appears once per arm and instance; later files win on a rerun
    unique = {r["key"]: r for r in rows}
    rows = list(unique.values())

    summary = {
        "paper": "arXiv:2510.08274",
        "generated_from": sources,
        "unique_runs": len(rows),
        "note": "Raw rows are not committed; regenerate with utils/summarise.py. "
        "Every BBS number quoted in README.md comes from this file; the classical "
        "baselines are in baselines.json.",
        "arms": summarise(rows),
        "paired_comparisons": [
            c for c in (paired(rows, a, b) for a, b in COMPARISONS) if c
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"{len(rows)} unique runs -> {len(summary['arms'])} arms, "
        f"{len(summary['paired_comparisons'])} paired comparisons -> {args.out}"
    )


if __name__ == "__main__":
    main()
