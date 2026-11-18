"""Read-only recomputation of E4/E5 descriptive statistics from raw metrics.json.

Shares no code with compute_statistics.py. Run from the paper folder:
``python -I utils/verify_e4_e5_statistics.py``.
"""

import glob
import json
import os

SETS = {
    "paper-table": {
        "pqc": ["cartpole_optimal/runs/e2e3_cartpole_optimal_full_original"]
        + [
            f"e4_pqc_depth/runs/e4_cartpole_pqc_l{layers}_original"
            for layers in (10, 15, 20, 25, 30)
        ],
        "nn": [
            f"e4_nn_width/runs/e4_cartpole_mlp_{w}_classical"
            for w in ("10_10", "15_15", "20_20", "24_24", "30_30", "64_64")
        ],
        "stats": "results/statistics.json",
    },
    "code-faithful": {
        "pqc": ["cf_cartpole_optimal/runs/e2e3_cartpole_optimal_full_codefaithful"]
        + [
            f"cf_e4_pqc_depth/runs/e4_cartpole_pqc_l{layers}_codefaithful"
            for layers in (10, 15, 20, 25, 30)
        ],
        "nn": [
            f"cf_e4_nn_width/runs/e4_cartpole_mlp_{w}_codefaithful"
            for w in ("10_10", "15_15", "20_20", "24_24", "30_30", "64_64")
        ],
        "stats": "results/statistics_codefaithful.json",
    },
}
ok = True
for name, spec in SETS.items():
    stats = json.load(open(spec["stats"]))
    print(f"== {name}")
    for family in ("pqc", "nn"):
        rows = []
        for rel in spec[family]:
            runs = glob.glob(
                os.path.join("outdir/batches", rel, "run_*", "metrics.json")
            )
            assert len(runs) == 1, (rel, runs)
            m = json.load(open(runs[0]))
            agents = m["agents"]
            assert len(agents) == 10 and sorted(a["seed"] for a in agents) == list(
                range(10)
            )
            cens = [
                a["solved_at_episode_index"] if a["solved"] else m["max_episodes"]
                for a in agents
            ]
            mean = sum(cens) / len(cens)
            fastest = min(
                [a["solved_at_episode_index"] for a in agents if a["solved"]],
                default=None,
            )
            rows.append((rel.split("/")[-1], mean, fastest))
            key = rel.split("/")[-1].replace(
                "_codefaithful", "_classical" if family == "nn" else "_original"
            )
            s = stats["configs"][key]
            match = abs(s["mean_censored"] - mean) < 1e-9 and s["best_agent"] == fastest
            ok &= match
            print(
                f"  {rows[-1][0]:42s} mean_censored={mean:.4f} fastest={fastest}  matches statistics file: {match}"
            )
        lo = min(rows, key=lambda r: r[1])
        fa = min(rows, key=lambda r: r[2] if r[2] is not None else 1e9)
        print(
            f"  {family}: lowest mean {lo[0]} ({lo[1]:.4f}); fastest agent {fa[2]} in {fa[0]}"
        )
print("ALL MATCH" if ok else "MISMATCH")
